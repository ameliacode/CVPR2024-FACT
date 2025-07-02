import argparse
import json
import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from configs.utils import setup_cfg
from home import get_project_base
from models.loss import MatchCriterion
from utils.dataset import DataLoader, create_dataset
from utils.evaluate import Checkpoint
from utils.train_tools import compute_null_weight, resume_ckpt, save_results


def evaluate(global_step, net, testloader, savedir):
    ckpt = Checkpoint(
        global_step + 1,
        bg_class=([] if net.cfg.eval_bg else testloader.dataset.bg_class),
    )
    net.eval()
    with torch.no_grad():
        for vnames, seq_list, train_label_list, eval_label_list in testloader:
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]
            video_saves = net(seq_list, train_label_list)
            save_results(ckpt, vnames, eval_label_list, video_saves)
    net.train()
    ckpt.compute_metrics()
    ckpt.save(os.path.join(savedir, f"{global_step + 1}.pth"))
    return ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", dest="cfg_file", nargs="*", default=[])
    parser.add_argument(
        "--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER
    )
    args = parser.parse_args()

    BASE = get_project_base()
    cfg = setup_cfg(args.cfg_file, args.set_cfgs)
    torch.cuda.set_device("cuda:%d" % cfg.aux.gpu)

    if cfg.aux.debug:
        seed = 1
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    logdir = os.path.join(BASE, cfg.aux.logdir)
    ckptdir = os.path.join(logdir, "ckpts")
    savedir = os.path.join(logdir, "saves")
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)

    with open(os.path.join(logdir, "args.json"), "w") as f:
        json.dump(cfg, f, indent=True)

    dataset, test_dataset = create_dataset(cfg)
    trainloader = DataLoader(
        dataset if not cfg.aux.debug else test_dataset, cfg.batch_size, shuffle=True
    )
    testloader = DataLoader(test_dataset, cfg.batch_size, shuffle=False)

    if cfg.dataset == "epic":
        from models.blocks_SepVerbNoun import FACT

        net = FACT(cfg, dataset.input_dimension, 98, 301)
    else:
        from models.blocks import FACT

        net = FACT(cfg, dataset.input_dimension, dataset.nclasses)

    if cfg.Loss.nullw == -1:
        compute_null_weight(cfg, dataset)
    net.mcriterion = MatchCriterion(cfg, dataset.nclasses, dataset.bg_class)

    global_step, ckpt_file = resume_ckpt(cfg, logdir)
    if ckpt_file is not None:
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        if "frame_pe.pe" in checkpoint:
            del checkpoint["frame_pe.pe"]
        if "action_pe.pe" in checkpoint:
            del checkpoint["action_pe.pe"]
        net.load_state_dict(checkpoint, strict=False)

    net.cuda()

    if cfg.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

    start_epoch = global_step // len(trainloader)
    ckpt = Checkpoint(
        -1, bg_class=([] if cfg.eval_bg else dataset.bg_class), eval_edit=False
    )
    best_ckpt, best_metric = None, 0

    for eidx in tqdm(range(start_epoch, cfg.epoch)):
        for vnames, seq_list, train_label_list, eval_label_list in trainloader:
            seq_list = [s.cuda() for s in seq_list]
            train_label_list = [s.cuda() for s in train_label_list]

            optimizer.zero_grad()
            loss, video_saves = net(seq_list, train_label_list, compute_loss=True)
            loss.backward()

            if cfg.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_grad_norm)

            optimizer.step()
            save_results(ckpt, vnames, eval_label_list, video_saves)

            if (global_step + 1) % cfg.aux.eval_every == 0:
                test_ckpt = evaluate(global_step, net, testloader, savedir)
                if test_ckpt.metrics["F1@0.50"] >= best_metric:
                    best_ckpt = test_ckpt
                    best_metric = test_ckpt.metrics["F1@0.50"]
                    torch.save(net.state_dict(), os.path.join(logdir, "best_ckpt.pth"))
            global_step += 1

        if cfg.lr_decay > 0 and (eidx + 1) % cfg.lr_decay == 0:
            for g in optimizer.param_groups:
                g["lr"] *= 0.1

    if best_ckpt is not None:
        best_ckpt.eval_edit = True
        best_ckpt.compute_metrics()
        best_ckpt.save(os.path.join(logdir, "best_metrics.pth"))

    open(os.path.join(logdir, "FINISH_PROOF"), "w").close()
