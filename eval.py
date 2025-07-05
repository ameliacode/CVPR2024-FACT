import numpy as np
import torch
from tqdm import tqdm

from configs.utils import get_cfg_defaults
from utils import utils
from utils.dataset import DataLoader, create_dataset
from utils.evaluate import Checkpoint, Video
from utils.train_tools import save_results

n_splits = 1

cfg = get_cfg_defaults()
cfg.merge_from_file(f"./configs/config.yaml")

ckpts = []
for split in range(1, n_splits + 1):
    cfg.split = f"split{split}"
    dataset, test_dataset = create_dataset(cfg)

    from models.blocks import FACT

    model = FACT(cfg, dataset.input_dimension, dataset.nclasses)
    weights = f"./ckpts/best_ckpt.pth"
    weights = torch.load(weights, map_location="cpu")
    if "frame_pe.pe" in weights:
        del weights["frame_pe.pe"]
    model.load_state_dict(weights, strict=False)
    model.eval().cuda()

ckpt = Checkpoint(-1, bg_class=([] if cfg.eval_bg else dataset.bg_class))
loader = DataLoader(test_dataset, 1, shuffle=False)
for vname, batch_seq, train_label_list, eval_label in tqdm(loader):
    shape = np.array(train_label_list).shape
    seq_list = [s.cuda() for s in batch_seq]
    train_label_list = [s.cuda() for s in train_label_list]
    video_saves = model(
        seq_list,
        [torch.tensor([0 for _ in range(shape[-1])], device="cuda:0")],
    )
    save_results(ckpt, vname, eval_label, video_saves)

ckpt.compute_metrics()
ckpts.append(ckpt)

print(utils.easy_reduce([c.metrics for c in ckpts]))
