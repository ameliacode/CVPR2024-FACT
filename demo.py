import os

import numpy as np
import torch
from torch.nn.functional import softmax

from configs.utils import setup_cfg
from models.blocks import FACT

# Load external YAML config
cfg = setup_cfg(["./configs/config.yaml"], None)  # ✅ pass as list

ckpt_path = "./ckpts/best_ckpt.pth"

torch.cuda.set_device(cfg.aux.gpu)
net = FACT(cfg, 54, 31).cuda()
state = torch.load(ckpt_path, map_location="cpu")

if "frame_pe.pe" in state:
    del state["frame_pe.pe"]

net.load_state_dict(state, strict=False)
net.eval().cuda()
seq_np = np.load(
    "./data/fs_tas/features/pose3d/men_olympic_short_program_2010_01_00011475_00015700.npy"
)
assert seq_np.shape[1] == 54

seq_tensor = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).cuda()
label = torch.tensor([30 for _ in range(seq_np.shape[0])], device="cuda:0").unsqueeze(0)

with torch.no_grad():
    print(seq_tensor.size(), label.size())
    output = net(
        seq_tensor,
        label,
        compute_loss=False,
    )

print([i for i in output[0]["pred"] if i != 30])
# If output is list: [logits]
# logits = output[0]
# probs = softmax(logits, dim=-1).cpu().numpy()
# pred_labels = np.argmax(probs, axis=1)

# sprint(pred_labels)
