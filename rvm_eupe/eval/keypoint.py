# Copyright (c) 2026 — RVM-EUPE authors.
# JHMDB keypoint tracking evaluation — PCK@0.1 metric.
# Protocol: frozen model, per-joint learned query → 2D coordinate prediction.
# PCK@0.1: predicted joint within 10% of torso diameter of ground truth.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from rvm_eupe.models.readout_heads import build_keypoint_head

# JHMDB has 15 joints
JHMDB_NUM_JOINTS = 15


@torch.no_grad()
def _extract_state(model, source_frames, device) -> Tensor:
    s = None
    for frame in source_frames:
        e_t = model.encoder(frame.to(device))
        s, _ = model.recurrent(e_t, s)
    return s


def train_and_eval_keypoint(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_dim: int,
    num_joints: int = JHMDB_NUM_JOINTS,
    num_heads: int = 12,
    train_steps: int = 10000,
    lr: float = 1e-3,
    pck_threshold: float = 0.1,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    Train keypoint readout head and evaluate PCK@0.1.

    DataLoader batch format:
        {
          "source_frames": [B, T, 3, H, W],
          "keypoints":     [B, num_joints, 2]  — (x, y) normalised to [0, 1]
          "torso_diam":    [B]                 — torso diameter in pixels / image_size
        }

    Returns:
        {"PCK@0.1": float}
    """
    model.eval()
    model.freeze_all()

    head = build_keypoint_head(state_dim, num_joints=num_joints, num_heads=num_heads).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    # --- Training (L2 loss on normalised coordinates) ---
    head.train()
    step = 0
    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break

            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            kp_gt = batch["keypoints"].to(device)   # [B, J, 2]

            with torch.no_grad():
                s = _extract_state(model, source_frames, device)

            pred = head(s)  # [B, J, 2]
            loss = F.mse_loss(pred, kp_gt)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

    # --- Evaluation (PCK@threshold) ---
    head.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in val_loader:
            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            kp_gt      = batch["keypoints"].to(device)   # [B, J, 2]
            torso_diam = batch["torso_diam"].to(device)  # [B]

            s    = _extract_state(model, source_frames, device)
            pred = head(s)  # [B, J, 2]

            # Euclidean distance, normalised by torso diameter
            dist = torch.norm(pred - kp_gt, dim=-1)  # [B, J]
            threshold = (torso_diam * pck_threshold).unsqueeze(1)   # [B, 1]
            correct += (dist < threshold).sum().item()
            total   += dist.numel()

    return {"PCK@0.1": correct / total if total > 0 else 0.0}
