# Copyright (c) 2026 — RVM-EUPE authors.
# ScanNet depth estimation evaluation (AbsRel metric).
# Protocol: frozen model, spatial AttentiveReadoutHead with 8x8 grid queries,
# log-depth L2 loss, AbsRel = mean(|pred - gt| / gt) on val set.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from rvm_eupe.models.readout_heads import build_depth_head


@torch.no_grad()
def _extract_state(model, source_frames, device) -> Tensor:
    s = None
    for frame in source_frames:
        e_t = model.encoder(frame.to(device))
        s, _ = model.recurrent(e_t, s)
    return s  # [B, N, D]


def train_and_eval_depth(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_dim: int,
    num_heads: int = 12,
    train_steps: int = 10000,
    lr: float = 1e-3,
    grid_h: int = 8,
    grid_w: int = 8,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    Train depth head on frozen hidden state, evaluate AbsRel on val set.

    DataLoader batch format:
        {
          "source_frames": [B, T, 3, H, W],
          "depth": [B, 1, H, W]  — metric depth in metres
        }

    Returns:
        metrics: {"AbsRel": float, "SqRel": float, "RMSE": float}
    """
    model.eval()
    model.freeze_all()

    head = build_depth_head(state_dim, grid_h=grid_h, grid_w=grid_w, num_heads=num_heads).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    # --- Training (log-depth L2 loss) ---
    head.train()
    step = 0
    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break

            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            depth_gt = batch["depth"].to(device)       # [B, 1, H, W]

            with torch.no_grad():
                s = _extract_state(model, source_frames, device)

            # Head outputs [B, grid_h*grid_w, 1] → reshape → upsample
            pred_log = head(s)  # [B, Q, 1]
            B = pred_log.shape[0]
            pred_log = pred_log.reshape(B, grid_h, grid_w, 1).permute(0, 3, 1, 2)  # [B, 1, gh, gw]
            pred_log_up = F.interpolate(pred_log, size=depth_gt.shape[-2:], mode="bilinear",
                                        align_corners=False)

            # Mask invalid depth pixels
            valid = (depth_gt > 0.1) & (depth_gt < 10.0)
            log_gt = torch.log(depth_gt.clamp(min=1e-3))
            loss = F.mse_loss(pred_log_up[valid], log_gt[valid])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

    # --- Evaluation ---
    head.eval()
    abs_rel_sum = sq_rel_sum = rmse_sum = n = 0.0

    with torch.no_grad():
        for batch in val_loader:
            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            depth_gt = batch["depth"].to(device)

            s = _extract_state(model, source_frames, device)
            pred_log = head(s).reshape(depth_gt.shape[0], grid_h, grid_w, 1).permute(0, 3, 1, 2)
            pred_depth = torch.exp(
                F.interpolate(pred_log, size=depth_gt.shape[-2:], mode="bilinear", align_corners=False)
            )

            valid = (depth_gt > 0.1) & (depth_gt < 10.0)
            gt_v = depth_gt[valid]
            pd_v = pred_depth[valid]

            abs_rel_sum += ((pd_v - gt_v).abs() / gt_v).sum().item()
            sq_rel_sum  += (((pd_v - gt_v) ** 2) / gt_v).sum().item()
            rmse_sum    += ((pd_v - gt_v) ** 2).sum().item()
            n           += valid.sum().item()

    if n == 0:
        return {"AbsRel": 0.0, "SqRel": 0.0, "RMSE": 0.0}

    return {
        "AbsRel": abs_rel_sum / n,
        "SqRel":  sq_rel_sum  / n,
        "RMSE":   (rmse_sum / n) ** 0.5,
    }
