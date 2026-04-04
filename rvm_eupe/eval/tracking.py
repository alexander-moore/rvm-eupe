# Copyright (c) 2026 — RVM-EUPE authors.
# Point/object tracking evaluation for Waymo Open Dataset and Perception Test.
#
# Waymo: box-coordinate queries → track centre predictions
# Perception Test: Fourier-embedded position queries → Average Jaccard (AJ)
#
# Both use the same AttentiveReadoutHead with SpatialQueryFactory.

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from rvm_eupe.models.readout_heads import AttentiveReadoutHead, SpatialQueryFactory


@torch.no_grad()
def _extract_state(model, source_frames, device) -> Tensor:
    s = None
    for frame in source_frames:
        e_t = model.encoder(frame.to(device))
        s, _ = model.recurrent(e_t, s)
    return s


def _build_tracking_head(state_dim: int, num_points: int, num_heads: int) -> AttentiveReadoutHead:
    """Tracking head: one query per point, output = 2D coordinate."""
    import torch.nn as nn
    factory = _PointQueryFactory(num_points, state_dim)
    return AttentiveReadoutHead(state_dim, factory, num_heads, num_blocks=4, output_dim=2)


class _PointQueryFactory(torch.nn.Module):
    """Learned queries for point tracking (one per tracked point)."""
    def __init__(self, num_points: int, dim: int) -> None:
        super().__init__()
        self.queries = torch.nn.Parameter(torch.zeros(1, num_points, dim))
        torch.nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        return self.queries.expand(batch_size, -1, -1)


def train_and_eval_perception_test(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    state_dim: int,
    num_points: int = 256,
    num_heads: int = 12,
    train_steps: int = 10000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    Perception Test point tracking — Average Jaccard (AJ).

    DataLoader batch format:
        {
          "source_frames":  [B, T, 3, H, W],
          "query_points":   [B, num_points, 2]   — (x,y) normalised [0,1]
          "target_points":  [B, num_points, 2]   — ground truth in target frame
          "occluded":       [B, num_points]       — bool, True if point is occluded
        }
    """
    model.eval()
    model.freeze_all()

    head = _build_tracking_head(state_dim, num_points, num_heads).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    # --- Training ---
    head.train()
    step = 0
    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break

            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            target_pts = batch["target_points"].to(device)   # [B, P, 2]
            occluded   = batch["occluded"].to(device)        # [B, P]

            with torch.no_grad():
                s = _extract_state(model, source_frames, device)

            pred = head(s)  # [B, P, 2]
            # Only supervise visible points
            visible = ~occluded
            if visible.any():
                loss = F.mse_loss(pred[visible], target_pts[visible])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            step += 1

    # --- Evaluation: Average Jaccard ---
    head.eval()
    # AJ = fraction of (point, threshold) pairs where distance < threshold
    # Thresholds: 1, 2, 4, 8, 16 pixels (following Kubric/Perception Test protocol)
    thresholds = [1, 2, 4, 8, 16]
    correct = {t: 0 for t in thresholds}
    total_visible = 0

    with torch.no_grad():
        for batch in val_loader:
            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            target_pts = batch["target_points"].to(device)
            occluded   = batch["occluded"].to(device)
            img_size   = batch.get("image_size", torch.tensor([256.0])).to(device)

            s    = _extract_state(model, source_frames, device)
            pred = head(s)  # [B, P, 2]

            # Convert normalised coords to pixels
            pred_px   = pred     * img_size.unsqueeze(-1).unsqueeze(-1)
            target_px = target_pts * img_size.unsqueeze(-1).unsqueeze(-1)

            visible = ~occluded
            dist = torch.norm(pred_px - target_px, dim=-1)  # [B, P]

            total_visible += visible.sum().item()
            for t in thresholds:
                correct[t] += (dist[visible] < t).sum().item()

    if total_visible == 0:
        return {"AJ": 0.0}

    aj = sum(correct[t] / total_visible for t in thresholds) / len(thresholds)
    return {"AJ": aj, **{f"PCK_{t}px": correct[t] / total_visible for t in thresholds}}
