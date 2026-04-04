# Copyright (c) 2026 — RVM-EUPE authors.
# Attentive-probe evaluation for action classification (Kinetics-400/700, SSv2).
# Protocol: frozen encoder + recurrent; train AttentiveReadoutHead for N steps.

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from rvm_eupe.models.readout_heads import build_action_cls_head


@torch.no_grad()
def _extract_state(model, source_frames: list[Tensor], device) -> Tensor:
    """Run frozen model to get hidden state s_T from source frames."""
    s = None
    for frame in source_frames:
        e_t = model.encoder(frame.to(device))
        s, _ = model.recurrent(e_t, s)
    return s  # [B, N, D]


def train_and_eval_action_cls(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    state_dim: int,
    num_heads: int = 12,
    train_steps: int = 10000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda"),
) -> dict:
    """
    Train an attentive readout head on frozen s_T and evaluate on val set.

    Args:
        model:        Frozen RecurrentVideoMAE.
        train_loader: DataLoader yielding {"source_frames": [B,T,3,H,W], "label": [B]}.
        val_loader:   Same.
        num_classes:  Number of action classes.
        state_dim:    embed_dim of the frozen model.

    Returns:
        metrics: {"top1_acc": float, "top5_acc": float}
    """
    model.eval()
    model.freeze_all()

    head = build_action_cls_head(state_dim, num_classes, num_heads).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    # --- Training ---
    head.train()
    step = 0
    while step < train_steps:
        for batch in train_loader:
            if step >= train_steps:
                break
            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            labels = batch["label"].to(device)

            with torch.no_grad():
                s = _extract_state(model, source_frames, device)

            logits = head(s).squeeze(1)  # [B, num_classes] (1 query)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

    # --- Evaluation ---
    head.eval()
    top1_correct = top5_correct = total = 0

    with torch.no_grad():
        for batch in val_loader:
            source_frames = [batch["source_frames"][:, t] for t in range(batch["source_frames"].shape[1])]
            labels = batch["label"].to(device)
            s = _extract_state(model, source_frames, device)
            logits = head(s).squeeze(1)

            _, top5 = logits.topk(5, dim=-1)
            top1_correct += (top5[:, 0] == labels).sum().item()
            top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.shape[0]

    return {
        "top1_acc": top1_correct / total if total > 0 else 0.0,
        "top5_acc": top5_correct / total if total > 0 else 0.0,
    }
