# Copyright (c) 2026 — RVM-EUPE authors.
# Learning rate schedulers and layerwise LR decay utilities.

import math
from typing import Dict, List

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Cosine warmup scheduler
# ---------------------------------------------------------------------------

def cosine_warmup_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Linear warmup then cosine decay.

    At step t:
      if t < warmup_steps: lr = base_lr * (t / warmup_steps)
      else:                 lr = base_lr * (cos_decay * (1 - min_ratio) + min_ratio)
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Layerwise LR decay
# ---------------------------------------------------------------------------

def get_layerwise_param_groups(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_decay: float = 0.75,
    backbone_attr: str = "encoder.backbone",
) -> List[Dict]:
    """
    Build AdamW parameter groups with per-layer LR decay for the EUPE backbone.

    Layers are indexed from the top (last block = highest LR, first block = lowest).
    The decay factor is applied multiplicatively per layer depth from the output.

    Non-backbone parameters (recurrent, decoder) receive base_lr unchanged.

    Args:
        model:           The full RecurrentVideoMAE model.
        base_lr:         Base learning rate for non-backbone parameters.
        weight_decay:    AdamW weight decay.
        backbone_decay:  Multiplicative decay per depth level (0.75 per RVM/EUPE convention).
        backbone_attr:   Dot-path to the backbone within the model.

    Returns:
        List of parameter group dicts for torch.optim.AdamW.
    """
    # Resolve backbone module
    backbone = model
    for attr in backbone_attr.split("."):
        backbone = getattr(backbone, attr)

    num_layers = getattr(backbone, "depth", 12)

    # Build per-layer LR multipliers: layer 0 (input) gets lowest LR
    # layer_lrs[i] = base_lr * decay^(num_layers - 1 - i)
    layer_lrs = {i: base_lr * (backbone_decay ** (num_layers - 1 - i))
                 for i in range(num_layers)}

    # Assign each backbone parameter to its layer index
    backbone_params: Dict[int, List[torch.nn.Parameter]] = {i: [] for i in range(-1, num_layers)}
    backbone_param_ids = set()

    for name, param in backbone.named_parameters():
        if not param.requires_grad:
            continue
        backbone_param_ids.add(id(param))

        # Determine layer index from parameter name
        layer_idx = -1  # default: pre-encoder (embed, cls token, etc.)
        for i in range(num_layers):
            if f"blocks.{i}." in name or f"blocks[{i}]." in name:
                layer_idx = i
                break
        backbone_params[layer_idx].append(param)

    # Build groups for backbone
    param_groups = []
    for layer_idx, params in backbone_params.items():
        if not params:
            continue
        lr = layer_lrs.get(layer_idx, base_lr * (backbone_decay ** (num_layers - 1)))
        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": f"backbone_layer_{layer_idx}",
        })

    # All other parameters at base_lr
    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_param_ids
    ]
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "non_backbone",
        })

    return param_groups
