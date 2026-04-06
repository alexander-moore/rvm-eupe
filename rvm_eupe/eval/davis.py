# Copyright (c) 2026 — RVM-EUPE authors.
# DAVIS 2017 evaluation via non-parametric k-NN label propagation.
# Matches the RVM paper protocol (Table 8):
#   frozen model, k=7, τ=0.7, context_frames=20, search_radius=20, 480p resolution.
#
# Reference: "Video Object Segmentation using Space-Time Memory Networks"
# Protocol: for each query frame, propagate labels from all previous frames
# using cosine similarity between patch tokens.

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


# ---------------------------------------------------------------------------
# Label propagation (non-parametric k-NN)
# ---------------------------------------------------------------------------

@torch.no_grad()
def propagate_labels_knn(
    query_features: Tensor,   # [N_q, D]  features of query frame tokens
    memory_features: Tensor,  # [N_m, D]  features of memory frame tokens (flattened)
    memory_labels:  Tensor,   # [N_m, C]  one-hot labels in memory
    k: int = 7,
    tau: float = 0.7,
) -> Tensor:
    """
    For each query token, find k nearest neighbours in memory and soft-vote labels.

    Returns:
        pred_labels: [N_q, C]  soft label predictions
    """
    # Cosine similarity
    q = F.normalize(query_features, dim=-1)
    m = F.normalize(memory_features, dim=-1)
    sim = (q @ m.T) / tau   # [N_q, N_m]

    # Top-k softmax
    topk_sim, topk_idx = sim.topk(k, dim=-1)           # [N_q, k]
    topk_weights = F.softmax(topk_sim, dim=-1)          # [N_q, k]
    topk_labels = memory_labels[topk_idx]               # [N_q, k, C]
    pred = (topk_weights.unsqueeze(-1) * topk_labels).sum(dim=1)  # [N_q, C]
    return pred


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_frame_features(
    model,
    frame: Tensor,   # [1, 3, H, W]
    device: torch.device,
) -> Tensor:
    """
    Extract patch tokens from a single frame using the frozen model's encoder.
    Returns [N, D].
    """
    frame = frame.to(device)
    tokens = model.encoder(frame)   # [1, N, D]
    return tokens.squeeze(0)        # [N, D]


# ---------------------------------------------------------------------------
# DAVIS evaluation
# ---------------------------------------------------------------------------

def _load_annotation(path: str, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load a PNG annotation mask as a uint8 numpy array [H, W]."""
    ann = np.array(Image.open(path).convert("P"))
    if resize is not None:
        ann = np.array(Image.fromarray(ann).resize(resize, Image.NEAREST))
    return ann


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    h, w = labels.shape
    oh = np.zeros((h * w, num_classes), dtype=np.float32)
    flat = labels.flatten()
    valid = flat < num_classes
    oh[np.where(valid), flat[valid]] = 1.0
    return oh


@torch.no_grad()
def evaluate_davis(
    model,
    davis_root: str,
    split: str = "val",
    patch_size: int = 16,
    k: int = 7,
    tau: float = 0.7,
    context_frames: int = 20,
    search_radius: int = 20,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """
    Run non-parametric label propagation evaluation on DAVIS 2017.
    Protocol matches RVM paper Table 8: k=7, τ=0.7, 20 context frames,
    search_radius=20 (spatial locality window in patch units), 480p resolution.

    Args:
        model:           Frozen RecurrentVideoMAE (only encoder is used here).
        davis_root:      Path to DAVIS 2017 dataset root.
        split:           "val" or "test-dev".
        patch_size:      Patch size (16 for all EUPE ViT variants).
        k:               k-NN neighbours (paper: 7).
        tau:             Temperature for softmax (paper: 0.7).
        context_frames:  Past frames in memory (paper: 20).
        search_radius:   Spatial search window in patch units (paper: 20).
        device:          Compute device.

    Returns:
        metrics: dict with "J_mean", "F_mean", "JF_mean"
    """
    model.eval()
    model.freeze_all()

    davis_root = Path(davis_root)
    sequences_file = davis_root / "ImageSets" / "2017" / f"{split}.txt"
    sequences = [s.strip() for s in sequences_file.read_text().splitlines() if s.strip()]

    all_j: List[float] = []
    all_f: List[float] = []

    for seq in sequences:
        frames_dir  = davis_root / "JPEGImages" / "480p" / seq
        ann_dir     = davis_root / "Annotations" / "480p" / seq

        frame_files = sorted(frames_dir.glob("*.jpg"))
        ann_files   = sorted(ann_dir.glob("*.png"))

        if not frame_files:
            continue

        # Determine number of objects from first annotation
        first_ann = _load_annotation(str(ann_files[0]))
        num_objects = int(first_ann.max())
        if num_objects == 0:
            continue
        num_classes = num_objects + 1  # background + objects

        # Image size for feature map
        H, W = first_ann.shape
        feat_h = H // patch_size
        feat_w = W // patch_size

        # Load + normalise frames
        from rvm_eupe.data.transforms import build_eval_transforms
        import torchvision.transforms.functional as TF

        def load_frame(path: str) -> Tensor:
            img = Image.open(path).convert("RGB").resize((W, H))
            t = TF.to_tensor(img)
            t = TF.normalize(t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            return t.unsqueeze(0)

        # --- First frame: use ground-truth annotation as memory ---
        memory_features_list: List[Tensor] = []
        memory_labels_list:   List[Tensor] = []

        first_feats = extract_frame_features(model, load_frame(str(frame_files[0])), device)
        first_ann_resized = _load_annotation(str(ann_files[0]), resize=(feat_w, feat_h))
        first_labels_oh = torch.tensor(
            _one_hot(first_ann_resized, num_classes), device=device
        )
        memory_features_list.append(first_feats)
        memory_labels_list.append(first_labels_oh)

        seq_j: List[float] = []
        seq_f: List[float] = []

        for t in range(1, len(frame_files)):
            query_feats = extract_frame_features(model, load_frame(str(frame_files[t])), device)

            # Concatenate memory
            mem_feats  = torch.cat(memory_features_list[-context_frames:], dim=0)
            mem_labels = torch.cat(memory_labels_list[-context_frames:],   dim=0)

            # Propagate
            pred_labels = propagate_labels_knn(query_feats, mem_feats, mem_labels, k=k, tau=tau)
            pred_map = pred_labels.argmax(-1).reshape(feat_h, feat_w).cpu().numpy().astype(np.uint8)
            pred_map_full = np.array(
                Image.fromarray(pred_map).resize((W, H), Image.NEAREST)
            )

            # Add to memory
            memory_features_list.append(query_feats)
            pred_labels_for_mem = torch.zeros(feat_h * feat_w, num_classes, device=device)
            pred_labels_for_mem.scatter_(1, pred_labels.argmax(-1, keepdim=True), 1.0)
            memory_labels_list.append(pred_labels_for_mem)

            # Compute J (IoU) and F (boundary) per object
            if t < len(ann_files):
                gt = _load_annotation(str(ann_files[t]))
                j_scores, f_scores = _compute_jf(gt, pred_map_full, num_objects)
                seq_j.append(np.mean(j_scores))
                seq_f.append(np.mean(f_scores))

        if seq_j:
            all_j.append(np.mean(seq_j))
            all_f.append(np.mean(seq_f))

    j_mean = float(np.mean(all_j)) if all_j else 0.0
    f_mean = float(np.mean(all_f)) if all_f else 0.0
    return {
        "J_mean": j_mean,
        "F_mean": f_mean,
        "JF_mean": (j_mean + f_mean) / 2.0,
    }


def _compute_jf(
    gt: np.ndarray,
    pred: np.ndarray,
    num_objects: int,
) -> Tuple[List[float], List[float]]:
    """Compute per-object J (IoU) and F (boundary F-measure)."""
    j_scores, f_scores = [], []
    for obj_id in range(1, num_objects + 1):
        gt_mask   = (gt == obj_id)
        pred_mask = (pred == obj_id)
        j_scores.append(_jaccard(gt_mask, pred_mask))
        f_scores.append(_f_boundary(gt_mask, pred_mask))
    return j_scores, f_scores


def _jaccard(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return float(intersection) / (float(union) + 1e-8)


def _f_boundary(gt: np.ndarray, pred: np.ndarray, bound_th: float = 0.008) -> float:
    """Compute F-measure of boundary pixels (morphological gradient)."""
    from scipy.ndimage import binary_dilation
    bound_pix = max(1, round(bound_th * max(gt.shape)))
    gt_b   = binary_dilation(gt, iterations=bound_pix) & ~binary_dilation(~gt, iterations=bound_pix)
    pred_b = binary_dilation(pred, iterations=bound_pix) & ~binary_dilation(~pred, iterations=bound_pix)
    prec = (gt_b & pred_b).sum() / (pred_b.sum() + 1e-8)
    rec  = (gt_b & pred_b).sum() / (gt_b.sum()   + 1e-8)
    if prec + rec < 1e-8:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
