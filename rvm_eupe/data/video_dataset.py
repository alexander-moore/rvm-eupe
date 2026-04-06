# Copyright (c) 2026 — RVM-EUPE authors.
# Base video clip sampler.
#
# Clip sampling strategy (matching RVM paper Section 3 / Supplementary Table 4):
#   - 4 consecutive source frames (stride 1)
#   - 4 target frames, each independently sampled at Δt ∈ [gap_min, gap_max]
#     frames after the last source frame (default gap_min=4, gap_max=48)
#
# Dataset format: each entry is a path to a video file (mp4/avi/webm) or a
# directory of JPEG frames.  A JSON/CSV index file lists these paths.

import json
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------------
# Frame loaders
# ---------------------------------------------------------------------------

def _load_video_frames_decord(
    video_path: str,
    frame_indices: List[int],
) -> List[Image.Image]:
    """Load specific frames from a video file using decord."""
    import decord  # optional dep; only used when videos are available
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path, num_threads=1)
    actual_len = len(vr)
    # Clamp indices to actual video length (index may have stale/default num_frames)
    clamped = [min(i, actual_len - 1) for i in frame_indices]
    frames = vr.get_batch(clamped)  # [T, H, W, 3] uint8
    return [Image.fromarray(f.numpy()) for f in frames]


def _load_frame_dir(
    frame_dir: str,
    frame_indices: List[int],
    extension: str = ".jpg",
) -> List[Image.Image]:
    """Load pre-extracted frames from a directory (frame_%06d.jpg naming)."""
    files = sorted(Path(frame_dir).glob(f"*{extension}"))
    return [Image.open(str(files[i])).convert("RGB") for i in frame_indices]


# ---------------------------------------------------------------------------
# VideoClipDataset
# ---------------------------------------------------------------------------

class VideoClipDataset(Dataset):
    """
    Base video clip dataset.

    Index file format (JSON list of dicts):
        [
          {"path": "/data/k700/video_001.mp4", "num_frames": 250},
          {"path": "/data/k700/frames/video_002/", "num_frames": 180},
          ...
        ]

    Args:
        index_path:        Path to JSON index file.
        transform:         Callable applied to a list of PIL Images (source + target).
        num_source_frames: Number of consecutive source frames (default 4).
        source_stride:     Temporal stride between source frames (default 1).
        gap_min:           Minimum gap (in frames) between last source and target (default 4).
        gap_max:           Maximum gap (default 48).
        min_video_frames:  Skip videos shorter than this (default: num_source_frames + gap_min).
        frame_format:      "video" (use decord) or "frames" (pre-extracted JPEGs).
    """

    def __init__(
        self,
        index_path: str,
        transform: Optional[Callable] = None,
        num_source_frames: int = 4,
        num_target_frames: int = 4,
        source_stride: int = 1,
        gap_min: int = 4,
        gap_max: int = 48,
        min_video_frames: int = 0,
        frame_format: str = "video",
    ) -> None:
        self.transform = transform
        self.num_source_frames = num_source_frames
        self.num_target_frames = num_target_frames
        self.source_stride = source_stride
        self.gap_min = gap_min
        self.gap_max = gap_max
        self.frame_format = frame_format

        with open(index_path) as f:
            raw = json.load(f)

        required_frames = (num_source_frames - 1) * source_stride + 1 + gap_min
        min_frames = max(min_video_frames, required_frames)
        self.entries = [e for e in raw if e.get("num_frames", 999) >= min_frames]

    def __len__(self) -> int:
        return len(self.entries)

    def _sample_indices(self, num_frames: int) -> Tuple[List[int], List[int]]:
        """
        Sample 4 consecutive source indices and 4 independent target indices.
        Each target is drawn with gap Δt ~ Uniform[gap_min, gap_max] from the
        last source frame (RVM paper Section 3 / Supplementary Table 4).
        Returns (source_indices, target_indices).
        """
        required = (self.num_source_frames - 1) * self.source_stride + 1
        latest_start = num_frames - required - self.gap_min
        if latest_start < 0:
            latest_start = 0
        start = random.randint(0, max(0, latest_start))

        source_indices = [start + i * self.source_stride for i in range(self.num_source_frames)]
        last_source = source_indices[-1]

        max_gap = min(self.gap_max, num_frames - last_source - 1)
        max_gap = max(max_gap, self.gap_min)  # guard for very short clips
        target_indices = [
            last_source + random.randint(self.gap_min, max_gap)
            for _ in range(self.num_target_frames)
        ]

        return source_indices, target_indices

    def _load_frames(self, entry: dict, indices: List[int]) -> List[Image.Image]:
        path = entry["path"]
        if self.frame_format == "video":
            return _load_video_frames_decord(path, indices)
        else:
            return _load_frame_dir(path, indices)

    def __getitem__(self, idx: int) -> dict:
        for attempt in range(5):
            try:
                entry = self.entries[(idx + attempt) % len(self.entries)]
                num_frames = entry.get("num_frames", 300)

                source_indices, target_indices = self._sample_indices(num_frames)
                all_indices = source_indices + target_indices

                frames = self._load_frames(entry, all_indices)

                if self.transform is not None:
                    frames = self.transform(frames)

                source_frames = frames[: self.num_source_frames]           # list of [3, H, W]
                target_frames = frames[self.num_source_frames :]           # list of [3, H, W]

                return {
                    "source_frames": torch.stack(source_frames),  # [T_src, 3, H, W]
                    "target_frames": torch.stack(target_frames),  # [T_tgt, 3, H, W]
                }
            except Exception:
                continue
        raise RuntimeError(f"Failed to load clip after 5 attempts, starting at idx={idx}")


def collate_fn(batch: List[dict]) -> dict:
    source_frames = torch.stack([b["source_frames"] for b in batch])  # [B, T_src, 3, H, W]
    target_frames = torch.stack([b["target_frames"] for b in batch])  # [B, T_tgt, 3, H, W]
    return {"source_frames": source_frames, "target_frames": target_frames}
