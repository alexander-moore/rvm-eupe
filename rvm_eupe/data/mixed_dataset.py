# Copyright (c) 2026 — RVM-EUPE authors.
# Weighted mixture of training video datasets, matching RVM paper mixture.

from typing import Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler

from .video_dataset import VideoClipDataset


# Dataset weights from RVM paper (Table in appendix)
DEFAULT_DATASET_WEIGHTS: Dict[str, float] = {
    "ssv2":        0.20,
    "kinetics700": 0.25,
    "howto100m":   0.30,
    "yt8m":        0.15,
    "ytbb":        0.10,
}


class MixedVideoDataset(Dataset):
    """
    Concatenation of multiple VideoClipDatasets with per-dataset weights.
    Uses WeightedRandomSampler to draw samples according to the weight vector.

    Args:
        datasets:  Dict mapping dataset name → VideoClipDataset.
        weights:   Dict mapping dataset name → sampling weight (need not sum to 1).
                   Defaults to RVM paper mixture if None.
    """

    def __init__(
        self,
        datasets: Dict[str, VideoClipDataset],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        if weights is None:
            weights = DEFAULT_DATASET_WEIGHTS

        # Only include datasets that are present
        active = {k: v for k, v in datasets.items() if k in weights}
        self._dataset = ConcatDataset(list(active.values()))

        # Build per-sample weights for WeightedRandomSampler
        sample_weights: List[float] = []
        for name, ds in active.items():
            w = weights[name]
            n = len(ds)
            per_sample = w / n if n > 0 else 0.0
            sample_weights.extend([per_sample] * n)

        self._sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> dict:
        return self._dataset[idx]

    def make_sampler(self, num_samples: Optional[int] = None) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for use in DataLoader."""
        n = num_samples or len(self._dataset)
        return WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=n,
            replacement=True,
        )
