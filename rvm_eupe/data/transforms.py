# Copyright (c) 2026 — RVM-EUPE authors.
# Video augmentation pipeline.
# All transforms operate on a list of PIL Images (one per frame) and return
# a list of tensors [3, H, W] with the same spatial crop applied to all frames.

import random
from typing import List, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class VideoRandomResizedCrop:
    """Same random crop applied to every frame in the clip."""

    def __init__(
        self,
        size: int = 256,
        scale: Tuple[float, float] = (0.5, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        i, j, h, w = T.RandomResizedCrop.get_params(frames[0], self.scale, self.ratio)
        return [TF.resized_crop(f, i, j, h, w, (self.size, self.size)) for f in frames]


class VideoRandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        if random.random() < self.p:
            return [TF.hflip(f) for f in frames]
        return frames


class VideoColorJitter:
    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        self.jitter = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, frames: List[Image.Image]) -> List[Image.Image]:
        # Same jitter params for all frames to preserve temporal coherence
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.jitter.get_params(
                self.jitter.brightness, self.jitter.contrast,
                self.jitter.saturation, self.jitter.hue,
            )
        out = []
        for f in frames:
            for fn_id in fn_idx:
                if fn_id == 0:
                    f = TF.adjust_brightness(f, brightness_factor)
                elif fn_id == 1:
                    f = TF.adjust_contrast(f, contrast_factor)
                elif fn_id == 2:
                    f = TF.adjust_saturation(f, saturation_factor)
                elif fn_id == 3:
                    f = TF.adjust_hue(f, hue_factor)
            out.append(f)
        return out


class VideoToTensor:
    """Converts list of PIL Images to list of float tensors [3, H, W]."""

    def __call__(self, frames: List[Image.Image]) -> List[torch.Tensor]:
        return [TF.to_tensor(f) for f in frames]


class VideoNormalize:
    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        return [TF.normalize(f, self.mean, self.std) for f in frames]


class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, frames):
        for t in self.transforms:
            frames = t(frames)
        return frames


def build_train_transforms(crop_size: int = 256) -> Compose:
    return Compose([
        VideoRandomResizedCrop(size=crop_size),
        VideoRandomHorizontalFlip(p=0.5),
        VideoColorJitter(0.4, 0.4, 0.4, 0.1),
        VideoToTensor(),
        VideoNormalize(),
    ])


def build_eval_transforms(resize: int = 224) -> Compose:
    """Centre-crop to fixed size, no augmentation."""
    class VideoCenterCrop:
        def __init__(self, size): self.size = size
        def __call__(self, frames):
            return [TF.center_crop(f, self.size) for f in frames]

    class VideoResize:
        def __init__(self, size): self.size = size
        def __call__(self, frames):
            return [TF.resize(f, self.size) for f in frames]

    return Compose([
        VideoResize(resize + 32),
        VideoCenterCrop(resize),
        VideoToTensor(),
        VideoNormalize(),
    ])
