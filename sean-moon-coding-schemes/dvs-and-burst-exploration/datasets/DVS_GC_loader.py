"""
DVSGesture dataloader helper using tonic.

This version avoids tonic.transforms.ToTensor() (some tonic versions don't have it)
by converting frames to torch tensors via a small custom transform.

Output per sample (after transforms):
  frames: torch.Tensor [T, 2, H, W]  (ON/OFF polarity channels)
  label:  int
"""

from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as T


def _to_torch_frames(x):
    """
    Convert frames to torch.Tensor and ensure channel-first layout.

    tonic ToFrame can yield either:
      - [T, H, W, C]
      - [T, C, H, W]
    Normalize to [T, C, H, W].
    """
    x = torch.as_tensor(x)
    if x.ndim == 4 and x.shape[-1] in (1, 2):
        x = x.permute(0, 3, 1, 2).contiguous()
    return x


def get_dvs_gesture_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 16,
    n_time_bins: int = 20,
    downsample: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for DVSGesture.
    """
    base_sensor_size = tonic.datasets.DVSGesture.sensor_size  # (H, W, 2)
    H, W, C = base_sensor_size

    if downsample > 1:
        H //= downsample
        W //= downsample

    sensor_size = (H, W, C)

    tfms = []
    if downsample > 1:
        tfms.append(T.Downsample(spatial_factor=downsample))

    tfms += [
        T.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
        _to_torch_frames,
    ]

    transform = T.Compose(tfms)

    # bypass tonic download if extracted folders already exist 
    root = Path(data_dir) / "DVSGesture"
    train_extracted = (root / "ibmGestureTrain").exists()
    test_extracted = (root / "ibmGestureTest").exists()

    if train_extracted and test_extracted:
        original_download = tonic.datasets.DVSGesture.download
        tonic.datasets.DVSGesture.download = lambda self: None
    else:
        original_download = None

    try:
        train_set = tonic.datasets.DVSGesture(
            save_to=data_dir,
            train=True,
            transform=transform,
        )
        test_set = tonic.datasets.DVSGesture(
            save_to=data_dir,
            train=False,
            transform=transform,
        )
    finally:
        if original_download is not None:
            tonic.datasets.DVSGesture.download = original_download

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, test_loader