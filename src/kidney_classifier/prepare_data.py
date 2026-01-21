from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _write_image(path: Path, size: int, kind: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    # Create two visually distinct synthetic classes:
    # - "normal": softer noise
    # - "tumor": sharper, higher-contrast noise with a bright blob
    if kind == "normal":
        arr = rng.normal(loc=120, scale=25, size=(size, size, 3))
    else:
        arr = rng.normal(loc=120, scale=45, size=(size, size, 3))
        cx, cy = rng.integers(size // 4, 3 * size // 4, size=2)
        rr = rng.integers(size // 10, size // 5)
        y, x = np.ogrid[:size, :size]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= rr**2
        arr[mask] = 255
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="PNG")


def prepare_synthetic_dataset(
    out_dir: Path,
    *,
    image_size: int,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> None:
    """
    Creates a tiny ImageFolder-style dataset:
      data/processed/train/{normal,tumor}/*.png
      data/processed/test/{normal,tumor}/*.png
    """
    classes = ["normal", "tumor"]
    for split, n in [("train", train_per_class), ("test", test_per_class)]:
        for cls in classes:
            d = out_dir / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                _write_image(
                    d / f"{cls}_{i}.png",
                    size=image_size,
                    kind=cls,
                    seed=seed + i + (0 if split == "train" else 10_000),
                )
