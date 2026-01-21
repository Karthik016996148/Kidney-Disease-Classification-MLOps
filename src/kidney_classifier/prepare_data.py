from __future__ import annotations

import random
from pathlib import Path
from shutil import copy2

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


def prepare_local_imagefolder_dataset(
    *,
    raw_dir: Path,
    out_dir: Path,
    train_ratio: float,
    test_ratio: float,
    seed: int,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
) -> None:
    """
    Prepare `data/processed` from a local ImageFolder-style dataset at `raw_dir`.

    Expected layout:
      data/raw/<class_name>/*.{jpg,png,...}

    Output layout:
      data/processed/train/<class_name>/*
      data/processed/test/<class_name>/*
    """
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir not found: {raw_dir}. Put images under data/raw/<class_name>/..."
        )

    if not (0.0 < train_ratio < 1.0) or not (0.0 < test_ratio < 1.0):
        raise ValueError("train_ratio and test_ratio must be within (0, 1)")
    if abs((train_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + test_ratio must equal 1.0")

    class_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class subdirectories found in: {raw_dir}")

    rng = random.Random(seed)
    for class_dir in class_dirs:
        cls = class_dir.name
        files = [p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
        if not files:
            raise ValueError(f"No images found for class '{cls}' under: {class_dir}")

        rng.shuffle(files)
        n_total = len(files)
        n_train = max(1, int(n_total * train_ratio))
        train_files = files[:n_train]
        test_files = files[n_train:]
        if not test_files:
            # ensure at least one test file if possible
            if n_total > 1:
                test_files = [train_files.pop()]

        for split, split_files in [("train", train_files), ("test", test_files)]:
            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                copy2(src, dest_dir / src.name)
