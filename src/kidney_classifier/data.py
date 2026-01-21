from __future__ import annotations

import json
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


def build_transforms(image_size: int) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def load_imagefolder(
    root: Path, image_size: int, batch_size: int, num_workers: int = 0
) -> torch.utils.data.DataLoader:
    ds = ImageFolder(str(root), transform=build_transforms(image_size))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def save_class_mapping(root: Path, class_to_idx: dict[str, int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(v): k for k, v in class_to_idx.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
