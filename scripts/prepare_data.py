from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from kidney_classifier.prepare_data import (
    prepare_local_imagefolder_dataset,
    prepare_synthetic_dataset,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=Path, default=Path("params.yaml"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = p.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    data_cfg = params.get("data", {})
    prep_cfg = params.get("prepare_data", {})

    source = str(data_cfg.get("source", "synthetic")).lower()
    seed = int(prep_cfg.get("seed", 42))
    image_size = int(prep_cfg.get("image_size", 224))

    if source == "local":
        raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
        train_ratio = float(data_cfg.get("train_ratio", 0.8))
        test_ratio = float(data_cfg.get("test_ratio", 0.2))
        prepare_local_imagefolder_dataset(
            raw_dir=raw_dir,
            out_dir=args.out_dir,
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    else:
        prepare_synthetic_dataset(
            args.out_dir,
            image_size=image_size,
            train_per_class=int(prep_cfg.get("train_per_class", 32)),
            test_per_class=int(prep_cfg.get("test_per_class", 16)),
            seed=seed,
        )


if __name__ == "__main__":
    main()
