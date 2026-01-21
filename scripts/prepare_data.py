from __future__ import annotations

import argparse
from pathlib import Path

from kidney_classifier.prepare_data import prepare_synthetic_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--train-per-class", type=int, default=32)
    p.add_argument("--test-per-class", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    prepare_synthetic_dataset(
        args.out_dir,
        image_size=args.image_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
