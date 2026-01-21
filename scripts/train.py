from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from kidney_classifier.train import train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=Path, default=Path("params.yaml"))
    p.add_argument("--train-dir", type=Path, default=Path("data/processed/train"))
    p.add_argument("--model-out", type=Path, default=Path("models/model.pt"))
    p.add_argument("--metrics-out", type=Path, default=Path("metrics/train.json"))
    p.add_argument("--classmap-out", type=Path, default=Path("models/classmap.json"))
    args = p.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    hp = params["train"]

    train(
        args.train_dir,
        args.model_out,
        args.metrics_out,
        args.classmap_out,
        image_size=int(hp["image_size"]),
        batch_size=int(hp["batch_size"]),
        epochs=int(hp["epochs"]),
        lr=float(hp["lr"]),
        pretrained=bool(hp["pretrained"]),
        seed=int(hp["seed"]),
    )


if __name__ == "__main__":
    main()
