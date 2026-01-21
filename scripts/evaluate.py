from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from kidney_classifier.eval import evaluate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=Path, default=Path("params.yaml"))
    p.add_argument("--test-dir", type=Path, default=Path("data/processed/test"))
    p.add_argument("--model-path", type=Path, default=Path("models/model.pt"))
    p.add_argument("--metrics-out", type=Path, default=Path("metrics/eval.json"))
    args = p.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    hp = params["train"]
    num_classes = int(params["data"]["num_classes"])

    evaluate(
        args.model_path,
        args.test_dir,
        args.metrics_out,
        image_size=int(hp["image_size"]),
        batch_size=int(hp["batch_size"]),
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()
