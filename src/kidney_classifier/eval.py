from __future__ import annotations

# ruff: noqa: I001

import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow

from kidney_classifier.data import load_imagefolder
from kidney_classifier.model import load_model


@torch.inference_mode()
def evaluate(
    model_path: Path,
    test_dir: Path,
    metrics_out: Path,
    *,
    image_size: int,
    batch_size: int,
    num_classes: int,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_imagefolder(test_dir, image_size=image_size, batch_size=batch_size)

    model = load_model(str(model_path), num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="eval"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += int(y.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)

    payload = {"eval_loss": avg_loss, "eval_acc": acc}
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # log into the active run if present (works when eval called after train in same script),
    # otherwise start a new run for evaluation
    if mlflow.active_run() is None:
        mlflow.set_experiment("kidney-classifier")
        with mlflow.start_run():
            mlflow.log_metrics(payload)
            mlflow.log_artifact(str(metrics_out))
    else:
        mlflow.log_metrics(payload)
        mlflow.log_artifact(str(metrics_out))

    return payload
