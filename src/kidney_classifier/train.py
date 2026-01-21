from __future__ import annotations

# ruff: noqa: I001

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow

from kidney_classifier.data import load_imagefolder, save_class_mapping
from kidney_classifier.model import build_model, save_model


def train(
    train_dir: Path,
    model_out: Path,
    metrics_out: Path,
    classmap_out: Path,
    *,
    image_size: int,
    batch_size: int,
    epochs: int,
    lr: float,
    pretrained: bool,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = load_imagefolder(train_dir, image_size=image_size, batch_size=batch_size)
    num_classes = len(loader.dataset.classes)

    # persist mapping for inference
    save_class_mapping(train_dir, loader.dataset.class_to_idx, classmap_out)

    model = build_model(num_classes=num_classes, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri(
        # default: local file store
        mlflow.get_tracking_uri() if mlflow.get_tracking_uri() else "file:./mlruns"
    )
    mlflow.set_experiment("kidney-classifier")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "lr": lr,
                "pretrained": pretrained,
                "seed": seed,
                "num_classes": num_classes,
                "device": str(device),
            }
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for x, y in tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * x.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.size(0))

            avg_loss = total_loss / max(total, 1)
            acc = correct / max(total, 1)
            mlflow.log_metrics({"train_loss": avg_loss, "train_acc": acc}, step=epoch)

        model_out.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(model_out))

        # Save a small metrics JSON for DVC
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(
            json.dumps({"train_loss": avg_loss, "train_acc": acc}, indent=2),
            encoding="utf-8",
        )

        # log artifacts
        mlflow.log_artifact(str(metrics_out))
        mlflow.log_artifact(str(classmap_out))
        mlflow.pytorch.log_model(model, artifact_path="model")

    return {"train_loss": avg_loss, "train_acc": acc, "num_classes": num_classes}
