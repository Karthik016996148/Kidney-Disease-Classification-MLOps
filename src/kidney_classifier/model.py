from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    # Lightweight, good baseline
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
