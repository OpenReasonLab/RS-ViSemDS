from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


def build_model(name: str, num_classes: int, device, train_mode: str = "head"):
    if name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        head_attr = "fc"
        in_features = model.fc.in_features
    elif name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        head_attr = "fc"
        in_features = model.fc.in_features
    elif name in {"vit_tiny", "vit_small"}:
        import timm

        timm_name = "vit_tiny_patch16_224" if name == "vit_tiny" else "vit_small_patch16_224"
        model = timm.create_model(timm_name, pretrained=True)
        head_attr = "head"
        in_features = model.head.in_features
    else:
        raise ValueError(f"Unsupported model: {name}")

    if train_mode == "head":
        for p in model.parameters():
            p.requires_grad = False
    elif train_mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError("--train-mode must be head or full")

    new_head = nn.Linear(in_features, num_classes)
    setattr(model, head_attr, new_head)
    return model.to(device)


def trainable_head_parameters(model, name: str):
    if name in {"resnet18", "resnet50"}:
        return model.fc.parameters()
    if name in {"vit_tiny", "vit_small"}:
        return model.head.parameters()
    raise ValueError(f"Unsupported model: {name}")


def classification_head(model, name: str):
    if name in {"resnet18", "resnet50"}:
        return model.fc
    if name in {"vit_tiny", "vit_small"}:
        return model.head
    raise ValueError(f"Unsupported model: {name}")


def set_head_train_mode(model, name: str) -> None:
    """Keep the frozen backbone deterministic while training only the head."""
    model.eval()
    if name in {"resnet18", "resnet50"}:
        model.fc.train()
        return
    if name in {"vit_tiny", "vit_small"}:
        model.head.train()
        return
    raise ValueError(f"Unsupported model: {name}")
