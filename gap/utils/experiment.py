"""Cross-cutting utilities of the package (model resolution, freezing)."""
from __future__ import annotations

import torch

from gap.utils.models import MODEL_REGISTRY


def resolve_model(model_key: str) -> str:
    return MODEL_REGISTRY.get(model_key, model_key)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
