"""Attribution for locating where to repair: the layers a prediction relies on (used by STAR)."""
from __future__ import annotations

from typing import List

import torch
from torch import nn


@torch.no_grad()
def layer_occlusion_losses(engine, input_ids: torch.Tensor, target_label: int) -> List[float]:
    """Loss of predicting `target_label` when the FFN of one layer is skipped (replaced by its input).

    The layers whose removal hurts most are the ones the prediction relies on.
    """
    target = torch.tensor([target_label], device=engine.device)

    def skip_layer(_module, inputs, _output):
        return inputs[0]

    losses = []
    for layer in range(engine.n_layers):
        handle = engine.mlp(layer).register_forward_hook(skip_layer)
        try:
            logits = engine.model(input_ids=input_ids).logits[:, -1, :]
            losses.append(nn.functional.cross_entropy(logits.to(torch.float), target).item())
        finally:
            handle.remove()
    return losses
