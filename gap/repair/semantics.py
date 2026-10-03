"""Semantic quantities at the level of neurons, in engine terms.

The chain is logits -> representation -> parameters:

  logits          the failing prediction is a contrast of two tokens, read by the covector
                  `c = a_target - a_argmax` (decoder atoms) off the normalised final State;
  representation  every component that writes into the residual stream at the predicting position shifts
                  that contrast by `c @ write / scale`: its direct contribution. The residual stream is a sum, so
                  the direct contributions of layers add up to the contrast;
  parameters      the FFN output of a layer is `sum_j key_j * w_j`, so the direct contribution of neuron j is
                  `key_j * (c @ w_j) / scale`. The pairing `c @ w_j` is a covector applied to a vector: it does
                  not depend on the hidden coordinates. The entries of `w_j` taken one by one do.

A neuron is therefore the smallest unit whose contribution to the Semantic State is well defined, and its
write vector `w_j` is the smallest thing a repair can change with a predictable semantic effect.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch


@torch.no_grad()
def neuron_semantic_deltas(engine, input_ids, target_label: int, argmax_label: int,
                           layers: Optional[List[int]] = None, observed=None) -> Dict[int, torch.Tensor]:
    """{layer: tensor[neurons]}: how much each neuron currently moves the logit contrast (target - argmax).

    Negative values push toward the wrong token. One forward pass, no backward pass.
    """
    layers = list(range(engine.n_layers)) if layers is None else layers
    keys, scale, _ = observed if observed is not None else engine.observe(input_ids, layers)
    contrast = engine.contrast(target_label, argmax_label)
    deltas = {}
    for layer in layers:
        readout = contrast @ engine.weight(layer).detach().float()           # c @ w_j for every neuron
        deltas[layer] = keys[layer] * readout / scale
    return deltas


@torch.no_grad()
def key_second_moments(engine, prompts: List[str], layers: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
    """{layer: tensor[neurons]}: mean squared neuron activation over all positions of `prompts`.

    This is the diagonal of the key covariance that null-space editing preserves: a neuron that is rarely
    active on unrelated text can be changed with little collateral effect. Cached on the model per prompt set.
    """
    layers = list(range(engine.n_layers)) if layers is None else layers
    cache = getattr(engine.model, "_key_second_moments", None)
    signature = (len(prompts), hash(tuple(prompts)))
    if cache is not None and cache[0] == signature and all(layer in cache[1] for layer in layers):
        return cache[1]

    sums = {layer: 0.0 for layer in layers}
    count = 0

    def hook(layer):
        def inner(_module, inputs):
            sums[layer] = sums[layer] + inputs[0][0].detach().float().square().sum(dim=0)
        return inner

    handles = [engine.ff_out(layer).register_forward_pre_hook(hook(layer)) for layer in layers]
    try:
        for prompt in prompts:
            input_ids = engine.encode(prompt)
            engine.model(input_ids=input_ids)
            count += input_ids.shape[1]
    finally:
        for handle in handles:
            handle.remove()
    moments = {layer: sums[layer] / max(count, 1) for layer in layers}
    engine.model._key_second_moments = (signature, moments)
    return moments


def top_k_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of the k largest entries of a vector (all entries if k covers the vector)."""
    if k <= 0 or k >= scores.numel():
        return torch.ones_like(scores, dtype=torch.bool)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[torch.topk(scores, k).indices] = True
    return mask
