"""STAR: Semantic Targeting for Analytical Repair.

For a failing token (the model produces `argmax`, the reference says `target`):
  1. semantic steer  -- the direction between the two tokens' semantic bases, robustly scaled;
  2. prior delta     -- for each repaired layer, the minimum-norm change of the FFN output weight whose
                        effect on the current neuron activations equals the steer (analytical solution);
  3. guided update   -- step by the prior's magnitude in the gradient's descent direction (`Seme`).
Steps 2 and 3 repeat for a few epochs, because the activations move as the weights change. All focus tokens of the
answer are solved for jointly.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from loguru import logger
from torch import nn

from .attribution import layer_occlusion_losses
from .engine import RepairEngine
from .hparams import STARHyperParams
from .seme import Seme, min_norm_coefficients


def semantic_steer(engine: RepairEngine, target_label: int, argmax_label: int) -> torch.Tensor:
    """Scaled semantic delta, shape [1, hidden]. Scaling by the inter-quartile range is robust to outliers."""
    steer = engine.semantic_delta(target_label, argmax_label)
    iqr = torch.quantile(steer, 0.75) - torch.quantile(steer, 0.25)
    return (steer / (iqr + 1e-8)).unsqueeze(0)


def select_layers(engine: RepairEngine, hparams: STARHyperParams, input_ids: torch.Tensor, target_label: int
                  ) -> List[int]:
    """The layers to repair, in ascending order: those whose FFN occlusion hurts the expected token most."""
    number = max(1, round(engine.n_layers * hparams.layer_ratio))
    if number >= engine.n_layers:
        return list(range(engine.n_layers))
    losses = layer_occlusion_losses(engine, input_ids, target_label)
    ranked = sorted(range(engine.n_layers), key=lambda layer: losses[layer], reverse=True)
    return sorted(ranked[:number])


def _forward_with_acts(engine: RepairEngine, input_ids: torch.Tensor, layers: List[int], positions: slice):
    """Run the model and collect the neuron activations of `layers` at `positions` ([points, neurons]).

    The activations are the input of the FFN output projection: for a gated FFN
    (`down_proj(act_fn(gate_proj(x)) * up_proj(x))`) this is what multiplies the write vectors.
    """
    acts: Dict[int, torch.Tensor] = {}

    def make_key_hook(layer: int):
        def hook(_module, inputs):
            acts[layer] = inputs[0][0, positions, :].detach().to(torch.float)
        return hook

    handles = [engine.ff_out(layer).register_forward_pre_hook(make_key_hook(layer)) for layer in layers]
    try:
        logits = engine.model(input_ids=input_ids).logits
    finally:
        for handle in handles:
            handle.remove()
    return acts, logits


def preservation_weights(engine: RepairEngine, hparams: STARHyperParams, layer: int, acts: torch.Tensor):
    """[neurons]: how little unrelated text uses each neuron (the inverse of its mean squared activation on the
    reference prompts), or None without null space."""
    if hparams.null_space == "none":
        return None
    moments = engine.model._key_second_moments[1][layer].to(acts.device)
    return 1.0 / (moments + 1e-3 * moments.mean())


def optimize(engine, hparams, layers, input_ids, act_positions, loss_positions, targets, steers) -> None:
    """The repair loop: `loss_positions` predict `targets`; activations at `act_positions` move along `steers`."""
    params = engine.set_trainable(layers)
    optimizer = Seme(params, lr=hparams.lr * hparams.scale)
    steers = steers.to(torch.float)
    engine.model.train()
    for epoch in range(hparams.epoch_num):
        acts, logits = _forward_with_acts(engine, input_ids, layers, act_positions)
        # priors in factored form; each is materialized, used and released inside optimizer.step()
        weights = {layer: preservation_weights(engine, hparams, layer, acts[layer]) for layer in layers}
        optimizer.set_priors({
            engine.weight(layer): (
                acts[layer] if weights[layer] is None else acts[layer] * weights[layer],
                min_norm_coefficients(acts[layer], steers, weights[layer]),
            ) for layer in layers
        })
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(logits[0, loss_positions, :].to(torch.float), targets)
        loss.backward()
        optimizer.step()
        logger.debug(f"epoch {epoch + 1}/{hparams.epoch_num}: loss {loss.item():.4f}")
        del acts, logits, loss
    optimizer.zero_grad(set_to_none=True)
    engine.model.eval()


def repair_jointly(engine: RepairEngine, hparams: STARHyperParams, source: str, target_ids: List[int]) -> None:
    source_ids = engine.encode(source)
    answer_ids = torch.tensor([target_ids], device=engine.device)
    input_ids = torch.cat([source_ids, answer_ids], dim=-1)
    num_source, num_target = source_ids.shape[1], len(target_ids)
    # position i predicts the token at i + 1; the activations are read at the predicting positions
    loss_positions = slice(num_source - 1, num_source - 1 + num_target)
    act_positions = loss_positions

    with torch.no_grad():
        argmax_ids = engine.model(input_ids=input_ids).logits[0, loss_positions, :].argmax(dim=-1).tolist()
    failing = [(index, t, a) for index, (t, a) in enumerate(zip(target_ids, argmax_ids)) if t != a]
    if not failing:
        return
    # a correct token has a zero steer (target and argmax coincide): the solve leaves its activations in place
    steers = torch.cat([semantic_steer(engine, t, a) for t, a in zip(target_ids, argmax_ids)], dim=0)
    first = failing[0][0]
    layers = select_layers(engine, hparams, input_ids[:, : num_source + first], target_ids[first])
    targets = torch.tensor(target_ids, device=engine.device)
    optimize(engine, hparams, layers, input_ids, act_positions, loss_positions, targets, steers)
    for _, t, a in failing:
        engine.report["points"].append({"target": t, "argmax": a, "layers": layers})


def apply_STAR_to_model(
    model,
    tok,
    hparams: STARHyperParams,
    batch_data: list,
):
    """Repair the model on one sample; returns the original weights of the changed parameters."""
    engine = RepairEngine.from_hparams(model, tok, hparams)
    data = batch_data[0]
    target_ids = tok(data["answer"], add_special_tokens=False)["input_ids"][: hparams.max_focus_tokens]
    try:
        if target_ids:
            repair_jointly(engine, hparams, data["question"], target_ids)
    except Exception:
        engine.rollback()
        engine.finish()
        raise
    return engine.finish()
