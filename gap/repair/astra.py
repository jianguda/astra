"""ASTRA: analytic repair of carrier neurons selected by contrastive semantics.

For a failing token, with `c` the covector of the logit contrast (target minus the best other token):

  direction   d = G^{-1} c, the lift of the contrast covector under the Gram of the decoder atoms (the direction
              that raises the contrast and moves every other logit least),
              normalised so that `c @ d = 1`: one unit of `d` in the residual stream is one logit of contrast
              before the final normalisation;
  neurons     a set S per repaired layer, chosen by contrastive semantics (`neuron_selection`);
  solution    the write vectors of the neurons in S move along `d`:  w_j <- w_j + lambda * u_j * d, where
              `u_j = D_j key_j` and D weights neurons by how little the reference behaviour uses them
              (`null_space`). Among all updates of S that close the gap, this is the one of minimum D-weighted
              norm, and it has rank one per layer: `d (x) u`.
  step        lambda = gap * scale / sum_layers(key @ u): the contrast gap divided by what one unit of lambda
              buys through the direct path. Later layers react to the change, so the gap is re-measured and
              the solve repeated (`max_rounds`); the total update has rank at most `max_rounds` per layer
              and failing token.

No backward pass is needed unless the neurons are chosen by gradient attribution.
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Dict, List

import torch
from loguru import logger

from .engine import RepairEngine
from .hparams import ASTRAHyperParams
from .semantics import neuron_semantic_deltas, top_k_mask


def neuron_budget(hparams: ASTRAHyperParams, width: int, tokens: int, layers: int) -> int:
    """Neurons per repaired layer: `neuron_num`, or in proportion to the answer length (`tokens_ratio`)."""
    if hparams.tokens_ratio <= 0:
        return hparams.neuron_num
    wanted = math.ceil(hparams.tokens_ratio * tokens / max(layers, 1))
    return int(min(max(wanted, hparams.neuron_min), max(hparams.neuron_min, hparams.neuron_max_share * width)))


def candidate_layers(engine: RepairEngine, hparams: ASTRAHyperParams) -> List[int]:
    return list(range(engine.n_layers))


def select_layers(hparams, layers, deltas) -> List[int]:
    number = max(1, round(len(layers) * hparams.layer_ratio))
    if number >= len(layers):
        return layers
    # the FFNs that currently write most along the contrast, in either direction
    score = {layer: float(deltas[layer].abs().sum()) for layer in layers}
    return sorted(sorted(layers, key=lambda layer: score[layer], reverse=True)[:number])


def preservation_weights(engine, hparams, layer: int, key: torch.Tensor) -> torch.Tensor:
    """u = D key: the key as the preservation criterion lets it be used."""
    if hparams.null_space == "none":
        return key.clone()
    if hparams.null_space == "diagonal":
        cache = getattr(engine.model, "_key_second_moments", None)
        if cache is None:
            raise RuntimeError("null_space='diagonal' needs reference prompts (Method.prepare(reference_prompts=...))")
        moments = cache[1][layer].to(key.device)
        return key / (moments + 1e-3 * moments.mean())
    raise ValueError(f"unknown null_space: {hparams.null_space}")


def select_neurons(hparams, weighted: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Boolean mask of the neurons of one layer that the update may change (sequential solve)."""
    if hparams.neuron_selection == "all":
        return torch.ones_like(weighted, dtype=torch.bool)
    if hparams.neuron_selection == "semantic":
        return top_k_mask(delta.abs(), hparams.neuron_num)
    raise ValueError(f"unknown neuron_selection: {hparams.neuron_selection} (expected 'all' or 'semantic')")


@torch.no_grad()
def repair_point(engine: RepairEngine, hparams: ASTRAHyperParams, input_ids, target_label: int) -> dict:
    """Close the contrast gap of one failing token; returns what was done."""
    pool = candidate_layers(engine, hparams)
    layers = None
    record = {"target": target_label, "rounds": 0, "layers": [], "neurons": 0, "fixed": False}

    for round_index in range(hparams.max_rounds):
        keys, scale, logits = engine.observe(input_ids, pool)
        others = logits.clone()
        others[target_label] = float("-inf")
        other_label = int(others.argmax())
        gap = logits[other_label] - logits[target_label] + hparams.margin
        if round_index == 0:
            record["argmax"] = other_label
        if gap <= 0:
            record["fixed"] = True
            break

        contrast = engine.contrast(target_label, other_label)
        direction = engine.semantic_delta(target_label, other_label).float()
        pairing = contrast @ direction
        if pairing <= 0:
            logger.warning("the semantic direction does not raise the contrast; point skipped")
            break
        direction = direction / pairing

        deltas = neuron_semantic_deltas(engine, input_ids, target_label, other_label, pool, (keys, scale, logits))
        if layers is None:
            layers = select_layers(hparams, pool, deltas)
            record["layers"] = layers

        weighted: Dict[int, torch.Tensor] = {}
        total = 0.0
        for layer in layers:
            u = preservation_weights(engine, hparams, layer, keys[layer])
            mask = select_neurons(hparams, u, deltas[layer])
            u = u * mask
            weighted[layer] = u
            total = total + keys[layer] @ u
        if total <= 0:
            break

        step = gap * scale / total
        changed = 0
        for layer in layers:
            columns = weighted[layer].nonzero(as_tuple=True)[0]
            if columns.numel() == 0:
                continue
            engine.backup(layer)
            weight = engine.weight(layer)
            update = direction[:, None] * (step * weighted[layer][columns])[None, :]
            # the sum is formed in float32: a small step would round away in the precision of the weights
            weight[:, columns] = (weight[:, columns].float() + update).to(weight.dtype)
            changed += int(columns.numel())
        record["rounds"] = round_index + 1
        record["neurons"] = max(record["neurons"], changed)
    else:
        logits = engine.observe(input_ids, [])[2]
        record["fixed"] = bool(int(logits.argmax()) == target_label)
    return record


@torch.no_grad()
def repair_joint(engine: RepairEngine, hparams: ASTRAHyperParams, sources, target_ids: List[int]) -> dict:
    """Close the contrast gaps of all failing answer tokens at once.

    Token t has a contrast covector c_t, a direction d_t = G^{-1} c_t (c_t . d_t = 1), the keys k_t that its
    position feeds the FFNs, and a gap g_t. The update of the selected neurons that closes every gap with minimum
    D-weighted norm is  dW_l = sum_s lambda_s d_s (x) (D_l k_{l,s}), where lambda solves the T x T system
    A lambda = g,  A_ts = (c_t . d_s) * sum_l k_{l,t}^T D_l k_{l,s} / scale_t.  The rank per layer is at most T.
    """
    pool = candidate_layers(engine, hparams)
    device = engine.device
    sources = [sources] if torch.is_tensor(sources) else list(sources)
    tgt_one = torch.tensor(target_ids, device=device)
    tgt = tgt_one.repeat(len(sources))
    arange = torch.arange(len(tgt), device=device)
    record = {"tokens": len(target_ids), "rounds": 0, "layers": [], "neurons": 0, "fixed": False, "solved": []}
    layers, masks = None, None

    for round_index in range(hparams.max_rounds):
        capture = pool if masks is None else layers          # once the selection is fixed only its layers are read
        observed = [engine.observe_answer(source_ids, tgt_one, capture) for source_ids in sources]
        keys = {layer: torch.cat([o[0][layer] for o in observed]) for layer in capture}
        scale = torch.cat([o[1] for o in observed])
        logits = torch.cat([o[2] for o in observed])
        del observed
        others = logits.clone()
        others[arange, tgt] = float("-inf")
        other_logit, other = others.max(-1)
        gap = other_logit - logits[arange, tgt] + hparams.margin
        if round_index == 0:
            record["failing"] = int((logits.argmax(-1) != tgt).sum())
        active = (gap > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            record["fixed"] = True
            break
        if active.numel() > hparams.max_focus_tokens:
            active = active[torch.topk(gap[active], hparams.max_focus_tokens).indices]
        record["solved"].append(int(active.numel()))
        del others, logits

        C = engine.contrasts(tgt[active], other[active]).float()
        Dirs = engine.semantic_deltas(tgt[active], other[active]).float()
        pairing = (C * Dirs).sum(-1)
        good = pairing > 0
        if not bool(good.all()):
            active, C, Dirs, pairing = active[good], C[good], Dirs[good], pairing[good]
            if active.numel() == 0:
                break
        Dirs = Dirs / pairing[:, None]
        s_act, g_act = scale[active], gap[active]

        selected = {layer: keys[layer][active] for layer in capture}            # [t, n]
        if layers is None:
            # layers and neurons are chosen in the first round only
            if hparams.attribution == "direct":
                deltas = {layer: selected[layer] * (C @ engine.weight(layer).detach().float()) / s_act[:, None]
                          for layer in pool}
                neuron_scores = {layer: deltas[layer].abs().sum(0) for layer in pool}
            else:                                              # 'ixg' or 'path': one pass with gradients
                neuron_scores = attribution_scores(engine, hparams, sources, tgt_one, active, other, pool)
            score = {layer: float(neuron_scores[layer].sum()) for layer in pool}
            number = max(1, round(len(pool) * hparams.layer_ratio))
            layers = sorted(sorted(pool, key=lambda l: score[l], reverse=True)[:number]) if number < len(pool) else pool
            record["layers"] = layers
            if hparams.neuron_selection != "all":
                width = next(iter(neuron_scores.values())).numel()
                budget = neuron_budget(hparams, width, len(target_ids), len(layers))
                record["neurons_per_layer"] = budget
                record["parameters"] = len(layers) * budget * engine.weight(layers[0]).shape[0]
                masks = {layer: top_k_mask(neuron_scores[layer], budget).float() for layer in layers}
            del neuron_scores

        K = torch.zeros(active.numel(), active.numel(), device=device)
        parts = {}
        for layer in layers:
            key = selected[layer]
            U = preserved_keys(engine, hparams, layer, key)
            # without masks (neuron_selection 'all') every neuron may change
            mask = masks[layer] if masks is not None else torch.ones(key.shape[-1], device=device)
            U = U * mask
            K += (key * mask) @ U.T
            parts[layer] = U
        A = (C @ Dirs.T) * K / s_act[:, None]
        ridge = hparams.ridge * A.diagonal().abs().mean()
        lam = torch.linalg.solve(A + ridge * torch.eye(A.shape[0], device=device), g_act)

        changed = 0
        for layer in layers:
            U = parts[layer]
            columns = (U.abs().sum(0) > 0).nonzero(as_tuple=True)[0]
            if columns.numel() == 0:
                continue
            engine.backup(layer)
            weight = engine.weight(layer)
            update = Dirs.T @ (lam[:, None] * U[:, columns])                       # [hidden, columns]
            weight[:, columns] = (weight[:, columns].float() + update).to(weight.dtype)
            changed += int(columns.numel())
        record["rounds"] = round_index + 1
        record["neurons"] = max(record["neurons"], changed)
    else:
        logits = torch.cat([engine.observe_answer(source_ids, tgt_one, [])[2] for source_ids in sources])
        record["fixed"] = bool((logits.argmax(-1) == tgt).all())
    if os.environ.get("GAP_GEOMETRY_LOG") and engine.weights_copy:
        log_patch_geometry(engine, record)
    return record


def attribution_scores(engine, hparams, sources, tgt_one, active, other, pool) -> dict:
    """{layer: [neurons]} attribution of the failing tokens by a pass with gradients: Input x Gradient of the
    contrast (`path`) or of the cross-entropy of the answer (`ixg`), summed over the answer positions."""
    T = len(tgt_one)
    scores = {layer: 0.0 for layer in pool}
    final_views = min(hparams.attribution_views, len(sources))
    for v in range(final_views):
        local = [i - v * T for i in active.tolist() if v * T <= i < (v + 1) * T]
        if not local:
            continue
        local = torch.tensor(local, device=engine.device)
        source_ids = sources[v]
        start = source_ids.shape[1] - 1
        keys, outs = {}, {}

        def pre(layer):
            def hook(_module, inputs):
                keys[layer] = inputs[0][0, start:start + T].detach().float()
            return hook

        def post(layer):
            def hook(_module, _inputs, output):
                outs[layer] = output.detach().requires_grad_(True)
                return outs[layer]
            return hook

        handles = []
        for layer in pool:
            handles.append(engine.ff_out(layer).register_forward_pre_hook(pre(layer)))
            handles.append(engine.ff_out(layer).register_forward_hook(post(layer)))
        try:
            with torch.enable_grad():
                ids = torch.cat([source_ids, tgt_one.reshape(1, -1).to(source_ids.dtype)], dim=-1)
                logits = engine.model(input_ids=ids).logits[0, start:start + T].float()[local]
                target = tgt_one[local]
                if hparams.attribution == "path":
                    rival = other[v * T + local]
                    objective = (logits.gather(1, target[:, None]) - logits.gather(1, rival[:, None])).sum()
                else:
                    objective = -torch.nn.functional.cross_entropy(logits, target, reduction="sum")
                objective.backward()
        finally:
            for handle in handles:
                handle.remove()
        with torch.no_grad():
            for layer in pool:
                grad = outs[layer].grad[0, start:start + T].float()
                scores[layer] = scores[layer] + (keys[layer] * (grad @ engine.weight(layer).detach().float())).abs().sum(0)
        del outs, keys, logits
    return scores


def preserved_keys(engine, hparams, layer: int, key: torch.Tensor) -> torch.Tensor:
    """U = D key for one layer (the preservation weights of the null space)."""
    if hparams.null_space == "none":
        return key.clone()
    if hparams.null_space == "diagonal":
        moments = engine.model._key_second_moments[1][layer].to(key.device)
        return key / (moments + 1e-3 * moments.mean())
    raise ValueError(f"unknown null_space: {hparams.null_space}")


def patch_scores(engine, hparams, selected, C, Dirs, s_act, g_act, pool):
    """{layer: [neurons]}: the predicted gain, the size and the mutual cosine of the patches that the neurons of every
    layer would carry in the unrestricted minimum-norm solution (every neuron may change).

    The solution gives each neuron a tailored patch dw_j = D_j sum_s lambda_s k_{s,j} d_s (a mix of the directions of
    the tokens it is active on); its first-order effect on the failure, through the direct path, is
    sum_s k_{s,j} (c_s . dw_j) / scale_s."""
    device = Dirs.device
    K = torch.zeros(C.shape[0], C.shape[0], device=device)
    for layer in pool:
        K += selected[layer] @ preserved_keys(engine, hparams, layer, selected[layer]).T
    A = (C @ Dirs.T) * K / s_act[:, None]
    lam = torch.linalg.solve(A + hparams.ridge * A.diagonal().abs().mean() * torch.eye(A.shape[0], device=device),
                             g_act)
    gain, size, cosine = {}, {}, {}
    for layer in pool:
        key = selected[layer]
        patch = Dirs.T @ (lam[:, None] * preserved_keys(engine, hparams, layer, key))          # [hidden, neurons]
        gain[layer] = (key * (C @ patch) / s_act[:, None]).sum(0)
        size[layer] = patch.norm(dim=0)
        top = torch.topk(gain[layer].clamp(min=0), min(32, patch.shape[1])).indices
        unit = patch[:, top] / (patch[:, top].norm(dim=0, keepdim=True) + 1e-12)
        gram = unit.T @ unit
        cosine[layer] = float((gram.sum() - gram.diagonal().sum()) / max(gram.numel() - gram.shape[0], 1))
    return gain, size, cosine


@torch.no_grad()
def log_patch_geometry(engine, record) -> None:
    """One JSON line per repair: how the final patch sits on the neurons it changed. `parallel` is the mean |cos|
    between the change of a write vector and the write vector itself (1: the neuron is only rescaled, 0: it is
    turned), `relative` the mean ratio of their norms."""
    parallel, relative = [], []
    for name, original in engine.weights_copy.items():
        layer = int(name.split(".layers.")[1].split(".")[0])
        weight = engine.weight(layer).detach().float()
        change = weight - original.to(weight.device).float()
        columns = (change.abs().sum(0) > 0).nonzero(as_tuple=True)[0]
        if columns.numel() == 0:
            continue
        w, d = original.to(weight.device).float()[:, columns], change[:, columns]
        parallel.append(float(((w * d).sum(0).abs() / (w.norm(dim=0) * d.norm(dim=0) + 1e-12)).mean()))
        relative.append(float((d.norm(dim=0) / (w.norm(dim=0) + 1e-12)).mean()))
    if parallel:
        with open(os.environ["GAP_GEOMETRY_LOG"], "a") as f:
            f.write(json.dumps({"parallel": sum(parallel) / len(parallel), "relative": sum(relative) / len(relative),
                                "tokens": record.get("tokens")}) + "\n")


def repair_views(data: dict, hparams: ASTRAHyperParams) -> List[str]:
    """The renderings of the repair prompt that are solved together (the original prompt first)."""
    pool_views = [data["question"]] + list(data.get("views", []))
    return pool_views[: hparams.views]


def _ranks(x: torch.Tensor) -> torch.Tensor:
    return x.argsort().argsort().float()


def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = _ranks(a), _ranks(b)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def _jaccard(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    left, right = set(torch.topk(a, k).indices.tolist()), set(torch.topk(b, k).indices.tolist())
    return len(left & right) / len(left | right)


@torch.no_grad()
def analyse_gap(engine: RepairEngine, hparams: ASTRAHyperParams, sources, target_ids: List[int]):
    """RQ1: how far the neurons that a failure points at are from the neurons that carry the patch.

    In the layers that contrastive semantics selects, and for the neurons of the unrestricted minimum-norm solution
    (every neuron may change), it reports: the rank correlation of the failure attribution (contrastive semantics) with the
    patch gain and with the patch size; the overlap of the top `neuron_num` sets; the mean cosine between the
    patches of different neurons (tailoring); and, after one round of the repair, how much the top set of the failure
    attribution has moved (the localisation depends on the patch). The weights are changed by the round: the caller
    restores them with `engine.weights_copy`."""
    pool = candidate_layers(engine, hparams)
    device = engine.device
    sources = list(sources)
    tgt_one = torch.tensor(target_ids, device=device)
    tgt = tgt_one.repeat(len(sources))
    arange = torch.arange(len(tgt), device=device)

    def measure():
        observed = [engine.observe_answer(source_ids, tgt_one, pool) for source_ids in sources]
        keys = {layer: torch.cat([o[0][layer] for o in observed]) for layer in pool}
        scale, logits = torch.cat([o[1] for o in observed]), torch.cat([o[2] for o in observed])
        others = logits.clone()
        others[arange, tgt] = float("-inf")
        other_logit, other = others.max(-1)
        return keys, scale, other, other_logit - logits[arange, tgt] + hparams.margin

    keys, scale, other, gap = measure()
    active = (gap > 0).nonzero(as_tuple=True)[0]
    if active.numel() < 2:
        return None
    C = engine.contrasts(tgt[active], other[active]).float()
    Dirs = engine.semantic_deltas(tgt[active], other[active]).float()
    pairing = (C * Dirs).sum(-1)
    good = pairing > 0
    active, C, Dirs, pairing = active[good], C[good], Dirs[good], pairing[good]
    Dirs = Dirs / pairing[:, None]
    s_act, g_act = scale[active], gap[active]
    selected = {layer: keys[layer][active] for layer in pool}
    failure = {layer: (selected[layer] * (C @ engine.weight(layer).detach().float()) / s_act[:, None]).abs().sum(0)
               for layer in pool}
    number = max(1, round(len(pool) * hparams.layer_ratio))
    layers = sorted(sorted(pool, key=lambda l: float(failure[l].sum()), reverse=True)[:number])
    gains, mass, cosine = patch_scores(engine, hparams, selected, C, Dirs, s_act, g_act, pool)
    k = neuron_budget(hparams, failure[layers[0]].numel(), len(target_ids), len(layers))
    stats = {
        "tokens": int(active.numel()),
        "spearman_gain": sum(_spearman(failure[l], gains[l]) for l in layers) / len(layers),
        "spearman_size": sum(_spearman(failure[l], mass[l]) for l in layers) / len(layers),
        "jaccard": sum(_jaccard(failure[l], gains[l].clamp(min=0), k) for l in layers) / len(layers),
        "patch_cosine": sum(cosine[l] for l in layers) / len(layers),
    }
    # the same repair, one round: does the failure attribution still point at the same neurons?
    one = copy.copy(hparams)
    one.max_rounds, one.attribution = 1, "direct"
    repair_joint(engine, one, sources, target_ids)
    keys2, scale2, other2, gap2 = measure()
    after = {layer: (keys2[layer][active] * (C @ engine.weight(layer).detach().float()) / scale2[active][:, None]
                     ).abs().sum(0) for layer in layers}
    stats["shift"] = 1.0 - sum(_jaccard(failure[l], after[l], k) for l in layers) / len(layers)
    return stats


def apply_ASTRA_to_model(model, tok, hparams: ASTRAHyperParams, batch_data: list):
    """Repair the model on one sample; returns the original weights of the changed parameters."""
    engine = RepairEngine.from_hparams(model, tok, hparams)
    data = batch_data[0]
    target_ids = tok(data["answer"], add_special_tokens=False)["input_ids"][: hparams.max_focus_tokens]
    try:
        if hparams.solve == "joint":
            record = repair_joint(engine, hparams, [engine.encode(v) for v in repair_views(data, hparams)], target_ids)
            engine.report["points"].append(record)
            engine.report["skipped"] += 0 if record["fixed"] else 1
            return engine.finish()
        for input_ids, target_label, _argmax_label in engine.failing_points(data["question"], target_ids):
            record = repair_point(engine, hparams, input_ids, target_label)
            engine.report["points"].append(record)
            if not record["fixed"]:
                engine.report["skipped"] += 1
    except Exception:
        engine.rollback()
        engine.finish()
        raise
    return engine.finish()
