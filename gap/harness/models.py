"""Model access shared by the two protocols: the model loader, greedy generation, output paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch

from gap.utils.paths import output_dir

from .chat import format_prompt, register_tokenizer, stop_strings


def load(model_key: str, device: str = None):
    """(model, tokenizer, checkpoint name) of a registry key, loaded in BF16 on CUDA.

    The model comes back frozen; the methods switch on the gradients they need and restore the flags.
    """
    from gap.utils.experiment import resolve_model
    from gap.utils.runtime import load_model, load_tokenizer

    name = resolve_model(model_key)
    _, model, precision = load_model(name, device=device)
    tok = load_tokenizer(name, revision=precision.get("revision"))
    register_tokenizer(name, tok)
    return model, tok, name


def stop_token_ids(tok, model_name: str) -> List[int]:
    ids = []
    for text in stop_strings(model_name):
        encoded = tok.encode(text, add_special_tokens=False)
        if encoded:
            ids.append(encoded[0])
    return ids or [tok.eos_token_id]


@torch.no_grad()
def generate(model, tok, model_name: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Greedy answer of the model to `prompt` (the chat template of the model is applied here)."""
    formatted = format_prompt(model_name, prompt)
    encoded = tok([formatted if formatted is not None else prompt], return_tensors="pt", add_special_tokens=False)
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    model.eval()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=encoded["attention_mask"].to(device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=stop_token_ids(tok, model_name),
    )
    return tok.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)


DRYRUN_SAMPLES = 4
DRYRUN_NEW_TOKENS = 128
REFERENCE_PROMPTS = 32


def start(method_name: str, model_key: str, overrides: dict = None, dryrun: bool = False, device: str = None):
    """(model, tokenizer, checkpoint name, method): the method is configured before the model is loaded."""
    from gap.repair import build_method

    method = build_method(method_name, model_key).override(**(overrides or {}))
    if dryrun:
        method.dryrun()
    method.overrides = dict(overrides or {})
    model, tok, name = load(model_key, device)
    return model, tok, name, method


def run_tag(method) -> str:
    """Name of a run: the method and its overridden hyperparameters."""
    tag = method.name
    for name, value in sorted(getattr(method, "overrides", {}).items()):
        tag += f"+{name}={value}"
    if os.environ.get("GAP_SHARD"):
        k, n = os.environ["GAP_SHARD"].split("/")
        tag += f"-s{k}of{n}"
    return tag.replace("/", "_").replace(" ", "")


def parse_overrides(pairs) -> dict:
    """`--set name=value` pairs; values are read as JSON when they parse (numbers, true/false, lists)."""
    import json

    overrides = {}
    for pair in pairs or ():
        name, _, text = pair.partition("=")
        if not name or not _:
            raise ValueError(f"--set expects name=value, got {pair!r}")
        try:
            overrides[name] = json.loads(text)
        except json.JSONDecodeError:
            overrides[name] = text
    return overrides


def results_dir(model_key: str, protocol: str, corpus: str, dryrun: bool = False) -> Path:
    suffix = ("_dryrun" if dryrun else "") + (f"_{os.environ['GAP_TAG']}" if os.environ.get("GAP_TAG") else "")
    path = output_dir(model_key, f"harness_{protocol}{suffix}") / corpus
    path.mkdir(parents=True, exist_ok=True)
    return path


def top_updates(mass: dict, k: int = 100) -> dict:
    """A compact record of where a repair changed the model: total update per parameter and the top-k neurons."""
    if not mass:
        return {}
    names = sorted(mass)
    totals = {name: float(torch.linalg.vector_norm(mass[name])) for name in names}
    flat = torch.cat([mass[name].flatten() for name in names])
    owner = torch.cat([torch.full((mass[name].numel(),), i) for i, name in enumerate(names)])
    index = torch.cat([torch.arange(mass[name].numel()) for name in names])
    values, order = torch.topk(flat, min(k, flat.numel()))
    neurons = [
        {"parameter": names[int(owner[j])], "neuron": int(index[j]), "mass": float(v)}
        for v, j in zip(values, order) if float(v) > 0
    ]
    return {"total": totals, "top_neurons": neurons}
