"""Techniques that change a language model, behind one interface.

    method = build_method("ASTRA", "oxcoder-9b")
    model = method.prepare(model, tok)                    # once per model
    state = method.apply(model, tok, {"question": ..., "answer": ...})
    ...evaluate the changed model...
    method.restore(model, state)                          # back to the original model

| name                      | what it is                                                                             |
| ------------------------- | -------------------------------------------------------------------------------------- |
| ASTRA                     | joint closed-form repair of carrier neurons selected by contrastive semantics          |
| STAR                      | prior-guided gradient repair of FFN write vectors (baseline)                           |
| AlphaEdit                 | null-space constrained editing (baseline)                                              |
| LowRank                   | low-rank adapters (LoRA, default configuration) on the FFN output projection down_proj |
| None                      | the unchanged model                                                                    |

ASTRA, STAR and AlphaEdit change `mlp.down_proj`; `update_mass` reports where.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from .hparams import (
    AlphaEditHyperParams, LoRAHyperParams, NoHyperParams, STARHyperParams, ASTRAHyperParams, load_hparams,
)


@dataclass(frozen=True)
class MethodSpec:
    hparams_class: type
    kind: str                      # 'weights': changes mlp.down_proj | 'adapter' | 'none'


METHODS = {
    "ASTRA": MethodSpec(ASTRAHyperParams, "weights"),
    "STAR": MethodSpec(STARHyperParams, "weights"),
    "AlphaEdit": MethodSpec(AlphaEditHyperParams, "weights"),
    "LowRank": MethodSpec(LoRAHyperParams, "adapter"),
    "None": MethodSpec(NoHyperParams, "none"),
}


def _apply_function(name: str) -> Callable:
    if name == "STAR":
        from .star import apply_STAR_to_model
        return apply_STAR_to_model
    if name == "ASTRA":
        from .astra import apply_ASTRA_to_model
        return apply_ASTRA_to_model
    if name == "AlphaEdit":
        from .alphaedit.main import apply_AlphaEdit_to_model
        return apply_AlphaEdit_to_model
    raise KeyError(name)


@torch.no_grad()
def restore_weights(model, weights_copy: dict) -> None:
    """Write the original weights, as returned by an `apply_*_to_model` function, back into the model."""
    for name, original in weights_copy.items():
        parameter = model.get_parameter(name)
        parameter[...] = original.to(parameter.device)


@torch.no_grad()
def update_mass(model, weights_copy: dict) -> dict:
    """L2 norm of the update of every neuron's write vector: {parameter name: tensor[neurons]}.

    Call it after `apply` and before `restore`. The parameters are FFN output weights ([hidden, neurons]),
    so a column is the write vector of one neuron; this locates an update for any weight-changing method.
    """
    mass = {}
    for name, original in weights_copy.items():
        parameter = model.get_parameter(name)
        delta = parameter.detach().float() - original.to(parameter.device).float()
        mass[name] = torch.linalg.vector_norm(delta, dim=0).cpu()
    return mass


class Method:
    def __init__(self, name: str, model_key: str):
        if name not in METHODS:
            raise KeyError(f"unknown method {name!r}; available: {sorted(METHODS)}")
        self.name = name
        self.spec = METHODS[name]
        self.model_key = model_key
        self.hparams = load_hparams(name, model_key)
        self._projection = None

    def override(self, **values) -> "Method":
        """Replace hyperparameters by name (unknown names are an error): the switch for variants of a method."""
        for name, value in values.items():
            if not hasattr(self.hparams, name):
                raise AttributeError(f"{self.name} has no hyperparameter {name!r}")
            setattr(self.hparams, name, value)
        return self

    def dryrun(self) -> "Method":
        """Shrink every budget so that one sample takes seconds; the code path stays the same."""
        small = {
            "max_focus_tokens": 8, "epoch_num": 2, "max_rounds": 2, "v_num_grad_steps": 5, "mom2_n_samples": 200,
            "steps": 2, "max_length": 512,
        }
        for name, value in small.items():
            if hasattr(self.hparams, name):
                setattr(self.hparams, name, min(value, getattr(self.hparams, name)))
        if hasattr(self.hparams, "mom2_dataset"):
            self.hparams.mom2_dataset = "wikitext"          # the small corpus; Wikipedia is tens of gigabytes
        return self

    def prepare(self, model, tok, reference_prompts=None):
        """Everything that is done once per model; returns the model to use from here on.

        `reference_prompts`: text whose behaviour a repair should preserve (used by ASTRA and STAR with
        `null_space='diagonal'`).
        """
        if self.spec.kind == "weights":
            from .layout import fit_layer_indices, resolve_hparams_modules
            resolve_hparams_modules(model, self.hparams)
            fit_layer_indices(model, self.hparams)
        if self.name == "AlphaEdit":
            from .alphaedit.main import null_space_projection
            self._projection = null_space_projection(model, tok, self.hparams)
        if self.name in ("ASTRA", "STAR"):
            if self.hparams.null_space == "diagonal":
                if not reference_prompts:
                    raise ValueError("ASTRA with null_space='diagonal' needs reference_prompts")
                from .engine import RepairEngine
                from .semantics import key_second_moments
                key_second_moments(RepairEngine.from_hparams(model, tok, self.hparams), list(reference_prompts))
        if self.spec.kind == "adapter":
            from . import lora
            model = lora.attach(model, self.hparams)
        return model

    def apply(self, model, tok, sample: dict):
        """Change the model for one sample (`question`, `answer`); returns the state `restore` needs."""
        if self.spec.kind == "none":
            return None
        if self.spec.kind == "adapter":
            from . import lora
            lora.reset(model)
            lora.update(model, tok, self.hparams, sample)
            return None
        apply = _apply_function(self.name)
        if self._projection is not None:
            return apply(model, tok, self.hparams, [sample], P=self._projection)
        return apply(model, tok, self.hparams, [sample])

    def restore(self, model, state) -> None:
        if self.spec.kind == "adapter":
            from . import lora
            lora.reset(model)
        elif state:
            restore_weights(model, state)

    def update_mass(self, model, state) -> dict:
        """Where the latest `apply` changed the model, per neuron (weight-changing methods and LoRA)."""
        if self.spec.kind == "adapter":
            from . import lora
            return {
                name: torch.linalg.vector_norm(delta, dim=0).cpu() for name, delta in lora.weight_deltas(model).items()
            }
        return update_mass(model, state) if state else {}


def build_method(name: str, model_key: str) -> Method:
    return Method(name, model_key)


__all__ = ["METHODS", "Method", "build_method", "restore_weights", "update_mass"]
