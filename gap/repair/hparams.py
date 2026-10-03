"""Hyperparameters of the methods, one dataclass per implementation and one JSON file per method and model.

`load_hparams("STAR", "oxcoder-9b")` reads `hparams/STAR/oxcoder-9b.json`; a model without a file of its own
gets `default.json`. Module templates in the files are written for the plain decoder layout;
`layout.resolve_hparams_modules` rewrites them to the loaded checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

HPARAMS_DIR = Path(__file__).resolve().parent / "hparams"


@dataclass
class HyperParams:
    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            return cls(**json.load(f))


@dataclass
class STARHyperParams(HyperParams):
    # model description
    model_name: str
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    lm_head_module: str = "lm_head"

    # which tokens of the answer to repair (all focus tokens are solved for jointly)
    max_focus_tokens: int = 30

    # share of layers that are repaired: those whose FFN the expected token relies on most (1.0 = all layers); the same
    # share as ASTRA
    layer_ratio: float = 0.25

    # optimization
    epoch_num: int = 8
    lr: float = 1e-2
    scale: int = 1                # multiplies lr

    # The neuron activations of a token are read at the position that predicts it, as the input of the FFN output
    # projection, i.e. what actually multiplies the write vectors (for a gated FFN: act_fn(gate) * up).

    # preservation, as for ASTRA: 'diagonal' weights the norm of the change of a neuron by the inverse of its mean
    # squared activation on reference prompts; 'none' is the plain minimum-norm prior
    null_space: str = "diagonal"


@dataclass
class ASTRAHyperParams(HyperParams):
    """ASTRA: analytic repair; the switches below are its components (see astra.py)."""
    # model description
    model_name: str
    rewrite_module_tmp: str = "model.layers.{}.mlp.down_proj"
    layer_module_tmp: str = "model.layers.{}"
    lm_head_module: str = "lm_head"

    # which tokens of the answer to repair (all of them)
    max_focus_tokens: int = 1000

    # layers: the share whose FFN has the largest contrastive semantics score
    layer_ratio: float = 0.25

    # neurons per repaired layer: 'semantic' (top by the attribution below) or 'all' (the plain minimum-norm solution)
    neuron_selection: str = "semantic"
    neuron_num: int = 64
    # budget in proportion to the answer: with `tokens_ratio` R > 0 each layer gets ceil(R * T / layers) neurons, T the
    # number of answer tokens, bounded below by `neuron_min` and above by `neuron_max_share` of the width; R = 0 uses
    # `neuron_num` as given
    tokens_ratio: float = 0.0
    neuron_min: int = 16
    neuron_max_share: float = 0.04

    # what the update is asked to preserve: 'diagonal' weights the norm of the change of a neuron by the inverse of
    # its mean squared activation on reference prompts, 'none' is the minimum Euclidean norm
    null_space: str = "diagonal"

    # the solve: the target must lead the best other token by `margin` logits; the linearisation is refreshed up to
    # `max_rounds` times
    margin: float = 5.0
    max_rounds: int = 5

    # 'joint': all failing tokens of the answer at once, one T x T linear system per round (`ridge` regularises it);
    # 'sequential': one failing token at a time (walk over the answer)
    solve: str = "joint"
    ridge: float = 1e-2

    # views of the repair pair: renderings of the repair prompt whose constraints are solved together, so that the
    # update does not depend on the wording: the first `views` of [original] + sample["views"]
    views: int = 2

    # attribution that picks layers and neurons, in the first round only: 'direct' (contrastive semantics, forward only),
    # 'path' (the same contrast attributed through the layers above, one backward pass) or 'ixg' (Input x Gradient of
    # the answer cross-entropy)
    attribution: str = "direct"
    attribution_views: int = 1        # views of the backward pass of 'path' and 'ixg'


@dataclass
class AlphaEditHyperParams(HyperParams):
    # Method
    model_name: str
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str
    window_size: int
    overlap: int
    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float


@dataclass
class LoRAHyperParams(HyperParams):
    model_name: str

    # low-rank adapters on the FFN output projection, the matrix that ASTRA, STAR and AlphaEdit change
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple = ("down_proj",)

    # one update per sample: language-modeling loss over prompt + answer, AdamW
    lr: float = 3e-4
    steps: int = 10
    max_length: int = 2048


@dataclass
class NoHyperParams(HyperParams):
    """The unchanged model."""
    model_name: str


def load_hparams(method: str, model_key: str):
    """Hyperparameters of `method` (a key of `gap.repair.METHODS`) for a model key of the registry."""
    from . import METHODS

    name = str(model_key).rstrip("/").split("/")[-1]
    path = HPARAMS_DIR / method / f"{name.lower()}.json"
    if path.is_file():
        return METHODS[method].hparams_class.from_json(path)
    # any other model (a registry key, a Hub id or a local path): the defaults, written for the plain decoder
    # layout; module paths and layer indices are fitted to the loaded model by `Method.prepare`
    hparams = METHODS[method].hparams_class.from_json(HPARAMS_DIR / method / "default.json")
    hparams.model_name = name
    return hparams
