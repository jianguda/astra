"""Central numerical policy (precisions, seeds).

Precision is intentionally asymmetric:

* frozen checkpoints execute in BF16 on supported CUDA and FP32 otherwise;
* everything a repair computes from the model (captured activations, logits,
  contrasts, the closed-form solves and the weight updates) is FP32.
"""
from __future__ import annotations

import random

import numpy as np
import torch

GLOBAL_SEED = 42

# Tensors captured from the model and ordinary linear algebra.
ANALYSIS_DTYPE = torch.float32

_SUPPORTED_MODEL_DTYPES = {torch.bfloat16, torch.float16, torch.float32}


def select_device(preferred: torch.device | str | None = None) -> torch.device:
    """Resolve an execution backend, optionally enforcing an explicit choice."""
    if preferred is not None and str(preferred).lower() != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch build/runtime.")
        if device.type not in {"cuda", "mps", "cpu"}:
            raise ValueError(f"unsupported execution device: {device}")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _primary_dtype(model: torch.nn.Module) -> torch.dtype:
    counts: dict[torch.dtype, int] = {}
    for parameter in model.parameters():
        if torch.is_floating_point(parameter):
            counts[parameter.dtype] = counts.get(parameter.dtype, 0) + parameter.numel()
    if not counts:
        raise RuntimeError("model has no floating-point parameters")
    return max(counts, key=counts.get)


def prepare_model_precision(
    model: torch.nn.Module, device: torch.device | str
) -> tuple[torch.nn.Module, dict[str, str]]:
    """Choose a stable frozen-checkpoint execution precision."""
    device = torch.device(device)
    loaded = _primary_dtype(model)
    if loaded not in _SUPPORTED_MODEL_DTYPES:
        raise RuntimeError(f"unsupported model dtype: {loaded}")

    if device.type == "cuda" and loaded == torch.bfloat16 and torch.cuda.is_bf16_supported():
        execution = torch.bfloat16
    else:
        execution = torch.float32

    if execution == torch.float32 and loaded != torch.float32:
        model = model.float()

    parameter_devices = {p.device for p in model.parameters()}

    def same_device(actual: torch.device, requested: torch.device) -> bool:
        if actual.type != requested.type:
            return False
        if actual.type != "cuda":
            return actual == requested
        requested_index = torch.cuda.current_device() if requested.index is None else requested.index
        actual_index = torch.cuda.current_device() if actual.index is None else actual.index
        return int(actual_index) == int(requested_index)

    already_resident = bool(parameter_devices) and all(same_device(d, device) for d in parameter_devices)
    if not already_resident:
        model = model.to(device)
    model = model.eval()
    actual = _primary_dtype(model)
    if actual != execution:
        raise RuntimeError(f"model dtype mismatch: expected {execution}, got {actual}")

    return model, {
        "loaded_dtype": str(loaded).removeprefix("torch."),
        "execution_dtype": str(actual).removeprefix("torch."),
        "analysis_dtype": str(ANALYSIS_DTYPE).removeprefix("torch."),
    }


def stabilize(reproducibility: bool = True, seed: int = GLOBAL_SEED) -> None:
    """Stabilize every RNG used by the paper protocol; called before every model run."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "manual_seed"):
        mps.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Transformers keeps a few library-level RNG entry points of its own.
    try:
        import transformers
    except ModuleNotFoundError:
        transformers = None
    if transformers is not None:
        transformers.set_seed(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not bool(reproducibility)
        torch.backends.cudnn.deterministic = bool(reproducibility)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    torch.use_deterministic_algorithms(bool(reproducibility))
    torch.set_float32_matmul_precision("highest")
