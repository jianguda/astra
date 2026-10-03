"""Frozen-LM runtime: model and tokenizer loading, layout discovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from gap.utils.numerics import (
    ANALYSIS_DTYPE,
    prepare_model_precision,
    select_device,
)
from gap.utils.settings import HF_HUB_CACHE
from gap.utils.experiment import freeze


def tokenizer_loading_policy(model_name: str) -> dict[str, object]:
    """Return the checkpoint-family tokenizer policy.

    Tokenization is part of the input protocol, not a cosmetic loader choice, so family-specific corrections are
    kept explicit.
    """
    name = str(model_name).lower()
    if "opencoder" in name or name.startswith("infly/"):
        # the tokenizer class (INFLMTokenizer) ships with the checkpoint
        return {
            "policy": "opencoder-remote-tokenizer-v1",
            "trust_remote_code": True,
        }
    return {"policy": "default-auto-v1"}


def load_tokenizer(model_name: str, *, revision: str | None = None):
    """Load the tokenizer of one checkpoint; `model_name` determines the family policy."""
    from transformers import AutoTokenizer

    policy = tokenizer_loading_policy(model_name)
    kwargs = {"cache_dir": str(HF_HUB_CACHE)}
    if revision:
        kwargs["revision"] = revision
    if bool(policy.get("trust_remote_code", False)):
        kwargs["trust_remote_code"] = True
    tok = AutoTokenizer.from_pretrained(str(model_name), **kwargs)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def _model_loader_class(config):
    """Resolve the checkpoint's declared Transformers model class when possible.

    Recent open-weight families increasingly publish a text decoder inside a
    conditional-generation wrapper (OxCoder-9B; OpenCoder and MiniCPM5 use direct
    causal-LM layouts).  Relying only on ``AutoModelForCausalLM`` therefore rejects
    otherwise perfectly usable text backbones.  Prefer the architecture declared by
    the checkpoint and fall back to the causal/multimodal Auto classes.
    """
    import transformers

    architectures = tuple(getattr(config, "architectures", None) or ())
    for name in architectures:
        cls = getattr(transformers, str(name), None)
        if cls is not None and hasattr(cls, "from_pretrained"):
            return cls, str(name)

    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM, "AutoModelForCausalLM"


def load_model(
    model_name: str,
    *,
    device: torch.device | str | None = None,
    revision: str | None = None,
):
    """Load a frozen LM checkpoint through its declared HF architecture.

    Only the language backbone is used, but some releases package that backbone
    in a multimodal conditional-generation wrapper.  We therefore load the
    checkpoint's declared class and let ``discover_model_layout`` select the text
    decoder.  The vision branch is never executed.
    """
    from transformers import AutoConfig
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()          # the weight-loading bar floods non-interactive logs
    device = select_device(device)
    common = {"cache_dir": str(HF_HUB_CACHE)}
    if revision:
        common["revision"] = revision
    config = AutoConfig.from_pretrained(model_name, **common)
    loader, loader_name = _model_loader_class(config)
    quant = getattr(config, "quantization_config", None)
    quant_method = None if quant is None else str(
        quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", "")).lower()

    # Preserve the checkpoint's stored dtype during loading.  The precision
    # policy is applied only after loading so FP32 checkpoints are never
    # silently downcast.  Eager attention remains the conservative path for MPS.
    kwargs = {
        **common,
        # Load MPS checkpoints directly in FP32 instead of loading a lower-
        # precision copy and expanding it afterwards.
        "dtype": torch.float32 if device.type == "mps" else "auto",
        "low_cpu_mem_usage": True,
    }
    # On CUDA, stream checkpoint shards directly to the target GPU instead of
    # first materializing the complete model on host RAM and then calling
    # ``model.to(cuda)``.  This does not change model precision or execution; it
    # only removes an avoidable host-memory peak.
    if device.type == "cuda":
        target_cuda = torch.cuda.current_device() if device.index is None else int(device.index)
        kwargs["device_map"] = {"": int(target_cuda)}
    if device.type == "mps":
        kwargs["attn_implementation"] = "eager"
    if quant_method:
        raise RuntimeError(f"unsupported quantized checkpoint ({quant_method}) for {model_name}; exact weights are needed")

    try:
        model = loader.from_pretrained(model_name, **kwargs)
    except (ValueError, TypeError) as first_error:
        # A small number of Transformers releases do not export the declared
        # architecture at top level even though an Auto multimodal mapping is
        # available.  Use that mapping as the alternate current Auto loader.
        import transformers
        auto_mm = getattr(transformers, "AutoModelForMultimodalLM", None)
        if auto_mm is None or loader_name == "AutoModelForMultimodalLM":
            raise
        try:
            model = auto_mm.from_pretrained(model_name, **kwargs)
            loader_name = "AutoModelForMultimodalLM"
        except Exception:
            raise first_error

    config = model.config
    packed = sum(int(p.numel()) for p in model.parameters() if not torch.is_floating_point(p))
    if packed:
        raise RuntimeError(f"{model_name}: {packed} non-floating (packed/quantized) parameters remain after loading")
    model, precision = prepare_model_precision(model, device)
    precision["revision"] = getattr(config, "_commit_hash", None) or revision
    precision["loader"] = loader_name
    freeze(model)
    print(
        "model precision: "
        f"parameters/execution={precision['execution_dtype']}, "
        f"analysis={precision['analysis_dtype']}, "
        f"loader={loader_name}"
    )
    return config, model, precision


@dataclass(frozen=True)
class ModelLayout:
    """Minimal architecture adapter for the residual-stream contract.

    The repair code depends on embeddings, repeated residual blocks, one final
    normalization, and a vocabulary readout.  Attribute paths are architecture
    details and belong here rather than in the methods.
    """

    name: str
    backbone: torch.nn.Module
    embedding: torch.nn.Module
    layers: Any
    final_norm: torch.nn.Module
    lm_head: torch.nn.Module


def _candidate_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    """Architecture paths that may own a text decoder or output head."""
    paths = (
        "",
        "model",
        "model.language_model",
        "model.language_model.model",
        "language_model",
        "language_model.model",
        "model.text_model",
        "text_model",
    )
    out: list[tuple[str, torch.nn.Module]] = []
    seen: set[int] = set()
    for path in paths:
        obj: object = model
        if path:
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
        if isinstance(obj, torch.nn.Module) and id(obj) not in seen:
            seen.add(id(obj))
            out.append((path or "root", obj))
    return out


def _output_embedding_module(model: torch.nn.Module) -> torch.nn.Module | None:
    """Resolve the language vocabulary head through public HF APIs first."""
    for _path, host in _candidate_modules(model):
        getter = getattr(host, "get_output_embeddings", None)
        if callable(getter):
            try:
                head = getter()
            except Exception:
                head = None
            if isinstance(head, torch.nn.Module):
                return head
        for name in ("lm_head", "embed_out"):
            head = getattr(host, name, None)
            if isinstance(head, torch.nn.Module):
                return head
    return None


def discover_model_layout(model: torch.nn.Module) -> ModelLayout:
    """Resolve the language residual-stream layout without model-name tests.

    Supported wrappers include direct decoder-only causal LMs, GPT-NeoX, and
    modern conditional-generation checkpoints whose text decoder sits under
    ``model.language_model`` (or an equivalent shallow wrapper path).
    """
    head = _output_embedding_module(model)

    for path, body in _candidate_modules(model):
        if (
            hasattr(body, "embed_tokens")
            and hasattr(body, "layers")
            and hasattr(body, "norm")
            and head is not None
        ):
            return ModelLayout(
                name="decoder_model" if path == "model" else f"decoder:{path}",
                backbone=body,
                embedding=body.embed_tokens,
                layers=body.layers,
                final_norm=body.norm,
                lm_head=head,
            )

    body = getattr(model, "gpt_neox", None)
    if (
        body is not None
        and hasattr(body, "embed_in")
        and hasattr(body, "layers")
        and hasattr(body, "final_layer_norm")
        and head is not None
    ):
        return ModelLayout(
            name="gpt_neox",
            backbone=body,
            embedding=body.embed_in,
            layers=body.layers,
            final_norm=body.final_layer_norm,
            lm_head=head,
        )

    candidates = ", ".join(path for path, _ in _candidate_modules(model))
    raise RuntimeError(
        "Unsupported language residual layout. Expected a decoder exposing "
        "{embed_tokens,layers,norm} (directly or inside a conditional-generation "
        "wrapper) or GPT-NeoX {embed_in,layers,final_layer_norm}, plus a vocabulary "
        f"output head. Checked module paths: {candidates}."
    )

def fixed_norm_gain(norm: torch.nn.Module) -> torch.Tensor:
    """Return the *effective* multiplicative gain of the final normalization.

    Most RMSNorm/LayerNorm modules store the gain directly.  Gemma-family and
    Qwen3.5 RMSNorm store a zero-centered offset and apply ``1 + weight`` in
    ``forward``.  The decoder atoms need the functional gain, not the raw
    parameter tensor.
    """
    if not hasattr(norm, "weight") or norm.weight is None:
        raise RuntimeError(f"Final norm {norm.__class__.__name__} has no learnable weight.")
    raw = norm.weight.detach().to(dtype=ANALYSIS_DTYPE)
    class_name = norm.__class__.__name__.lower()
    unit_offset = bool(getattr(norm, "unit_offset", False))
    # Gemma and Qwen3.5 (and Qwen3-Next) store a zero-centered offset and apply `1 + weight`
    if unit_offset or (("gemma" in class_name or "qwen3_5" in class_name or "qwen3next" in class_name)
                       and "rmsnorm" in class_name):
        return raw + 1.0
    return raw


__all__ = [
    "ModelLayout", "discover_model_layout", "load_model", "load_tokenizer", "tokenizer_loading_policy",
    "fixed_norm_gain",
]
