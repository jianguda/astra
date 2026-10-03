"""Canonical model aliases of the paper suite."""
from __future__ import annotations

MODEL_REGISTRY = {
    # one canonical key per checkpoint
    # large code model, loaded through its conditional-generation wrapper (text decoder only)
    "oxcoder-9b": "OrionLLM/OxCoder-9B",
    # large code model with a direct causal-LM layout
    "opencoder-8b-instruct": "infly/OpenCoder-8B-Instruct",
    # small model
    "minicpm5-1b": "openbmb/MiniCPM5-1B",
}

def _local_models() -> dict:
    """Extra models of this machine: `models.local.json` in the project root (or `GAP_MODELS_FILE`).

        {"small-a": "/path/or/hub-id"}

    The value is a Hugging Face id or a local directory (or a dictionary with that under `repo`). These models are
    for quick experiments; the paper suite is the registry above.
    """
    import json
    import os
    from pathlib import Path

    path = Path(os.environ.get("GAP_MODELS_FILE", Path(__file__).resolve().parents[2] / "models.local.json"))
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {key: ({"repo": value} if isinstance(value, str) else dict(value)) for key, value in entries.items()}


MODEL_REGISTRY.update({key: meta["repo"] for key, meta in _local_models().items()})


def artifact_model_name(model_key: str) -> str:
    """Stable filesystem name for a registered model key or Hugging Face model id."""
    name = MODEL_REGISTRY.get(str(model_key), str(model_key))
    return name.strip("/").replace("/", "__").replace(" ", "_")
