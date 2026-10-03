"""Filesystem layout for command-owned results and reusable caches."""
from __future__ import annotations

from pathlib import Path

from gap.utils.models import artifact_model_name
from gap.utils.settings import CACHE_ROOT, OUTPUT_ROOT


def output_dir(model_key: str, stage: str) -> Path:
    return OUTPUT_ROOT / artifact_model_name(model_key) / str(stage)


def cache_dir(model_key: str) -> Path:
    return CACHE_ROOT / artifact_model_name(model_key)


__all__ = ["output_dir", "cache_dir"]
