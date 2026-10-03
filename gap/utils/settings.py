"""Shared package configuration only. Domain-specific settings live beside their code."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
OUTPUT_ROOT = Path(os.environ.get("GAP_OUTPUT_ROOT", PROJECT_ROOT / "outputs")).expanduser().resolve()
CACHE_ROOT = Path(os.environ.get("GAP_CACHE_ROOT", PROJECT_ROOT / "cache")).expanduser().resolve()
WORKSPACE_ROOT = Path(os.environ.get("GAP_WORKSPACE_ROOT", PROJECT_ROOT.parent)).expanduser().resolve()
HF_HOME = Path(os.environ.get("HF_HOME", WORKSPACE_ROOT / "_hf")).expanduser().resolve()
HF_HUB_CACHE = Path(os.environ.get("HF_HUB_CACHE", HF_HOME / "hub")).expanduser().resolve()
