"""Portable cache layout and early process defaults.

This module is imported from :mod:`gap.__init__` before Torch, Transformers,
or Datasets.  Only environment variables that must exist before those libraries
are imported are set here.  User-provided values always win.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


def workspace_root() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    project_root = package_root.parent
    return Path(os.environ.get("GAP_WORKSPACE_ROOT", project_root.parent)).expanduser().resolve()


_OMP_NUM_THREADS_RE = re.compile(r"^[1-9]\d*(?:\s*,\s*[1-9]\d*)*$")


def sanitize_openmp_environment() -> None:
    """Drop only an invalid externally supplied ``OMP_NUM_THREADS`` value.

    GNU libgomp parses this variable when Torch/NumPy native libraries load.
    Empty strings and launcher placeholders such as ``auto`` or ``$(nproc)``
    trigger a warning before the package starts.  Thread count is execution
    scheduling, not part of the scientific protocol, so an invalid external
    value is removed and libgomp falls back to its default.  Valid OpenMP
    integer/list syntax is preserved exactly.
    """
    value = os.environ.get("OMP_NUM_THREADS")
    if value is not None and not _OMP_NUM_THREADS_RE.fullmatch(value.strip()):
        os.environ.pop("OMP_NUM_THREADS", None)


def configure_workspace_environment() -> None:
    """Set current, documented process defaults before heavyweight imports."""
    sanitize_openmp_environment()
    hf = workspace_root() / "_hf"
    defaults = {
        "HF_HOME": str(hf),
        "HF_HUB_CACHE": str(hf / "hub"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


__all__ = [
    "configure_workspace_environment",
    "sanitize_openmp_environment",
    "workspace_root",
]
