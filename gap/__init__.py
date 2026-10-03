"""ASTRA: analytic repair of code language models, and its evaluation."""

from .utils.cache import configure_workspace_environment

# Configure portable caches before heavyweight libraries import.
configure_workspace_environment()

__version__ = "1.0.0"

__all__ = ["__version__"]
