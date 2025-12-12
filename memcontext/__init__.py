"""Package entry for MemContext playground."""

from importlib import import_module
from typing import Any

__all__ = ["Memcontext"]

def __getattr__(name: str) -> Any:
    """Lazily load heavy modules such as Memcontext."""
    if name == "Memcontext":
        module = import_module(".memcontext", __name__)
        return module.Memcontext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")