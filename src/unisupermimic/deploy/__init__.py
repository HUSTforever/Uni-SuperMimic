"""Offline deployment bundle export and loading."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["DeploymentBundle", "export_bundle"]


def __getattr__(name: str) -> Any:
    if name == "bundle":
        module = importlib.import_module(f"{__name__}.bundle")
        globals()["bundle"] = module
        return module
    if name in __all__:
        return getattr(__getattr__("bundle"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
