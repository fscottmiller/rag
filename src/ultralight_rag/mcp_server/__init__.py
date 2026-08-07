from typing import Any

from .server import create_mcp, get_mcp

__all__ = ["create_mcp", "mcp"]


def __getattr__(name: str) -> Any:
    if name == "mcp":
        return get_mcp()
    raise AttributeError(name)
