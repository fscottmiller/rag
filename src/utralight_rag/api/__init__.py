from .main import create_app, get_app

__all__ = ["app", "create_app"]


def __getattr__(name: str):
    if name == "app":
        return get_app()
    raise AttributeError(name)
