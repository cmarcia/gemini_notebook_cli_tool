"""Backend package and factory function."""

from __future__ import annotations

from gemini_notebook_poc.backends.base import BaseNotebookBackend
from gemini_notebook_poc.backends.enterprise import EnterpriseBackend
from gemini_notebook_poc.backends.mock import MockBackend
from gemini_notebook_poc.backends.notebooklm_backend import NotebookLMBackend
from gemini_notebook_poc.config import AppConfig


def get_backend(config: AppConfig) -> BaseNotebookBackend:
    """Instantiate the appropriate backend based on configuration."""
    mode = config.backend_mode.lower()
    if mode == "mock":
        return MockBackend(config)
    elif mode == "notebooklm":
        return NotebookLMBackend(config)
    elif mode == "enterprise":
        return EnterpriseBackend(config)
    else:
        # Default to enterprise
        return EnterpriseBackend(config)


__all__ = [
    "BaseNotebookBackend",
    "EnterpriseBackend",
    "MockBackend",
    "NotebookLMBackend",
    "get_backend",
]
