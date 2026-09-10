"""NotebookMatch data model."""

from __future__ import annotations

from dataclasses import dataclass


# from gemini_notebook_poc.orchestrator import NotebookMatch
@dataclass
class NotebookMatch:
    """Represents a notebook matched by semantic topic search."""

    notebook_id: str
    notebook_title: str
    reason: str = ""
    relevance: int = 5

__all__ = ["NotebookMatch"]
