"""NotebookAnswer data model."""

from __future__ import annotations

from dataclasses import dataclass, field


#from gemini_notebook_poc.orchestrator import NotebookAnswer


@dataclass
class NotebookAnswer:
    """Represents a grounded answer retrieved from an individual notebook."""

    notebook_id: str
    notebook_title: str
    answer: str
    citations: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


__all__ = ["NotebookAnswer"]
