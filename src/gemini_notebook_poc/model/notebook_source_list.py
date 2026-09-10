"""NotebookSourceList data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# from gemini_notebook_poc.orchestrator import NotebookSourceList


@dataclass
class NotebookSourceList:
    """Represents a notebook and its associated uploaded sources."""

    notebook_id: str
    notebook_title: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    error_message: str | None = None

    @property
    def source_count(self) -> int:
        """Total number of sources contained in this notebook."""
        return len(self.sources)



__all__ = ["NotebookSourceList"]
