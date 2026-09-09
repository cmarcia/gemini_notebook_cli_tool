"""NotebookInfo data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class NotebookInfo:
    """Represents a notebook container."""

    id: str
    title: str
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_count: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def display_name(self) -> str:
        """Return title or formatted fallback."""
        return self.title if self.title else f"Notebook ({self.id[:8]}...)"
