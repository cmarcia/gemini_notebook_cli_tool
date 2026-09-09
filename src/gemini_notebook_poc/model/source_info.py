"""SourceInfo data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SourceInfo:
    """Represents a document or data source within a notebook."""

    id: str
    title: str
    source_type: str = "Document"  # e.g., "Google Docs", "PDF", "Text", "URL"
    snippet: str = ""
    full_content: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def display_name(self) -> str:
        """Return title or formatted fallback."""
        return self.title if self.title else f"Source ({self.id[:8]}...)"
