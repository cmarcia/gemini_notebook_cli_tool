"""GroundedAnswer data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroundedAnswer:
    """Represents the response from querying a notebook or source."""

    answer: str
    citations: list[str] = field(default_factory=list)
    source_titles: list[str] = field(default_factory=list)
    notebook_id: str = ""
    source_id: str | None = None
    model_name: str = ""
