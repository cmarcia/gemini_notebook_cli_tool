"""ChatTurn data model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChatTurn:
    """Represents a single turn in a conversation."""

    role: str  # "user" or "model"
    content: str
    citations: list[str] = field(default_factory=list)
