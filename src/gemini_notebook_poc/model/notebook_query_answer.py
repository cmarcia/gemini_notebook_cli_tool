"""NotebookQueryAnswer data model."""

from __future__ import annotations

from dataclasses import dataclass, field

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer


@dataclass
class NotebookQueryAnswer:
    """Represents the unified response for single or multi-notebook queries."""

    synthesized_answer: str
    notebook_answers: list[NotebookAnswer] = field(default_factory=list)
    model_name: str = "NotebookOrchestrator + LLM"
    all_citations: list[str] = field(default_factory=list)

    @property
    def successful_count(self) -> int:
        """Number of notebooks that returned successful grounded answers."""
        return sum(1 for a in self.notebook_answers if a.success)

    @property
    def is_multi_notebook(self) -> bool:
        """True if this answer represents a synthesis of multiple notebooks."""
        return len(self.notebook_answers) > 1


# Backward-compatible alias retained in the model module to avoid importing orchestrator.
MultiNotebookAnswer = NotebookQueryAnswer


__all__ = ["MultiNotebookAnswer", "NotebookQueryAnswer"]
