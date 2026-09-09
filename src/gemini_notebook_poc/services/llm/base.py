"""Protocol defining the LLM cognitive and synthesis operations interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.notebook_match import NotebookMatch


@runtime_checkable
class ILLMService(Protocol):
    """Protocol for LLM reasoning, synthesis, and semantic retrieval."""

    async def synthesize(self, question: str, answers: list[NotebookAnswer]) -> str:
        """Synthesize multiple grounded answers into a unified response with attributions."""
        ...

    async def semantic_search(
        self, topic: str, catalog: list[NotebookInfo], limit: int | None = None
    ) -> list[NotebookMatch]:
        """Perform semantic topic search across a notebook catalog."""
        ...
