"""Base abstract interface for Gemini Notebook backends."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from gemini_notebook_poc.container import create_orchestrator
from gemini_notebook_poc.model import (
    GroundedAnswer,
    NotebookInfo,
    NotebookMatch,
    NotebookQueryAnswer,
    SourceInfo,
)


class BaseNotebookBackend(ABC):
    """Abstract interface representing a notebook provider."""

    @abstractmethod
    async def list_notebooks(self) -> list[NotebookInfo]:
        """List available notebooks."""
        raise NotImplementedError

    @abstractmethod
    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        """Retrieve details of a single notebook."""
        raise NotImplementedError

    @abstractmethod
    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        """List all sources associated with a notebook."""
        raise NotImplementedError

    @abstractmethod
    async def get_source(self, notebook_id: str, source_id: str) -> SourceInfo:
        """Retrieve a specific source by ID."""
        raise NotImplementedError

    @abstractmethod
    async def ask_question(
        self,
        notebook_id: str,
        source_id: str | None,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        """Ask a question grounded on a specific source or the whole notebook."""
        raise NotImplementedError

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        """Create a new notebook."""
        raise NotImplementedError

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        """Update a notebook's title or description."""
        raise NotImplementedError

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook by ID."""
        raise NotImplementedError

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> SourceInfo:
        """Add a source document to a notebook."""
        raise NotImplementedError

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        """Remove a source document from a notebook."""
        raise NotImplementedError

    async def ask_notebooks(
        self,
        notebook_ids: list[str] | str,
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Ask a question across one or more notebooks and synthesize results."""
        orchestrator = create_orchestrator(getattr(self, "config", None))
        return await orchestrator.query(
            notebook_ids=notebook_ids,
            question=question,
            notebook_titles=notebook_titles,
        )

    async def ask_multi_notebooks(
        self,
        notebook_ids: list[str],
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Ask a question across multiple notebooks (backward-compatible alias)."""
        return await self.ask_notebooks(
            notebook_ids=notebook_ids,
            question=question,
            notebook_titles=notebook_titles,
        )

    async def list_multi_sources(
        self,
        notebook_ids: list[str],
    ) -> dict[str, list[SourceInfo]]:
        """List sources for multiple notebooks concurrently."""

        tasks = [self.list_sources(nid) for nid in notebook_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, list[SourceInfo]] = {}
        for nid, res in zip(notebook_ids, results, strict=False):
            out[nid] = res if isinstance(res, list) else []
        return out

    async def find_notebooks(
        self,
        topic: str,
        limit: int | None = None,
        notebooks: list[NotebookInfo] | None = None,
    ) -> list[NotebookMatch]:
        """Find notebooks relevant to a topic using semantic search."""
        orchestrator = create_orchestrator(getattr(self, "config", None))
        return await orchestrator.find_notebooks(topic, notebooks=notebooks, limit=limit)
