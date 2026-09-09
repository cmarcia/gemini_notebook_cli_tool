"""Protocol defining the notebook storage and CRUD operations interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.source_info import SourceInfo


@runtime_checkable
class INotebookService(Protocol):
    """Protocol for notebook repository and CRUD operations."""

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        """Create a new notebook."""
        ...

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        """Retrieve a notebook by ID."""
        ...

    async def list_notebooks(self) -> list[NotebookInfo]:
        """List all available notebooks."""
        ...

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        """Update a notebook's title or description."""
        ...

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook by ID."""
        ...

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        """List all sources in a notebook."""
        ...

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> SourceInfo:
        """Add a source document to a notebook."""
        ...

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        """Remove a source document from a notebook."""
        ...

    async def query_notebook(self, notebook_id: str, question: str) -> NotebookAnswer:
        """Query a single notebook's grounded knowledge base."""
        ...
