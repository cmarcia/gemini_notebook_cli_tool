"""Direct NotebookLM web/consumer backend using notebooklm-py."""

from __future__ import annotations

from typing import Any

from gemini_notebook_poc.backends.base import BaseNotebookBackend
from gemini_notebook_poc.application_configuration import ApplicationConfiguration
from gemini_notebook_poc.model import (
    GroundedAnswer,
    NotebookInfo,
    NotebookQueryAnswer,
    SourceInfo,
)
from gemini_notebook_poc.orchestrator import NotebookOrchestrator
from gemini_notebook_poc.services.notebook.notebooklm import NotebookLMService


class NotebookLMBackend(BaseNotebookBackend):
    """Backend connecting to NotebookLM (web/enterprise) via notebooklm-py."""

    def __init__(self, config: ApplicationConfiguration):
        self.config = config
        self.service = NotebookLMService()

    async def _get_client(self):
        try:
            from notebooklm import NotebookLMClient

            # Attempts to load authentication tokens from ~/.notebooklm/profiles/default/storage_state.json
            return NotebookLMClient.from_storage()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize NotebookLM client: {e}\n\n"
                "Please run `uv run notebooklm login` once to authenticate your session via browser, "
                "or switch to BACKEND_MODE=enterprise in your .env file."
            ) from e

    async def list_notebooks(self) -> list[NotebookInfo]:
        """List all notebooks via NotebookLMClient."""
        client_cm = await self._get_client()
        async with client_cm as client:
            notebooks_data = await client.notebooks.list()
            result: list[NotebookInfo] = []
            for n in notebooks_data:
                nid = getattr(n, "id", "") or getattr(n, "notebook_id", "")
                title = getattr(n, "title", "Untitled Notebook")
                sources = getattr(n, "sources", []) or []
                updated_at = str(getattr(n, "updated_at", "") or getattr(n, "modified_time", ""))
                created_at = str(getattr(n, "created_at", ""))
                result.append(
                    NotebookInfo(
                        id=str(nid),
                        title=title,
                        source_count=len(sources),
                        updated_at=updated_at,
                        created_at=created_at,
                        raw=n.__dict__ if hasattr(n, "__dict__") else {},
                    )
                )
            return result

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        """Create a new notebook in NotebookLM."""
        return await self.service.create_notebook(title, description)

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        """Update a notebook's title or description in NotebookLM."""
        return await self.service.update_notebook(notebook_id, title=title, description=description)

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook by ID in NotebookLM."""
        return await self.service.delete_notebook(notebook_id)

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> SourceInfo:
        """Add a source document to a notebook in NotebookLM."""
        return await self.service.add_source(
            notebook_id, title=title, content=content, source_type=source_type
        )

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        """Remove a source document from a notebook in NotebookLM."""
        return await self.service.delete_source(notebook_id, source_id)

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        """Get notebook details."""
        notebooks = await self.list_notebooks()
        for nb in notebooks:
            if nb.id == notebook_id or nb.id.startswith(notebook_id):
                return nb
        return NotebookInfo(id=notebook_id, title=f"Notebook {notebook_id}")

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        """List sources in the notebook."""
        client_cm = await self._get_client()
        async with client_cm as client:
            sources_data = await client.sources.list(notebook_id)
            result: list[SourceInfo] = []
            for s in sources_data:
                sid = getattr(s, "id", "") or getattr(s, "source_id", "")
                title = getattr(s, "title", "Untitled Source")
                stype = getattr(s, "source_type", "Document")
                snippet = getattr(s, "snippet", "") or getattr(s, "summary", "")
                result.append(
                    SourceInfo(
                        id=str(sid),
                        title=str(title),
                        source_type=str(stype),
                        snippet=str(snippet),
                        raw=s.__dict__ if hasattr(s, "__dict__") else {},
                    )
                )
            return result

    async def get_source(self, notebook_id: str, source_id: str) -> SourceInfo:
        """Retrieve a specific source."""
        sources = await self.list_sources(notebook_id)
        for s in sources:
            if s.id == source_id or s.id.startswith(source_id):
                return s
        return SourceInfo(id=source_id, title=f"Source {source_id}")

    async def ask_question(
        self,
        notebook_id: str,
        source_id: str | None,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        """Ask a question to the notebook using NotebookLM RAG."""
        client_cm = await self._get_client()
        async with client_cm as client:
            source_ids = [source_id] if source_id else None
            res = await client.chat.ask(notebook_id, question, source_ids=source_ids)
            answer_text = getattr(res, "answer", str(res))

            citations: list[str] = []
            references = getattr(res, "references", []) or []
            for ref in references:
                num = getattr(ref, "citation_number", None)
                cited_text = getattr(ref, "cited_text", None)
                sid = getattr(ref, "source_id", "")
                if cited_text:
                    prefix = f"[{num}] " if num else ""
                    citations.append(f'{prefix}"{cited_text[:120]}..."')
                elif sid:
                    citations.append(f"Source ID: {sid[:12]}...")

            if not citations and source_id:
                citations.append(f"Grounded on source: {source_id}")

            return GroundedAnswer(
                answer=answer_text,
                citations=citations,
                notebook_id=notebook_id,
                source_id=source_id,
                model_name="NotebookLM-Engine",
            )

    async def ask_notebooks(
        self,
        notebook_ids: list[str] | str,
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Query one or more notebooks in parallel and synthesize with Gemini."""
        orchestrator = NotebookOrchestrator(
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
        )
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
        """Query multiple notebooks (backward-compatible alias)."""
        return await self.ask_notebooks(
            notebook_ids=notebook_ids,
            question=question,
            notebook_titles=notebook_titles,
        )
