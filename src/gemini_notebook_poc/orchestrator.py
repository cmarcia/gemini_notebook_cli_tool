"""Unified Notebook Orchestrator using Dependency Injection.

This module coordinates workflows between notebook storage (CRUD) and LLM cognitive
services (synthesis, semantic discovery) via injected services:
- INotebookService: Responsible for Notebook & Source CRUD and single-notebook grounding.
- ILLMService: Responsible for cross-notebook synthesis and semantic search.

Key Features:
- Dependency Inversion & Injection: Orchestrator depends on protocols, not concrete clients.
- First-Class CRUD: Create, read, update, and delete notebooks and sources.
- Unified Query: Handles single notebook queries or multi-notebook scatter-gather workflows.
- Standalone Usability: Defaults gracefully if injected services are not provided.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Configure module-level logger
logger = logging.getLogger("gemini_notebook_poc.orchestrator")


# =====================================================================
# Custom Exceptions Hierarchy
# =====================================================================


class NotebookOrchestratorError(Exception):
    """Base exception for all errors raised by NotebookOrchestrator."""


class NotebookAuthError(NotebookOrchestratorError):
    """Raised when authentication with NotebookLM or Gemini fails."""


class NotebookNotFoundError(NotebookOrchestratorError):
    """Raised when a requested notebook cannot be located."""


class NotebookQueryError(NotebookOrchestratorError):
    """Raised when an unrecoverable error occurs while querying a notebook."""


class SynthesisError(NotebookOrchestratorError):
    """Raised when cross-notebook LLM synthesis fails fatally."""


# =====================================================================
# Data Models
# =====================================================================


@dataclass
class NotebookMatch:
    """Represents a notebook matched by semantic topic search."""

    notebook_id: str
    notebook_title: str
    reason: str = ""
    relevance: int = 5


@dataclass
class NotebookAnswer:
    """Represents a grounded answer retrieved from an individual notebook."""

    notebook_id: str
    notebook_title: str
    answer: str
    citations: list[str] = field(default_factory=list)
    success: bool = True
    error_message: str | None = None


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


# Backward-compatibility alias
MultiNotebookAnswer = NotebookQueryAnswer


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


# =====================================================================
# Unified Notebook Orchestrator (Dependency Injection)
# =====================================================================


class NotebookOrchestrator:
    """Orchestrates workflows between injected INotebookService and ILLMService."""

    def __init__(
        self,
        notebook_service: Any | None = None,
        llm_service: Any | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-3.6-flash",
        max_retries: int = 4,
        base_retry_delay: float = 1.0,
    ) -> None:
        """Initialize orchestrator with injected services or fallback defaults."""
        self._notebook_service = notebook_service
        self._llm_service = llm_service
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay

    async def _get_notebooklm_client(self) -> Any:
        """Backward-compatibility helper for tests or direct NotebookLM client access."""
        from notebooklm import NotebookLMClient

        return NotebookLMClient.from_storage()

    @property
    def notebook_service(self) -> Any:
        if self._notebook_service is None:
            from gemini_notebook_poc.config import AppConfig
            from gemini_notebook_poc.services.notebook.notebooklm import (
                NotebookLMService,
            )

            cfg = AppConfig.load()
            if cfg.backend_mode.lower() == "mock":
                from gemini_notebook_poc.services.notebook.mock import (
                    MockNotebookService,
                )

                self._notebook_service = MockNotebookService()
            elif cfg.backend_mode.lower() == "enterprise":
                from gemini_notebook_poc.services.notebook.enterprise import (
                    EnterpriseNotebookService,
                )

                self._notebook_service = EnterpriseNotebookService(cfg)
            else:
                self._notebook_service = NotebookLMService(
                    client_factory=self._get_notebooklm_client
                )
        return self._notebook_service

    @property
    def llm_service(self) -> Any:
        if self._llm_service is None:
            from gemini_notebook_poc.config import AppConfig
            from gemini_notebook_poc.services.llm.gemini import GeminiLLMService
            from gemini_notebook_poc.services.llm.mock import MockLLMService

            cfg = AppConfig.load()
            api_key = self.gemini_api_key if self.gemini_api_key is not None else cfg.gemini_api_key
            if api_key.strip():
                self._llm_service = GeminiLLMService(
                    api_key=api_key,
                    model=self.gemini_model or cfg.gemini_model,
                    max_retries=self.max_retries,
                    base_retry_delay=self.base_retry_delay,
                )
            else:
                self._llm_service = MockLLMService()
        return self._llm_service

    # -----------------------------------------------------------------
    # Notebook CRUD Operations (Delegated to INotebookService)
    # -----------------------------------------------------------------

    async def create_notebook(self, title: str, description: str = "") -> Any:
        """Create a new notebook."""
        logger.info("Creating notebook '%s'...", title)
        return await self.notebook_service.create_notebook(title, description)

    async def get_notebook(self, notebook_id: str) -> Any:
        """Retrieve a notebook by ID."""
        return await self.notebook_service.get_notebook(notebook_id)

    async def list_notebooks(self) -> list[Any]:
        """List all available notebooks."""
        return await self.notebook_service.list_notebooks()

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Update a notebook's title or description."""
        logger.info("Updating notebook %s...", notebook_id)
        return await self.notebook_service.update_notebook(
            notebook_id, title=title, description=description
        )

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook by ID."""
        logger.info("Deleting notebook %s...", notebook_id)
        return await self.notebook_service.delete_notebook(notebook_id)

    # -----------------------------------------------------------------
    # Source CRUD Operations (Delegated to INotebookService)
    # -----------------------------------------------------------------

    async def list_sources(
        self,
        notebook_ids: str | Sequence[str],
        notebook_titles: dict[str, str] | None = None,
    ) -> list[NotebookSourceList]:
        """Fetch all sources for one or multiple notebooks concurrently."""
        if isinstance(notebook_ids, str):
            ids_list = [s.strip() for s in notebook_ids.split(",") if s.strip()]
        else:
            ids_list = [str(nid).strip() for nid in notebook_ids if str(nid).strip()]

        if not ids_list:
            logger.warning("list_sources() invoked with empty notebook_ids.")
            return []

        titles_map = dict(notebook_titles) if notebook_titles else {}

        async def _fetch(nid: str) -> NotebookSourceList:
            try:
                title = titles_map.get(nid)
                if not title:
                    try:
                        nb = await self.notebook_service.get_notebook(nid)
                        title = getattr(nb, "title", nid)
                    except Exception:
                        title = nid

                raw_srcs = await self.notebook_service.list_sources(nid)
                parsed = [
                    {
                        "id": getattr(s, "id", ""),
                        "title": getattr(s, "title", "Untitled Source"),
                        "source_type": getattr(s, "source_type", "Document"),
                        "snippet": getattr(s, "snippet", ""),
                    }
                    for s in raw_srcs
                ]
                return NotebookSourceList(
                    notebook_id=nid,
                    notebook_title=title,
                    sources=parsed,
                )
            except Exception as exc:
                logger.warning("Failed to list sources for notebook %s: %s", nid, exc)
                return NotebookSourceList(
                    notebook_id=nid,
                    notebook_title=titles_map.get(nid, nid),
                    sources=[],
                    error_message=str(exc),
                )

        tasks = [_fetch(nid) for nid in ids_list]
        return list(await asyncio.gather(*tasks))

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> Any:
        """Add a source document to a notebook."""
        logger.info("Adding source '%s' to notebook %s...", title, notebook_id)
        return await self.notebook_service.add_source(
            notebook_id, title=title, content=content, source_type=source_type
        )

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        """Remove a source document from a notebook."""
        logger.info("Deleting source %s from notebook %s...", source_id, notebook_id)
        return await self.notebook_service.delete_source(notebook_id, source_id)

    # -----------------------------------------------------------------
    # Cognitive / LLM Operations (Delegated to ILLMService)
    # -----------------------------------------------------------------

    async def synthesize(self, question: str, notebook_answers: list[NotebookAnswer]) -> str:
        """Synthesize multiple grounded answers using injected LLM service."""
        return await self.llm_service.synthesize(question, notebook_answers)

    async def find_notebooks(
        self,
        topic: str,
        notebooks: list[Any] | None = None,
        limit: int | None = None,
    ) -> list[NotebookMatch]:
        """Find notebooks relevant to a topic using injected LLM service."""
        catalog = notebooks
        if catalog is None:
            catalog = await self.notebook_service.list_notebooks()
        return await self.llm_service.semantic_search(topic, catalog, limit=limit)

    async def query_single_notebook(
        self,
        client: Any,
        notebook_id: str,
        question: str,
        notebook_title: str | None = None,
    ) -> NotebookAnswer:
        """Query individual notebook (kept for backward compatibility)."""
        title = notebook_title
        if not title:
            try:
                nb = await self.notebook_service.get_notebook(notebook_id)
                title = getattr(nb, "title", notebook_id)
            except Exception:
                title = notebook_id
        if client is not None and hasattr(client, "chat"):
            try:
                res = await client.chat.ask(notebook_id, question)
                answer_text = getattr(res, "answer", str(res))
                citations: list[str] = []
                for ref in getattr(res, "references", []) or []:
                    cited_text = getattr(ref, "cited_text", None)
                    if cited_text:
                        citations.append(f'"{cited_text[:120]}..."')
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=answer_text,
                    citations=citations,
                    success=True,
                )
            except Exception as exc:
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=f"Error querying notebook: {exc}",
                    citations=[],
                    success=False,
                    error_message=str(exc),
                )

        # Fallback to injected notebook_service
        return await self.notebook_service.query_notebook(notebook_id, question)

    async def query(
        self,
        notebook_ids: list[str] | str,
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Unified query: query one or multiple notebooks in parallel and synthesize via LLM."""
        if isinstance(notebook_ids, str):
            ids_list = [s.strip() for s in notebook_ids.split(",") if s.strip()]
        else:
            ids_list = [str(nid).strip() for nid in notebook_ids if str(nid).strip()]

        if not ids_list:
            logger.warning("query() invoked with empty notebook_ids.")
            return NotebookQueryAnswer(
                synthesized_answer="No notebooks were provided to query.",
                notebook_answers=[],
            )

        logger.info("Querying %d notebook(s) for: '%.50s...'", len(ids_list), question)
        titles_map = dict(notebook_titles) if notebook_titles else {}

        # Query all notebooks concurrently via notebook_service
        async def _ask(nid: str) -> NotebookAnswer:
            start = time.monotonic()
            try:
                ans = await self.notebook_service.query_notebook(nid, question)
                if nid in titles_map:
                    ans.notebook_title = titles_map[nid]
                elif (
                    not ans.notebook_title
                    or ans.notebook_title.startswith("Notebook (")
                    or ans.notebook_title == nid
                ):
                    try:
                        nb = await self.notebook_service.get_notebook(nid)
                        if nb and nb.title:
                            ans.notebook_title = nb.title
                    except Exception:
                        pass
                logger.info("Queried notebook %s in %.2fs", nid, time.monotonic() - start)
                return ans
            except Exception as exc:
                logger.warning("Query failed for notebook %s: %s", nid, exc)
                title = titles_map.get(nid)
                if not title:
                    try:
                        nb = await self.notebook_service.get_notebook(nid)
                        title = getattr(nb, "title", nid)
                    except Exception:
                        title = nid
                return NotebookAnswer(
                    notebook_id=nid,
                    notebook_title=title,
                    answer=f"Error querying notebook: {exc}",
                    success=False,
                    error_message=str(exc),
                )

        tasks = [_ask(nid) for nid in ids_list]
        answers = list(await asyncio.gather(*tasks))

        # Synthesize via llm_service
        synthesized_text = await self.llm_service.synthesize(question, answers)

        all_citations: list[str] = []
        for a in answers:
            for c in a.citations:
                all_citations.append(f"[{a.notebook_title}] {c}")

        return NotebookQueryAnswer(
            synthesized_answer=synthesized_text,
            notebook_answers=answers,
            model_name=f"{getattr(self.llm_service, 'model', 'LLM')} + NotebookService",
            all_citations=all_citations,
        )

    # Alias for backward compatibility
    query_notebooks = query


# Backward-compatibility alias
MultiNotebookOrchestrator = NotebookOrchestrator


# =====================================================================
# Standalone CLI Entrypoint
# =====================================================================


async def _standalone_main() -> None:
    """CLI entrypoint for testing orchestrator, CRUD, and synthesis directly."""
    parser = argparse.ArgumentParser(
        description="Unified Notebook Orchestrator (Storage CRUD & LLM Cognitive Services)",
    )
    # CRUD Options
    parser.add_argument(
        "--create-notebook",
        "-c",
        metavar="TITLE",
        help="Create a new notebook with the specified title",
    )
    parser.add_argument(
        "--delete-notebook",
        "-d",
        metavar="ID",
        help="Delete a notebook by ID",
    )
    parser.add_argument(
        "--update-notebook",
        metavar="ID",
        help="Notebook ID to update (requires --title or --description)",
    )
    parser.add_argument(
        "--title",
        help="New title when updating a notebook",
    )
    parser.add_argument(
        "--description",
        help="Description for creating or updating a notebook",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available notebooks",
    )

    # Query & Search Options
    parser.add_argument(
        "--notebooks",
        "-n",
        help="Comma-separated list or single Notebook ID",
    )
    parser.add_argument(
        "--find",
        "-f",
        help="Search notebooks by semantic topic using Gemini",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=3,
        help="Max notebooks to query when using --find with -q (default: 3)",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Question to query across notebooks",
    )
    parser.add_argument(
        "--sources",
        "-s",
        action="store_true",
        help="List all sources for the specified notebooks",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from gemini_notebook_poc.container import create_orchestrator

    orchestrator = create_orchestrator()

    # 1. Create Notebook
    if args.create_notebook:
        nb = await orchestrator.create_notebook(
            title=args.create_notebook,
            description=args.description or "",
        )
        print(f"Created notebook successfully:\n  Name: {nb.title}\n  ID:   {nb.id}")
        return

    # 2. Delete Notebook
    if args.delete_notebook:
        success = await orchestrator.delete_notebook(args.delete_notebook)
        if success:
            print(f"Deleted notebook {args.delete_notebook} successfully.")
        else:
            print(f"Failed to delete notebook {args.delete_notebook} (not found or error).")
        return

    # 3. Update Notebook
    if args.update_notebook:
        if not args.title and not args.description:
            print("Error: --update-notebook requires --title and/or --description.")
            sys.exit(1)
        nb = await orchestrator.update_notebook(
            notebook_id=args.update_notebook,
            title=args.title,
            description=args.description,
        )
        print(f"Updated notebook {nb.id}:\n  Title: {nb.title}\n  Description: {nb.description}")
        return

    # 4. List Notebooks
    if args.list:
        nbs = await orchestrator.list_notebooks()
        print(f"\nFound {len(nbs)} notebook(s):")
        print("-" * 75)
        for idx, n in enumerate(nbs, 1):
            print(f"{idx:2d}. Notebook Name: {n.title}")
            print(f"    Notebook ID:   {n.id}")
            if n.description:
                print(f"    Description:   {n.description}")
        print("-" * 75)
        return

    # 5. Semantic Find
    if args.find:
        print(f"Searching notebooks for topic: '{args.find}'...")
        matches = await orchestrator.find_notebooks(args.find)
        print(f"\nFound {len(matches)} matching notebook(s):")
        print("-" * 75)
        for idx, m in enumerate(matches, 1):
            score = f"{m.relevance}/5"
            print(f"{idx:2d}. Notebook Name: {m.notebook_title} [{score}]")
            print(f"    Notebook ID:   {m.notebook_id}")
            print(f"    Why:           {m.reason}")
        print("-" * 75)

        if args.question:
            top_matches = matches[: args.limit]
            target_ids = [m.notebook_id for m in top_matches]
            print(f"\nQuerying top {len(target_ids)} matched notebooks with question...")
            result = await orchestrator.query(target_ids, args.question)
            print("\n" + "=" * 70)
            print("SYNTHESIZED ANSWER:")
            print("=" * 70)
            print(result.synthesized_answer)
        return

    nb_ids = [s.strip() for s in (args.notebooks or "").split(",") if s.strip()]
    if not nb_ids:
        print(
            "Please provide --create-notebook, --delete-notebook, --list, --notebooks (-n), or --find (-f)."
        )
        sys.exit(1)

    # 6. List Sources
    if args.sources:
        print(f"Fetching sources for {len(nb_ids)} notebook(s)...")
        results = await orchestrator.list_sources(nb_ids)
        for nb_res in results:
            print("\n" + "=" * 70)
            print(
                f"SOURCES IN: {nb_res.notebook_title} (Notebook ID: {nb_res.notebook_id}) - {len(nb_res.sources)} sources"
            )
            print("=" * 70)
            if nb_res.error_message:
                print(f"Error: {nb_res.error_message}")
                continue
            for idx, s in enumerate(nb_res.sources, 1):
                print(
                    f"  {idx}. [Notebook: {nb_res.notebook_title}] {s['title']} [{s.get('source_type', 'Document')}] (Source ID: {s['id']})"
                )
        if not args.question:
            return

    # 7. Unified Query
    if args.question:
        desc = f"{len(nb_ids)} notebook(s)" if len(nb_ids) > 1 else f"notebook {nb_ids[0]}"
        print(f"\nQuerying {desc}...")
        result = await orchestrator.query(nb_ids, args.question)

        print("\n" + "=" * 70)
        print("SYNTHESIZED ANSWER:" if result.is_multi_notebook else "GROUNDED ANSWER:")
        print("=" * 70)
        print(result.synthesized_answer)

        if result.is_multi_notebook:
            print("\n" + "-" * 70)
            print("PER-NOTEBOOK GROUNDING BREAKDOWN:")
            print("-" * 70)
            for nb in result.notebook_answers:
                status = "[Success]" if nb.success else "[Failed]"
                print(f"\n[Notebook: {nb.notebook_title}] (ID: {nb.notebook_id}) {status}")
                if nb.success:
                    print(f"Answer snippet: {nb.answer[:200]}...")
                    if nb.citations:
                        print(f"Citations: {len(nb.citations)} reference(s)")


if __name__ == "__main__":
    asyncio.run(_standalone_main())
