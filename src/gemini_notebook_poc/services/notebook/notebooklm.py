"""NotebookLM web implementation of INotebookService using notebooklm-py."""

from __future__ import annotations

import logging
from typing import Any

from gemini_notebook_poc.exceptions import NotebookAuthError, NotebookNotFoundError
from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.source_info import SourceInfo

logger = logging.getLogger("gemini_notebook_poc.services.notebook.notebooklm")


class NotebookLMService:
    """Connects to NotebookLM (web/consumer) using notebooklm-py."""

    def __init__(
        self,
        auth_token: str = "",
        profile: str = "default",
        client_factory: Any | None = None,
    ) -> None:
        self.auth_token = auth_token
        self.profile = profile
        self._client_factory = client_factory
        self._titles_cache: dict[str, str] = {}

    async def _get_client(self) -> Any:
        if self._client_factory is not None:
            return await self._client_factory()
        try:
            from notebooklm import NotebookLMClient

            return NotebookLMClient.from_storage(profile=self.profile)
        except Exception as exc:
            msg = (
                f"Failed to initialize NotebookLM client: {exc}\n"
                "Please authenticate by running: notebooklm login --browser chrome"
            )
            logger.error("NotebookLM client initialization failed: %s", exc)
            raise NotebookAuthError(msg) from exc

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            res = await client.notebooks.create(title=title)
            nid = getattr(res, "id", "") or getattr(res, "notebook_id", "")
            ntitle = getattr(res, "title", title)
            logger.info("Created NotebookLM notebook '%s' (ID: %s)", ntitle, nid)
            info = NotebookInfo(
                id=str(nid),
                title=str(ntitle),
                description=description,
                source_count=0,
            )
            self._titles_cache[str(nid)] = str(ntitle)
            return info

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        nbs = await self.list_notebooks()
        for nb in nbs:
            if nb.id == notebook_id:
                self._titles_cache[nb.id] = nb.title
                return nb
        raise NotebookNotFoundError(f"NotebookLM notebook '{notebook_id}' not found.")

    async def list_notebooks(self) -> list[NotebookInfo]:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            raw_nbs = await client.notebooks.list()
            out: list[NotebookInfo] = []
            for n in raw_nbs:
                nid = str(getattr(n, "id", "") or getattr(n, "notebook_id", ""))
                ntitle = str(getattr(n, "title", "Untitled Notebook"))
                ndesc = getattr(n, "description", "")
                self._titles_cache[nid] = ntitle
                out.append(
                    NotebookInfo(
                        id=nid,
                        title=ntitle,
                        description=str(ndesc),
                    )
                )
            return out

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            if title and hasattr(client.notebooks, "rename"):
                await client.notebooks.rename(notebook_id, title)
            elif title and hasattr(client.notebooks, "update"):
                await client.notebooks.update(notebook_id, title=title)
            return await self.get_notebook(notebook_id)

    async def delete_notebook(self, notebook_id: str) -> bool:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            try:
                if hasattr(client.notebooks, "delete"):
                    await client.notebooks.delete(notebook_id)
                    logger.info("Deleted NotebookLM notebook %s", notebook_id)
                    return True
                return False
            except Exception as exc:
                logger.warning("Failed to delete notebook %s: %s", notebook_id, exc)
                return False

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            raw_sources = await client.sources.list(notebook_id)
            out: list[SourceInfo] = []
            for s in raw_sources:
                out.append(
                    SourceInfo(
                        id=getattr(s, "id", "") or getattr(s, "source_id", ""),
                        title=getattr(s, "title", "Untitled Source"),
                        source_type=getattr(s, "source_type", "Document"),
                        snippet=getattr(s, "snippet", "") or getattr(s, "summary", ""),
                    )
                )
            return out

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> SourceInfo:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            if hasattr(client.sources, "add_text"):
                res = await client.sources.add_text(notebook_id, title=title, content=content)
                sid = getattr(res, "id", "") or getattr(res, "source_id", "")
                return SourceInfo(id=str(sid), title=title, snippet=content[:150])
            raise NotImplementedError(
                "Adding sources via NotebookLM API is not supported in this client version."
            )

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            try:
                if hasattr(client.sources, "delete"):
                    await client.sources.delete(notebook_id, source_id)
                    logger.info("Deleted source %s from notebook %s", source_id, notebook_id)
                    return True
                return False
            except Exception as exc:
                logger.warning("Failed to delete source %s: %s", source_id, exc)
                return False

    async def query_notebook(self, notebook_id: str, question: str) -> NotebookAnswer:
        client_ctx = await self._get_client()
        async with client_ctx as client:
            title = self._titles_cache.get(notebook_id)
            if not title:
                try:
                    nb = await self.get_notebook(notebook_id)
                    title = nb.title
                    self._titles_cache[notebook_id] = title
                except Exception:
                    title = notebook_id
            try:
                llm_response = await client.chat.ask(notebook_id, question)
                answer_text = getattr(llm_response, "answer", str(llm_response))

                citations: list[str] = []
                references = getattr(llm_response, "references", []) or []
                for ref in references:
                    num = getattr(ref, "citation_number", None)
                    cited_text = getattr(ref, "cited_text", None)
                    sid = getattr(ref, "source_id", "")
                    prefix = f"[{num}] " if num else ""
                    if cited_text:
                        citations.append(f'{prefix}"{cited_text[:120]}..."')
                    elif sid:
                        citations.append(f"Source ID: {sid[:12]}...")

                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=answer_text,
                    citations=citations,
                    success=True,
                )
            except Exception as exc:
                logger.warning("Failed to query notebook %s: %s", notebook_id, exc)
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=f"Error querying notebook: {exc}",
                    citations=[],
                    success=False,
                    error_message=str(exc),
                )
