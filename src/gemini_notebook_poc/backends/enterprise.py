"""Google Cloud Gemini Enterprise / Discovery Engine API backend."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from google import genai

from gemini_notebook_poc.backends.base import BaseNotebookBackend
from gemini_notebook_poc.application_configuration import ApplicationConfiguration
from gemini_notebook_poc.model import (
    GroundedAnswer,
    NotebookInfo,
    NotebookQueryAnswer,
    SourceInfo,
)
from gemini_notebook_poc.orchestrator import (
    NotebookAnswer,
    NotebookOrchestrator,
)
from gemini_notebook_poc.services.notebook.enterprise import EnterpriseNotebookService


class EnterpriseAPIError(RuntimeError):
    """Raised when the Google Cloud Discovery Engine API encounters an error."""


class EnterpriseBackend(BaseNotebookBackend):
    """Backend utilizing Google Cloud Gemini Enterprise (Discovery Engine API)."""

    def __init__(self, config: ApplicationConfiguration):
        self.config = config
        self.project_id = config.gcp_project_id or "default"
        self.location = config.gcp_location or "global"
        self.base_url = (
            f"https://discoveryengine.googleapis.com/v1alpha/projects/"
            f"{self.project_id}/locations/{self.location}"
        )
        self.service = EnterpriseNotebookService(config)

    def _get_headers(self) -> dict[str, str]:
        token = self.config.get_bearer_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if self.project_id and self.project_id != "default":
            headers["X-Goog-User-Project"] = self.project_id
        return headers

    async def _request(
        self,
        method: str,
        url: str,
        *,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue an authenticated HTTP request to Discovery Engine endpoints."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            request_fn = getattr(client, method.lower(), None)
            if callable(request_fn):
                return await request_fn(url, headers=self._get_headers(), **kwargs)
            return await client.request(method, url, headers=self._get_headers(), **kwargs)

    def _handle_http_error(self, response: httpx.Response, action_name: str) -> None:
        try:
            err_data = response.json()
            err_obj = err_data.get("error", {})
            message = err_obj.get("message", response.text)
            status = err_obj.get("status", "")
        except Exception:
            message = response.text
            status = str(response.status_code)

        if "SERVICE_DISABLED" in str(err_data) or "has not been used in project" in message:
            enable_url = f"https://console.developers.google.com/apis/api/discoveryengine.googleapis.com/overview?project={self.project_id}"
            raise EnterpriseAPIError(
                f"Discovery Engine API is not enabled in project '{self.project_id}'.\n\n"
                f"To enable it, run:\n"
                f"  gcloud services enable discoveryengine.googleapis.com --project={self.project_id}\n\n"
                f"Or visit:\n  {enable_url}\n\n"
                f"Note: Gemini Notebook Enterprise also requires a Gemini Enterprise or Education license."
            )

        if response.status_code == 403:
            raise EnterpriseAPIError(
                f"Permission denied while attempting to {action_name} on project '{self.project_id}'.\n"
                f"Details: {message}\n"
                f"Ensure your GCP identity has the 'Discovery Engine Viewer' or 'Cloud NotebookLM User' role."
            )

        if (
            response.status_code in (404, 503)
            or "Method not found" in message
            or "UNAVAILABLE" in status
        ):
            raise EnterpriseAPIError(
                f"Gemini Notebook Enterprise endpoint is not active on project '{self.project_id}'.\n"
                f"Details: {message}\n\n"
                f"Google Cloud Gemini Notebooks require a project tied to an active Gemini Enterprise "
                f"or Gemini Education Premium license. You can switch to MOCK mode to test the full "
                f"experience with your real Gemini API key and sample notebooks."
            )

        raise EnterpriseAPIError(
            f"Google Cloud API error during {action_name} ({response.status_code} {status}): {message}"
        )

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        return await self.service.create_notebook(title, description)

    async def update_notebook(
        self, notebook_id: str, title: str | None = None, description: str | None = None
    ) -> NotebookInfo:
        return await self.service.update_notebook(notebook_id, title, description)

    async def delete_notebook(self, notebook_id: str) -> bool:
        return await self.service.delete_notebook(notebook_id)

    async def add_source(
        self, notebook_id: str, title: str, content: str, source_type: str = "Document"
    ) -> SourceInfo:
        return await self.service.add_source(notebook_id, title, content, source_type)

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        return await self.service.delete_source(notebook_id, source_id)

    async def list_notebooks(self) -> list[NotebookInfo]:
        """List notebooks using projects.locations.notebooks:listRecentlyViewed."""
        url = f"{self.base_url}/notebooks:listRecentlyViewed"

        response = await self._request("GET", url)
        if response.status_code != 200:
            self._handle_http_error(response, "list notebooks")

        data = response.json()
        items = data.get("notebooks", [])
        notebooks: list[NotebookInfo] = []
        for item in items:
            # Format: projects/{project}/locations/{loc}/notebooks/{id}
            full_name = item.get("name", "")
            notebook_id = full_name.split("/")[-1] if full_name else "unknown"
            title = (
                item.get("title") or item.get("displayName") or f"Notebook {notebook_id[:8]}"
            )
            created_at = item.get("createTime", "")
            updated_at = item.get("updateTime", "")
            sources = item.get("sources", [])
            notebooks.append(
                NotebookInfo(
                    id=notebook_id,
                    title=title,
                    description=item.get("description", ""),
                    created_at=created_at,
                    updated_at=updated_at,
                    source_count=len(sources),
                    raw=item,
                )
            )
        return notebooks

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        """Fetch details of a single notebook."""
        url = f"{self.base_url}/notebooks/{notebook_id}"

        response = await self._request("GET", url)
        if response.status_code != 200:
            self._handle_http_error(response, f"get notebook '{notebook_id}'")

        item = response.json()
        full_name = item.get("name", "")
        nid = full_name.split("/")[-1] if full_name else notebook_id
        title = item.get("title") or item.get("displayName") or f"Notebook {nid[:8]}"
        sources = item.get("sources", [])
        return NotebookInfo(
            id=nid,
            title=title,
            description=item.get("description", ""),
            created_at=item.get("createTime", ""),
            updated_at=item.get("updateTime", ""),
            source_count=len(sources),
            raw=item,
        )

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        """Fetch all sources associated with a notebook."""
        # 1. First attempt to fetch the notebook and inspect its embedded sources
        notebook = await self.get_notebook(notebook_id)
        raw_sources = notebook.raw.get("sources", [])

        # 2. Also try querying the sub-resource endpoint
        sources_url = f"{self.base_url}/notebooks/{notebook_id}/sources"

        response = await self._request("GET", sources_url)
        if response.status_code == 200:
            data = response.json()
            if data.get("sources"):
                raw_sources = data["sources"]

        result: list[SourceInfo] = []
        for idx, s in enumerate(raw_sources):
            if isinstance(s, str):
                sid = s.split("/")[-1]
                result.append(
                    SourceInfo(
                        id=sid, title=f"Source {sid[:8]}", source_type="Resource", raw={"name": s}
                    )
                )
                continue

            full_name = s.get("name", "")
            sid = full_name.split("/")[-1] if full_name else f"source-{idx + 1}"
            title = s.get("title") or s.get("displayName") or f"Source {idx + 1}"

            # Detect content type
            stype = "Document"
            preview = ""
            if "googleDriveContent" in s:
                stype = "Google Drive"
            elif "userContent" in s:
                stype = "User Upload"
                preview = str(s["userContent"])[:200]
            elif "textContent" in s:
                stype = "Text"
                preview = s["textContent"].get("content", "")[:200]
            elif "webContent" in s:
                stype = "Web URL"
                preview = s["webContent"].get("url", "")

            result.append(
                SourceInfo(
                    id=sid,
                    title=title,
                    source_type=stype,
                    snippet=preview,
                    raw=s,
                )
            )

        return result

    async def get_source(self, notebook_id: str, source_id: str) -> SourceInfo:
        """Fetch details of a single source."""
        url = f"{self.base_url}/notebooks/{notebook_id}/sources/{source_id}"

        response = await self._request("GET", url)
        if response.status_code != 200:
            # Fallback: search among list_sources
            sources = await self.list_sources(notebook_id)
            for s in sources:
                if s.id == source_id:
                    return s
            self._handle_http_error(response, f"get source '{source_id}'")

        s = response.json()
        full_name = s.get("name", "")
        sid = full_name.split("/")[-1] if full_name else source_id
        title = s.get("title") or s.get("displayName") or f"Source {sid[:8]}"
        return SourceInfo(id=sid, title=title, raw=s)

    async def ask_question(
        self,
        notebook_id: str,
        source_id: str | None,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        """Ask a question grounded on a specific source using Gemini."""
        # Retrieve target source info if specified
        target_source: SourceInfo | None = None
        source_titles: list[str] = []
        if source_id:
            try:
                target_source = await self.get_source(notebook_id, source_id)
                source_titles.append(target_source.title)
            except Exception:
                source_titles.append(f"Source {source_id}")

        # Grounding content preparation
        grounding_context = ""
        if target_source:
            grounding_context = (
                f"Source Title: {target_source.title}\n"
                f"Source Type: {target_source.source_type}\n"
                f"Source Snippet/Content: {target_source.snippet or target_source.full_content or 'Metadata loaded'}\n"
            )

        # Use Gemini API to answer with source grounding
        gemini_key = self.config.gemini_api_key.strip()
        if not gemini_key:
            return GroundedAnswer(
                answer=(
                    f"Connected to Enterprise Notebook '{notebook_id}' (Source: {target_source.title if target_source else 'All'}).\n\n"
                    "Notice: To generate live AI answers with citation grounding, please set GEMINI_API_KEY in your .env file."
                ),
                citations=[f"Source: {t}" for t in source_titles],
                source_titles=source_titles,
                notebook_id=notebook_id,
                source_id=source_id,
                model_name=self.config.gemini_model,
            )

        client = genai.Client(api_key=gemini_key)  # type: ignore[attr-defined]
        system_instruction = (
            "You are the Gemini Notebook Enterprise assistant, an advanced source-grounded RAG reasoning model. "
            "Your task is to answer the user's questions strictly based on the provided notebook and source context. "
            "Whenever you provide factual claims or summarize points, explicitly cite your source using [Source: <title>]. "
            "If the provided context does not contain enough information to answer definitively, state that clearly."
        )

        prompt_parts = [
            "=== NOTEBOOK CONTEXT ===",
            f"Notebook ID: {notebook_id}",
            f"Active Source: {target_source.title if target_source else 'All notebook sources'}",
            grounding_context,
            "=== USER QUESTION ===",
            question,
        ]
        full_prompt = "\n\n".join(prompt_parts)

        response = await asyncio.to_thread(
            client.models.generate_content,
            model=self.config.gemini_model,
            contents=full_prompt,
            config={
                "system_instruction": system_instruction,
                "automatic_function_calling": {"disable": True},
            },
        )
        answer_text = response.text or "No response received."

        # Extract citations
        citations: list[str] = []
        if target_source:
            citations.append(f"Source: {target_source.title} ({target_source.source_type})")
        else:
            citations.extend([f"Source: {t}" for t in source_titles])

        return GroundedAnswer(
            answer=answer_text,
            citations=citations,
            source_titles=source_titles,
            notebook_id=notebook_id,
            source_id=source_id,
            model_name=self.config.gemini_model,
        )

    async def ask_notebooks(
        self,
        notebook_ids: list[str] | str,
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Query one or more Enterprise notebooks and synthesize results."""
        if isinstance(notebook_ids, str):
            ids_list = [s.strip() for s in notebook_ids.split(",") if s.strip()]
        else:
            ids_list = [str(nid).strip() for nid in notebook_ids if str(nid).strip()]

        titles_map = dict(notebook_titles) if notebook_titles else {}

        async def _query_single_enterprise_nb(nid: str) -> NotebookAnswer:
            try:
                title = titles_map.get(nid)
                if not title:
                    nb = await self.get_notebook(nid)
                    title = nb.title
                ans = await self.ask_question(nid, None, question)
                return NotebookAnswer(
                    notebook_id=nid,
                    notebook_title=title,
                    answer=ans.answer,
                    citations=ans.citations,
                    success=True,
                )
            except Exception as e:
                title = titles_map.get(nid, nid)
                return NotebookAnswer(
                    notebook_id=nid,
                    notebook_title=title,
                    answer=f"Error querying enterprise notebook: {e}",
                    success=False,
                    error_message=str(e),
                )

        tasks = [_query_single_enterprise_nb(nid) for nid in ids_list]
        answers = await asyncio.gather(*tasks)

        orchestrator = NotebookOrchestrator(
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
        )
        synthesized = await orchestrator.synthesize(question, list(answers))
        all_citations: list[str] = []
        for a in answers:
            for c in a.citations:
                all_citations.append(f"[{a.notebook_title}] {c}")

        return NotebookQueryAnswer(
            synthesized_answer=synthesized,
            notebook_answers=list(answers),
            model_name=f"{self.config.gemini_model} + Enterprise",
            all_citations=all_citations,
        )

    async def ask_multi_notebooks(
        self,
        notebook_ids: list[str],
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Query multiple Enterprise notebooks (backward-compatible alias)."""
        return await self.ask_notebooks(
            notebook_ids,
            question,
            notebook_titles=notebook_titles,
        )
