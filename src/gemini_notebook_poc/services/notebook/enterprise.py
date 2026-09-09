"""Google Cloud Discovery Engine implementation of INotebookService."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from gemini_notebook_poc.config import AppConfig
from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.source_info import SourceInfo
from gemini_notebook_poc.orchestrator import NotebookAuthError, NotebookNotFoundError

logger = logging.getLogger("gemini_notebook_poc.services.notebook.enterprise")


class EnterpriseNotebookService:
    """Connects to Google Cloud Discovery Engine (Gemini Enterprise DataStores)."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.project_id = config.gcp_project_id or "default"
        self.location = config.gcp_location or "global"
        self.base_url = (
            f"https://discoveryengine.googleapis.com/v1alpha/projects/"
            f"{self.project_id}/locations/{self.location}"
        )
        self._titles_cache: dict[str, str] = {}

    def _get_headers(self) -> dict[str, str]:
        token = self.config.get_bearer_token()
        if not token:
            raise NotebookAuthError(
                "No valid GCP Bearer Token found. Please authenticate via `gcloud auth application-default login` "
                "or set GCP_ACCESS_TOKEN in your environment."
            )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Goog-User-Project": self.project_id,
        }

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        url = f"{self.base_url}/collections/default_collection/dataStores"
        data_store_id = f"ds-{title.lower().replace(' ', '-')[:20]}"
        payload = {
            "displayName": title,
            "industryVertical": "GENERIC",
            "solutionTypes": ["SOLUTION_TYPE_SEARCH"],
            "contentConfig": "CONTENT_REQUIRED",
        }
        params = {"dataStoreId": data_store_id}

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, params=params, headers=self._get_headers())
            if resp.status_code in (200, 201):
                data = resp.json()
                return NotebookInfo(
                    id=data.get("name", data_store_id).split("/")[-1],
                    title=title,
                    description=description,
                )
            raise RuntimeError(
                f"Failed to create Enterprise DataStore ({resp.status_code}): {resp.text}"
            )

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        nbs = await self.list_notebooks()
        for nb in nbs:
            if nb.id == notebook_id:
                self._titles_cache[nb.id] = nb.title
                return nb
        raise NotebookNotFoundError(f"Enterprise DataStore '{notebook_id}' not found.")

    async def list_notebooks(self) -> list[NotebookInfo]:
        url = f"{self.base_url}/collections/default_collection/dataStores"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, headers=self._get_headers())
            if resp.status_code != 200:
                logger.warning("Discovery Engine list failed (%d): %s", resp.status_code, resp.text)
                return []
            data = resp.json()
            data_stores = data.get("dataStores", [])
            out: list[NotebookInfo] = []
            for ds in data_stores:
                full_name = ds.get("name", "")
                ds_id = full_name.split("/")[-1] if "/" in full_name else full_name
                title = ds.get("displayName", ds_id)
                self._titles_cache[ds_id] = title
                out.append(
                    NotebookInfo(
                        id=ds_id,
                        title=title,
                        raw=ds,
                    )
                )
            return out

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}"
        payload: dict[str, Any] = {}
        if title:
            payload["displayName"] = title
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.patch(url, json=payload, headers=self._get_headers())
            if resp.status_code == 200:
                return await self.get_notebook(notebook_id)
            raise RuntimeError(f"Failed to update DataStore: {resp.text}")

    async def delete_notebook(self, notebook_id: str) -> bool:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.delete(url, headers=self._get_headers())
            return resp.status_code in (200, 204)

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, headers=self._get_headers())
            if resp.status_code != 200:
                return []
            data = resp.json()
            docs = data.get("documents", [])
            out: list[SourceInfo] = []
            for d in docs:
                doc_id = d.get("id", "")
                title = d.get("jsonData", {}).get("title") or doc_id
                out.append(
                    SourceInfo(
                        id=doc_id,
                        title=title,
                        source_type="Document",
                        snippet=d.get("jsonData", {}).get("content", "")[:120],
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
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents"
        doc_id = f"doc-{title.lower().replace(' ', '-')[:20]}"
        payload = {
            "id": doc_id,
            "jsonData": {"title": title, "content": content},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload, headers=self._get_headers())
            if resp.status_code in (200, 201):
                return SourceInfo(id=doc_id, title=title, snippet=content[:120])
            raise RuntimeError(f"Failed to add document: {resp.text}")

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents/{source_id}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.delete(url, headers=self._get_headers())
            return resp.status_code in (200, 204)

    async def query_notebook(self, notebook_id: str, question: str) -> NotebookAnswer:
        title = self._titles_cache.get(notebook_id)
        if not title:
            try:
                nb = await self.get_notebook(notebook_id)
                title = nb.title
                self._titles_cache[notebook_id] = title
            except Exception:
                title = notebook_id

        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/servingConfigs/default_search:search"
        payload = {"query": question, "pageSize": 5}
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.post(url, json=payload, headers=self._get_headers())
                if resp.status_code != 200:
                    return NotebookAnswer(
                        notebook_id=notebook_id,
                        notebook_title=title,
                        answer=f"Discovery Engine search error ({resp.status_code}): {resp.text}",
                        success=False,
                    )
                data = resp.json()
                results = data.get("results", [])
                citations = [f"Doc: {r.get('document', {}).get('id', '')}" for r in results]
                summary = (
                    data.get("summary", {}).get("summaryText", "")
                    or f"Found {len(results)} grounded document(s)."
                )
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=summary,
                    citations=citations,
                    success=True,
                )
            except Exception as exc:
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=f"Error querying enterprise notebook: {exc}",
                    success=False,
                    error_message=str(exc),
                )
