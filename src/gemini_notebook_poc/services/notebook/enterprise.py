"""Google Cloud Discovery Engine implementation of INotebookService."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from gemini_notebook_poc.application_configuration import ApplicationConfiguration
from gemini_notebook_poc.exceptions import NotebookAuthError, NotebookNotFoundError
from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.source_info import SourceInfo

logger = logging.getLogger("gemini_notebook_poc.services.notebook.enterprise")


class EnterpriseNotebookService:
    """Connects to Google Cloud Discovery Engine (Gemini Enterprise DataStores)."""

    def __init__(self, application_configuration: ApplicationConfiguration) -> None:
        self.configuration = application_configuration
        self.project_id = application_configuration.gcp_project_id or "default"
        self.location = application_configuration.gcp_location or "global"
        self.base_url = (
            f"https://discoveryengine.googleapis.com/v1alpha/projects/"
            f"{self.project_id}/locations/{self.location}"
        )
        self._titles_cache: dict[str, str] = {}

    def _get_headers(self) -> dict[str, str]:
        token = self.configuration.get_bearer_token()
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

    async def _request(
        self,
        method: str,
        url: str,
        *,
        timeout: float = 20.0,
        **kwargs: Any,
    ) -> httpx.Response:
        """Issue a Discovery Engine HTTP request with shared auth headers."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.request(method, url, headers=self._get_headers(), **kwargs)

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

        response = await self._request("POST", url, timeout=30.0, json=payload, params=params)
        if response.status_code in (200, 201):
            data = response.json()
            return NotebookInfo(
                id=data.get("name", data_store_id).split("/")[-1],
                title=title,
                description=description,
            )
        raise RuntimeError(
            f"Failed to create Enterprise DataStore ({response.status_code}): {response.text}"
        )

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        notebooks = await self.list_notebooks()
        for noteboook in notebooks:
            if noteboook.id == notebook_id:
                self._titles_cache[noteboook.id] = noteboook.title
                return noteboook
        raise NotebookNotFoundError(f"Enterprise DataStore '{notebook_id}' not found.")

    async def list_notebooks(self) -> list[NotebookInfo]:
        url = f"{self.base_url}/collections/default_collection/dataStores"
        response = await self._request("GET", url)
        if response.status_code != 200:
            logger.warning("Discovery Engine list failed (%d): %s", response.status_code, response.text)
            return []
        data = response.json()
        data_stores = data.get("dataStores", [])
        out: list[NotebookInfo] = []
        for data_store in data_stores:
            full_name = data_store.get("name", "")
            data_store_id = full_name.split("/")[-1] if "/" in full_name else full_name
            display_name = data_store.get("displayName", data_store_id)
            self._titles_cache[data_store_id] = display_name
            out.append(
                NotebookInfo(
                    id=data_store_id,
                    title=display_name,
                    raw=data_store,
                )
            )
        return out

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
    ) -> NotebookInfo:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}"
        payload: dict[str, Any] = {}
        if title:
            payload["displayName"] = title
        response = await self._request("PATCH", url, json=payload)
        if response.status_code == 200:
            return await self.get_notebook(notebook_id)
        raise RuntimeError(f"Failed to update DataStore: {response.text}")

    async def delete_notebook(self, notebook_id: str) -> bool:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}"
        response = await self._request("DELETE", url)
        return response.status_code in (200, 204)

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents"
        response = await self._request("GET", url)
        if response.status_code != 200:
            return []
        data = response.json()
        documents = data.get("documents", [])
        out: list[SourceInfo] = []
        for document in documents:
            document_id = document.get("id", "")
            document_title = document.get("jsonData", {}).get("title") or document_id
            out.append(
                SourceInfo(
                    id=document_id,
                    title=document_title,
                    source_type="Document",
                    snippet=document.get("jsonData", {}).get("content", "")[:120],
                )
            )
        return out

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
    ) -> SourceInfo:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents"
        document_id = f"doc-{title.lower().replace(' ', '-')[:20]}"
        payload = {
            "id": document_id,
            "jsonData": {"title": title, "content": content},
        }
        response = await self._request("POST", url, json=payload)
        if response.status_code in (200, 201):
            return SourceInfo(id=document_id, title=title, snippet=content[:120])
        raise RuntimeError(f"Failed to add document: {response.text}")

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/branches/0/documents/{source_id}"
        response = await self._request("DELETE", url)
        return response.status_code in (200, 204)

    async def query_notebook(self, notebook_id: str, question: str) -> NotebookAnswer:
        title = self._titles_cache.get(notebook_id)
        if not title:
            try:
                notebook = await self.get_notebook(notebook_id)
                title = notebook.title
                self._titles_cache[notebook_id] = title
            except Exception:
                title = notebook_id

        url = f"{self.base_url}/collections/default_collection/dataStores/{notebook_id}/servingConfigs/default_search:search"
        payload = {"query": question, "pageSize": 5}
        try:
            response = await self._request("POST", url, timeout=30.0, json=payload)
            if response.status_code != 200:
                return NotebookAnswer(
                    notebook_id=notebook_id,
                    notebook_title=title,
                    answer=f"Discovery Engine search error ({response.status_code}): {response.text}",
                    success=False,
                )
            data = response.json()
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
        except Exception as ex:
            return NotebookAnswer(
                notebook_id=notebook_id,
                notebook_title=title,
                answer=f"Error querying enterprise notebook: {ex}",
                success=False,
                error_message=str(ex),
            )
