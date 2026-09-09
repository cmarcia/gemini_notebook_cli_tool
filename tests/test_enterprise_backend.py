"""Unit tests for EnterpriseBackend URL resolution and error handling."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gemini_notebook_poc.backends.enterprise import (
    EnterpriseAPIError,
    EnterpriseBackend,
)
from gemini_notebook_poc.config import AppConfig


def test_enterprise_backend_url():
    config = AppConfig(
        backend_mode="enterprise",
        gcp_project_id="my-enterprise-project",
        gcp_location="us",
    )
    backend = EnterpriseBackend(config)
    assert (
        backend.base_url
        == "https://discoveryengine.googleapis.com/v1alpha/projects/my-enterprise-project/locations/us"
    )


@pytest.mark.asyncio
async def test_enterprise_backend_disabled_api_error():
    config = AppConfig(
        backend_mode="enterprise",
        gcp_project_id="taxes-507506",
        gcp_access_token="fake-token",
    )
    backend = EnterpriseBackend(config)

    mock_resp = httpx.Response(
        status_code=403,
        json={
            "error": {
                "code": 403,
                "message": "Discovery Engine API has not been used in project taxes-507506 before or it is disabled.",
                "status": "PERMISSION_DENIED",
                "details": [{"reason": "SERVICE_DISABLED"}],
            }
        },
        request=httpx.Request("GET", "https://discoveryengine.googleapis.com"),
    )

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        with pytest.raises(EnterpriseAPIError) as excinfo:
            await backend.list_notebooks()
        assert "Discovery Engine API is not enabled" in str(excinfo.value)
        assert "gcloud services enable discoveryengine.googleapis.com" in str(excinfo.value)
