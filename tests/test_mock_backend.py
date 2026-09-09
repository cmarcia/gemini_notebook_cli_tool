"""Unit tests for MockBackend functionality."""

import pytest

from gemini_notebook_poc.backends.mock import MockBackend
from gemini_notebook_poc.config import AppConfig


@pytest.mark.asyncio
async def test_mock_backend_list_notebooks():
    config = AppConfig(backend_mode="mock")
    backend = MockBackend(config)

    notebooks = await backend.list_notebooks()
    assert len(notebooks) >= 3
    assert any("Financial" in nb.title for nb in notebooks)


@pytest.mark.asyncio
async def test_mock_backend_list_sources():
    config = AppConfig(backend_mode="mock")
    backend = MockBackend(config)

    sources = await backend.list_sources("nb-enterprise-finance-q3")
    assert len(sources) == 3
    assert any("10-Q" in s.title for s in sources)


@pytest.mark.asyncio
async def test_mock_backend_get_source():
    config = AppConfig(backend_mode="mock")
    backend = MockBackend(config)

    source = await backend.get_source("nb-enterprise-finance-q3", "src-10q-sec-filing")
    assert source.id == "src-10q-sec-filing"
    assert "10-Q" in source.title


@pytest.mark.asyncio
async def test_mock_backend_ask_question_simulated():
    config = AppConfig(backend_mode="mock", gemini_api_key="")
    backend = MockBackend(config)

    answer = await backend.ask_question(
        notebook_id="nb-enterprise-finance-q3",
        source_id="src-10q-sec-filing",
        question="What was the total revenue in Q3?",
    )
    assert answer.notebook_id == "nb-enterprise-finance-q3"
    assert answer.source_id == "src-10q-sec-filing"
    assert len(answer.citations) > 0
    assert "10-Q" in answer.citations[0]
