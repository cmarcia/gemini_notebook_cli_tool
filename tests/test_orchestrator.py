"""Unit tests for the unified NotebookOrchestrator, exceptions, retries, and logging."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemini_notebook_poc.backends.mock import MockBackend
from gemini_notebook_poc.cli import resolve_notebook_selection
from gemini_notebook_poc.config import AppConfig
from gemini_notebook_poc.model import NotebookInfo
from gemini_notebook_poc.orchestrator import (
    MultiNotebookAnswer,
    MultiNotebookOrchestrator,
    NotebookAnswer,
    NotebookAuthError,
    NotebookNotFoundError,
    NotebookOrchestrator,
    NotebookOrchestratorError,
    NotebookQueryAnswer,
    NotebookQueryError,
    NotebookSourceList,
    SynthesisError,
)
from gemini_notebook_poc.services.notebook.mock import MockNotebookService
from gemini_notebook_poc.services.notebook.notebooklm import NotebookLMService


def test_exception_hierarchy():
    assert issubclass(NotebookAuthError, NotebookOrchestratorError)
    assert issubclass(NotebookNotFoundError, NotebookOrchestratorError)
    assert issubclass(NotebookQueryError, NotebookOrchestratorError)
    assert issubclass(SynthesisError, NotebookOrchestratorError)
    assert issubclass(NotebookOrchestratorError, Exception)


def test_backward_compatibility_aliases():
    assert NotebookOrchestrator is MultiNotebookOrchestrator
    assert NotebookQueryAnswer is MultiNotebookAnswer


def test_individual_model_module_imports():
    from gemini_notebook_poc.model.chat_turn import ChatTurn as CT
    from gemini_notebook_poc.model.grounded_answer import GroundedAnswer as GA
    from gemini_notebook_poc.model.notebook_answer import NotebookAnswer as NA
    from gemini_notebook_poc.model.notebook_info import NotebookInfo as NI
    from gemini_notebook_poc.model.notebook_match import NotebookMatch as NM
    from gemini_notebook_poc.model.notebook_query_answer import (
        MultiNotebookAnswer as MNA,
    )
    from gemini_notebook_poc.model.notebook_query_answer import (
        NotebookQueryAnswer as NQA,
    )
    from gemini_notebook_poc.model.notebook_source_list import NotebookSourceList as NSL
    from gemini_notebook_poc.model.source_info import SourceInfo as SI

    assert NI is not None
    assert SI is not None
    assert GA is not None
    assert CT is not None
    assert NA is not None
    assert NQA is not None
    assert MNA is not None
    assert NSL is not None
    assert NM is not None


def test_dataclass_properties():
    # Test NotebookQueryAnswer properties
    ans1 = NotebookAnswer(
        notebook_id="nb-1",
        notebook_title="Title 1",
        answer="Answer 1",
        success=True,
    )
    ans2 = NotebookAnswer(
        notebook_id="nb-2",
        notebook_title="Title 2",
        answer="Answer 2",
        success=False,
        error_message="Timeout",
    )

    query_ans = NotebookQueryAnswer(
        synthesized_answer="Synthesized",
        notebook_answers=[ans1, ans2],
        all_citations=["[Title 1] Ref 1"],
    )

    assert query_ans.successful_count == 1
    assert query_ans.is_multi_notebook is True

    # Single notebook answer
    single_query_ans = NotebookQueryAnswer(
        synthesized_answer="Answer 1",
        notebook_answers=[ans1],
    )
    assert single_query_ans.successful_count == 1
    assert single_query_ans.is_multi_notebook is False

    # NotebookSourceList properties
    src_list = NotebookSourceList(
        notebook_id="nb-1",
        notebook_title="Title 1",
        sources=[{"id": "s1", "title": "Doc 1"}, {"id": "s2", "title": "Doc 2"}],
    )
    assert src_list.source_count == 2


def test_resolve_notebook_selection():
    notebooks = [
        NotebookInfo(id="nb-1", title="Event-Driven Architecture"),
        NotebookInfo(id="nb-2", title="Cloudinary Media Prep"),
        NotebookInfo(id="nb-3", title="Redis Invalidation Strategies"),
    ]

    # Test single number
    res = resolve_notebook_selection("1", notebooks)
    assert len(res) == 1
    assert res[0].id == "nb-1"

    # Test multiple comma-separated numbers
    res = resolve_notebook_selection("1, 3", notebooks)
    assert len(res) == 2
    assert [n.id for n in res] == ["nb-1", "nb-3"]

    # Test title substring matching
    res = resolve_notebook_selection("Event, Redis", notebooks)
    assert len(res) == 2
    assert [n.id for n in res] == ["nb-1", "nb-3"]

    # Test deduplication
    res = resolve_notebook_selection("1, 1, Event", notebooks)
    assert len(res) == 1
    assert res[0].id == "nb-1"

    # Test list of titles with spaces (repeated -n flags)
    res = resolve_notebook_selection(
        ["Cloudinary Media Prep", "Redis Invalidation Strategies"], notebooks
    )
    assert len(res) == 2
    assert [n.id for n in res] == ["nb-2", "nb-3"]

    # Test comma-separated string of titles with spaces
    res = resolve_notebook_selection(
        "Cloudinary Media Prep, Redis Invalidation Strategies", notebooks
    )
    assert len(res) == 2
    assert [n.id for n in res] == ["nb-2", "nb-3"]


@pytest.mark.asyncio
async def test_orchestrator_query_empty():
    orchestrator = NotebookOrchestrator()
    result = await orchestrator.query([], "Any question?")
    assert result.synthesized_answer == "No notebooks were provided to query."
    assert len(result.notebook_answers) == 0


@pytest.mark.asyncio
async def test_orchestrator_query_single_str_id():
    orchestrator = NotebookOrchestrator(gemini_api_key="")

    mock_client = MagicMock()
    mock_res = MagicMock()
    mock_res.answer = "Direct answer from single notebook."
    mock_res.references = []
    mock_client.chat.ask = AsyncMock(return_value=mock_res)

    class MockAsyncContext:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    with patch.object(
        orchestrator, "_get_notebooklm_client", AsyncMock(return_value=MockAsyncContext())
    ):
        result = await orchestrator.query(
            notebook_ids="nb-single-123",
            question="What is the architecture?",
            notebook_titles={"nb-single-123": "My Architecture Notebook"},
        )

        assert isinstance(result, NotebookQueryAnswer)
        assert len(result.notebook_answers) == 1
        assert result.successful_count == 1
        assert result.is_multi_notebook is False
        assert result.synthesized_answer == "Direct answer from single notebook."
        assert result.notebook_answers[0].notebook_title == "My Architecture Notebook"


@pytest.mark.asyncio
async def test_orchestrator_query_multiple_ids():
    orchestrator = NotebookOrchestrator(gemini_api_key="")

    mock_client = MagicMock()

    async def mock_ask(nid, q):
        res = MagicMock()
        res.answer = f"Answer for {nid}"
        res.references = []
        return res

    mock_client.chat.ask = AsyncMock(side_effect=mock_ask)

    class MockAsyncContext:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    with patch.object(
        orchestrator, "_get_notebooklm_client", AsyncMock(return_value=MockAsyncContext())
    ):
        result = await orchestrator.query(
            notebook_ids=["nb-1", "nb-2"],
            question="Compare microservices and monoliths",
            notebook_titles={"nb-1": "Microservices", "nb-2": "Monoliths"},
        )

        assert isinstance(result, NotebookQueryAnswer)
        assert len(result.notebook_answers) == 2
        assert result.successful_count == 2
        assert result.is_multi_notebook is True
        assert "Microservices" in result.synthesized_answer
        assert "Monoliths" in result.synthesized_answer


@pytest.mark.asyncio
async def test_orchestrator_query_single_notebook_resilience():
    orchestrator = NotebookOrchestrator()

    mock_client = MagicMock()
    mock_ask_res = MagicMock()
    mock_ask_res.answer = "Grounded response text."
    mock_ask_res.references = []
    mock_client.chat.ask = AsyncMock(return_value=mock_ask_res)

    res = await orchestrator.query_single_notebook(
        mock_client,
        "nb-123",
        "Test question?",
        "Test Title",
    )
    assert res.success is True
    assert res.notebook_title == "Test Title"
    assert res.answer == "Grounded response text."

    mock_client.chat.ask = AsyncMock(side_effect=RuntimeError("Network timeout"))
    res_fail = await orchestrator.query_single_notebook(
        mock_client,
        "nb-123",
        "Test question?",
        "Test Title",
    )
    assert res_fail.success is False
    assert "Network timeout" in res_fail.error_message


@pytest.mark.asyncio
async def test_orchestrator_synthesize_single_answer():
    orchestrator = NotebookOrchestrator(gemini_api_key="")
    answers = [
        NotebookAnswer(
            notebook_id="nb-1",
            notebook_title="Architecture",
            answer="Use event-driven Kafka pipelines.",
            citations=["[1] Kafka design guide"],
            success=True,
        )
    ]
    synth = await orchestrator.synthesize("How to scale?", answers)
    assert synth == "Use event-driven Kafka pipelines."


@pytest.mark.asyncio
async def test_orchestrator_synthesize_fallback_without_gemini_key():
    orchestrator = NotebookOrchestrator(gemini_api_key="")
    answers = [
        NotebookAnswer(
            notebook_id="nb-1",
            notebook_title="Architecture",
            answer="Use event streams.",
            success=True,
        ),
        NotebookAnswer(
            notebook_id="nb-2",
            notebook_title="Caching",
            answer="Use Redis cache-aside.",
            success=True,
        ),
    ]
    synth = await orchestrator.synthesize("How to scale?", answers)
    assert "Architecture" in synth
    assert "Caching" in synth
    assert "Use event streams" in synth
    assert "Use Redis cache-aside" in synth


@pytest.mark.asyncio
async def test_synthesize_rate_limit_backoff_and_retry(monkeypatch):
    orchestrator = NotebookOrchestrator(
        gemini_api_key="mock-key",
        max_retries=3,
        base_retry_delay=0.01,
    )

    call_count = 0

    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")
        resp = MagicMock()
        resp.text = "Synthesized response after backoff."
        return resp

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_generate_content

    import google.genai

    monkeypatch.setattr(google.genai, "Client", lambda api_key: mock_client)

    answers = [
        NotebookAnswer(notebook_id="nb-1", notebook_title="Title 1", answer="Answer 1"),
        NotebookAnswer(notebook_id="nb-2", notebook_title="Title 2", answer="Answer 2"),
    ]

    result = await orchestrator.synthesize("Question?", answers)
    assert result == "Synthesized response after backoff."
    assert call_count == 3


@pytest.mark.asyncio
async def test_synthesize_fallback_when_retries_exhausted(monkeypatch):
    orchestrator = NotebookOrchestrator(
        gemini_api_key="mock-key",
        max_retries=2,
        base_retry_delay=0.01,
    )

    def mock_generate_content(*args, **kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = mock_generate_content

    import google.genai

    monkeypatch.setattr(google.genai, "Client", lambda api_key: mock_client)

    answers = [
        NotebookAnswer(notebook_id="nb-1", notebook_title="Title 1", answer="Answer 1"),
        NotebookAnswer(notebook_id="nb-2", notebook_title="Title 2", answer="Answer 2"),
    ]

    result = await orchestrator.synthesize("Question?", answers)
    assert "rate limit reached for synthesis" in result
    assert "Title 1" in result
    assert "Title 2" in result


@pytest.mark.asyncio
async def test_orchestrator_list_sources_single_and_multiple():
    orchestrator = NotebookOrchestrator()

    mock_client = MagicMock()

    async def mock_sources_list(nid):
        src = MagicMock()
        src.id = f"s-{nid}"
        src.title = f"Doc for {nid}"
        src.source_type = "Document"
        src.snippet = "Sample snippet"
        return [src]

    mock_client.sources.list = AsyncMock(side_effect=mock_sources_list)

    class MockAsyncContext:
        async def __aenter__(self):
            return mock_client

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    with patch.object(
        orchestrator, "_get_notebooklm_client", AsyncMock(return_value=MockAsyncContext())
    ):
        single_res = await orchestrator.list_sources("nb-100")
        assert len(single_res) == 1
        assert single_res[0].source_count == 1
        assert single_res[0].sources[0]["id"] == "s-nb-100"

        multi_res = await orchestrator.list_sources(["nb-100", "nb-200"])
        assert len(multi_res) == 2
        assert multi_res[1].sources[0]["id"] == "s-nb-200"


@pytest.mark.asyncio
async def test_find_notebooks_keyword_fallback():
    orchestrator = NotebookOrchestrator(gemini_api_key="")
    sample_notebooks = [
        {"id": "nb-1", "title": "Building Event-Driven Microservices Architecture"},
        {"id": "nb-2", "title": "Enterprise Financial Reports Q3"},
        {"id": "nb-3", "title": "Clean Code and Software Architecture Patterns"},
    ]

    matches = await orchestrator.find_notebooks("software architecture", notebooks=sample_notebooks)
    assert len(matches) == 2
    matched_ids = [m.notebook_id for m in matches]
    assert "nb-1" in matched_ids
    assert "nb-3" in matched_ids
    assert "nb-2" not in matched_ids


@pytest.mark.asyncio
async def test_find_notebooks_mock_llm(monkeypatch):
    orchestrator = NotebookOrchestrator(gemini_api_key="test-key")
    sample_notebooks = [
        {"id": "nb-1", "title": "Software Architecture in Practice"},
        {"id": "nb-2", "title": "Marketing 101"},
    ]

    fake_response = MagicMock()
    fake_response.text = (
        '[{"id": "nb-1", "title": "Software Architecture in Practice", '
        '"reason": "Focuses directly on software architecture design principles.", "relevance": 5}]'
    )

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = fake_response

    import google.genai

    monkeypatch.setattr(google.genai, "Client", lambda api_key: mock_client)

    matches = await orchestrator.find_notebooks("architecture", notebooks=sample_notebooks)
    assert len(matches) == 1
    assert matches[0].notebook_id == "nb-1"
    assert matches[0].relevance == 5
    assert "architecture design principles" in matches[0].reason


@pytest.mark.asyncio
async def test_mock_backend_ask_notebooks():
    config = AppConfig(backend_mode="mock", gemini_api_key="")
    backend = MockBackend(config)

    result = await backend.ask_notebooks(
        notebook_ids=["nb-enterprise-finance-q3", "nb-cloud-architecture-v2"],
        question="What are the key financial and cloud architecture priorities?",
    )

    assert isinstance(result, NotebookQueryAnswer)
    assert len(result.notebook_answers) == 2
    assert result.successful_count == 2
    assert any("Financial" in a.notebook_title for a in result.notebook_answers)
    assert len(result.all_citations) > 0


@pytest.mark.asyncio
async def test_mock_backend_find_notebooks():
    config = AppConfig(backend_mode="mock", gemini_api_key="")
    backend = MockBackend(config)

    matches = await backend.find_notebooks("architecture")
    assert len(matches) >= 1
    assert any(m.notebook_id == "nb-cloud-architecture-v2" for m in matches)


# =====================================================================
# Dependency Injection, Protocol, and CRUD Lifecycle Tests
# =====================================================================


def test_protocol_compliance():
    from gemini_notebook_poc.services.llm.base import ILLMService
    from gemini_notebook_poc.services.llm.gemini import GeminiLLMService
    from gemini_notebook_poc.services.llm.mock import MockLLMService
    from gemini_notebook_poc.services.notebook.base import INotebookService
    from gemini_notebook_poc.services.notebook.enterprise import (
        EnterpriseNotebookService,
    )
    from gemini_notebook_poc.services.notebook.mock import MockNotebookService

    mock_nb = MockNotebookService()
    assert isinstance(mock_nb, INotebookService)

    nlm_nb = NotebookLMService()
    assert isinstance(nlm_nb, INotebookService)

    ent_nb = EnterpriseNotebookService(AppConfig(backend_mode="enterprise"))
    assert isinstance(ent_nb, INotebookService)

    mock_llm = MockLLMService()
    assert isinstance(mock_llm, ILLMService)

    gem_llm = GeminiLLMService(api_key="fake-key")
    assert isinstance(gem_llm, ILLMService)


@pytest.mark.asyncio
async def test_orchestrator_dependency_injection():
    from gemini_notebook_poc.services.llm.mock import MockLLMService
    from gemini_notebook_poc.services.notebook.mock import MockNotebookService

    custom_nb_service = MockNotebookService()
    custom_llm_service = MockLLMService(prefix="Custom-DI-Prefix")

    orchestrator = NotebookOrchestrator(
        notebook_service=custom_nb_service,
        llm_service=custom_llm_service,
    )

    assert orchestrator.notebook_service is custom_nb_service
    assert orchestrator.llm_service is custom_llm_service

    # Test synthesized response uses the injected custom service
    result = await orchestrator.query(
        notebook_ids=["nb-enterprise-finance-q3", "nb-cloud-architecture-v2"],
        question="What is the summary?",
    )
    assert "Custom-DI-Prefix" in result.synthesized_answer


@pytest.mark.asyncio
async def test_orchestrator_crud_lifecycle():
    from gemini_notebook_poc.services.llm.mock import MockLLMService
    from gemini_notebook_poc.services.notebook.mock import MockNotebookService

    nb_service = MockNotebookService()
    llm_service = MockLLMService()
    orchestrator = NotebookOrchestrator(
        notebook_service=nb_service,
        llm_service=llm_service,
    )

    # 1. Create Notebook
    created = await orchestrator.create_notebook(
        title="Microservices Patterns",
        description="Event sourcing and CQRS patterns",
    )
    assert created.title == "Microservices Patterns"
    assert created.id.startswith("nb-")

    # 2. Get Notebook
    fetched = await orchestrator.get_notebook(created.id)
    assert fetched.title == "Microservices Patterns"
    assert fetched.description == "Event sourcing and CQRS patterns"

    # 3. Update Notebook
    updated = await orchestrator.update_notebook(
        notebook_id=created.id,
        title="Advanced Microservices Patterns",
        description="Updated description with Saga patterns",
    )
    assert updated.title == "Advanced Microservices Patterns"
    assert "Saga" in updated.description

    # 4. Add Source
    source = await orchestrator.add_source(
        notebook_id=created.id,
        title="Saga Pattern Specification.md",
        content="Saga orchestrates distributed transactions through compensating actions.",
        source_type="Markdown Document",
    )
    assert source.title == "Saga Pattern Specification.md"
    assert source.id.startswith("src-")

    # 5. List Sources
    src_lists = await orchestrator.list_sources(created.id)
    assert len(src_lists) == 1
    assert src_lists[0].source_count == 1
    assert src_lists[0].sources[0]["title"] == "Saga Pattern Specification.md"

    # 6. Delete Source
    deleted_source = await orchestrator.delete_source(created.id, source.id)
    assert deleted_source is True
    remaining_srcs = await orchestrator.list_sources(created.id)
    assert remaining_srcs[0].source_count == 0

    # 7. Delete Notebook
    deleted_nb = await orchestrator.delete_notebook(created.id)
    assert deleted_nb is True

    # 8. Verify Deletion
    all_nbs = await orchestrator.list_notebooks()
    assert not any(n.id == created.id for n in all_nbs)


def test_container_factories():
    from gemini_notebook_poc.container import (
        create_orchestrator,
        get_llm_service,
        get_notebook_service,
    )
    from gemini_notebook_poc.services.llm.gemini import GeminiLLMService
    from gemini_notebook_poc.services.llm.mock import MockLLMService
    from gemini_notebook_poc.services.notebook.enterprise import (
        EnterpriseNotebookService,
    )
    from gemini_notebook_poc.services.notebook.mock import MockNotebookService

    # Mock mode
    mock_cfg = AppConfig(backend_mode="mock", gemini_api_key="")
    nb_svc = get_notebook_service(mock_cfg)
    assert isinstance(nb_svc, MockNotebookService)
    llm_svc = get_llm_service(mock_cfg)
    assert isinstance(llm_svc, MockLLMService)

    # Enterprise mode
    ent_cfg = AppConfig(backend_mode="enterprise", gemini_api_key="real-key")
    ent_nb = get_notebook_service(ent_cfg)
    assert isinstance(ent_nb, EnterpriseNotebookService)
    gem_llm = get_llm_service(ent_cfg)
    assert isinstance(gem_llm, GeminiLLMService)

    # Orchestrator creation
    orch = create_orchestrator(mock_cfg)
    assert isinstance(orch, NotebookOrchestrator)
    assert isinstance(orch.notebook_service, MockNotebookService)
    assert isinstance(orch.llm_service, MockLLMService)


@pytest.mark.asyncio
async def test_mock_backend_crud_delegation():
    config = AppConfig(backend_mode="mock", gemini_api_key="")
    backend = MockBackend(config)

    # Create notebook via backend
    nb = await backend.create_notebook("Backend Test Notebook", "Testing delegation")
    assert nb.title == "Backend Test Notebook"

    # Add source via backend
    src = await backend.add_source(nb.id, "Test Doc", "Content of test document")
    assert src.title == "Test Doc"

    # List sources via backend
    sources = await backend.list_sources(nb.id)
    assert len(sources) == 1
    assert sources[0].id == src.id

    # Delete source via backend
    del_src = await backend.delete_source(nb.id, src.id)
    assert del_src is True

    # Delete notebook via backend
    del_nb = await backend.delete_notebook(nb.id)
    assert del_nb is True


@pytest.mark.asyncio
async def test_orchestrator_query_resolves_real_notebook_titles():
    # Setup mock notebook service with specific notebook titles
    mock_service = MockNotebookService()
    orch = NotebookOrchestrator(notebook_service=mock_service)

    # Query using only notebook ID, with no notebook_titles dictionary passed
    res = await orch.query(
        notebook_ids=["nb-enterprise-finance-q3"],
        question="What were the audit findings?",
    )
    assert len(res.notebook_answers) == 1
    # Ensure real notebook title is shown, NOT "Notebook (nb-enterprise)" or "nb-enterprise-finance-q3"
    assert (
        res.notebook_answers[0].notebook_title
        == "Enterprise Financial Insights & Risk Assessment Q3"
    )


@pytest.mark.asyncio
async def test_notebooklm_service_resolves_title_from_cache():

    client_mock = MagicMock()
    mock_ask_res = MagicMock()
    mock_ask_res.answer = "Direct answer."
    mock_ask_res.references = []
    client_mock.chat.ask = AsyncMock(return_value=mock_ask_res)

    class MockAsyncContext:
        async def __aenter__(self):
            return client_mock

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    service = NotebookLMService(client_factory=AsyncMock(return_value=MockAsyncContext()))
    service._titles_cache["nb-real-id"] = "My Genuine Architecture Notebook"

    ans = await service.query_notebook("nb-real-id", "Explain the setup")
    assert ans.notebook_title == "My Genuine Architecture Notebook"
    assert "Notebook (" not in ans.notebook_title
