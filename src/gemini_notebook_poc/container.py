"""Dependency Injection Container for wiring Services and Orchestrator."""

from __future__ import annotations

import logging

from gemini_notebook_poc.config import AppConfig
from gemini_notebook_poc.services.llm import (
    GeminiLLMService,
    ILLMService,
    MockLLMService,
)
from gemini_notebook_poc.services.notebook import (
    EnterpriseNotebookService,
    INotebookService,
    MockNotebookService,
    NotebookLMService,
)

logger = logging.getLogger("gemini_notebook_poc.container")


def get_notebook_service(config: AppConfig | None = None) -> INotebookService:
    """Resolve the appropriate INotebookService implementation based on config."""
    cfg = config or AppConfig.load()
    mode = cfg.backend_mode.lower()

    if mode == "mock":
        logger.debug("Binding MockNotebookService")
        return MockNotebookService()
    elif mode == "enterprise":
        logger.debug("Binding EnterpriseNotebookService")
        return EnterpriseNotebookService(cfg)
    else:
        logger.debug("Binding NotebookLMService")
        return NotebookLMService()


def get_llm_service(config: AppConfig | None = None) -> ILLMService:
    """Resolve the appropriate ILLMService implementation based on config."""
    cfg = config or AppConfig.load()

    if cfg.gemini_api_key.strip():
        logger.debug("Binding GeminiLLMService with model=%s", cfg.gemini_model)
        return GeminiLLMService(
            api_key=cfg.gemini_api_key,
            model=cfg.gemini_model,
        )
    logger.debug("Binding MockLLMService (offline fallback)")
    return MockLLMService()


def create_orchestrator(
    config: AppConfig | None = None,
    notebook_service: INotebookService | None = None,
    llm_service: ILLMService | None = None,
):
    """Build and inject a NotebookOrchestrator instance."""
    from gemini_notebook_poc.orchestrator import NotebookOrchestrator

    cfg = config or AppConfig.load()
    nb_svc = notebook_service or get_notebook_service(cfg)
    l_svc = llm_service or get_llm_service(cfg)

    return NotebookOrchestrator(
        notebook_service=nb_svc,
        llm_service=l_svc,
    )
