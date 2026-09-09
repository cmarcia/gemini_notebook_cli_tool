"""Data models package for Gemini Notebook / NotebookLM PoC."""

from __future__ import annotations

from gemini_notebook_poc.model.chat_turn import ChatTurn
from gemini_notebook_poc.model.grounded_answer import GroundedAnswer
from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.notebook_match import NotebookMatch
from gemini_notebook_poc.model.notebook_query_answer import (
    MultiNotebookAnswer,
    NotebookQueryAnswer,
)
from gemini_notebook_poc.model.notebook_source_list import NotebookSourceList
from gemini_notebook_poc.model.source_info import SourceInfo
from gemini_notebook_poc.orchestrator import (
    NotebookAuthError,
    NotebookNotFoundError,
    NotebookOrchestratorError,
    NotebookQueryError,
    SynthesisError,
)

__all__ = [
    "ChatTurn",
    "GroundedAnswer",
    "MultiNotebookAnswer",
    "NotebookAnswer",
    "NotebookAuthError",
    "NotebookInfo",
    "NotebookMatch",
    "NotebookNotFoundError",
    "NotebookOrchestratorError",
    "NotebookQueryAnswer",
    "NotebookQueryError",
    "NotebookSourceList",
    "SourceInfo",
    "SynthesisError",
]
