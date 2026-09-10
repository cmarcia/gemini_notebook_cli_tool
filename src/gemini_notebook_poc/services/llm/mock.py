"""Mock offline implementation of ILLMService."""

from __future__ import annotations

import keyword
import logging
from typing import Any

from pygments.lexer import words

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.notebook_match import NotebookMatch

logger = logging.getLogger("gemini_notebook_poc.services.llm.mock")


class MockLLMService:
    """Provides deterministic offline LLM synthesis and semantic search."""

    def __init__(self, prefix: str = "Mock-LLM-Synthesizer") -> None:
        self.prefix = prefix

    async def synthesize(self, question: str, answers: list[NotebookAnswer]) -> str:
        valid_answers = [answer for answer in answers if answer.success and answer.answer.strip()]
        if not valid_answers:
            return "No valid answers could be retrieved from the selected notebooks."

        if len(valid_answers) == 1:
            return valid_answers[0].answer

        sections = [
            f"### [Grounded in: {answer.notebook_title}]\n{answer.answer}" for answer in valid_answers
        ]
        return (
            f"> [{self.prefix}]: Synthesized cross-notebook response for question: '{question}'\n\n"
            + "\n\n---\n\n".join(sections)
        )

    async def semantic_search(
        self, topic: str, catalog: list[NotebookInfo], limit: int | None = None
    ) -> list[NotebookMatch]:
        clean_topic = topic.strip().lower()
        if not clean_topic or not catalog:
            return []

        def _get_notebook_info(item: Any) -> tuple[str, str, str]:
            if isinstance(item, dict):
                return (
                    str(item.get("id", "")),
                    str(item.get("title", "Untitled")),
                    str(item.get("description", "")),
                )
            return (
                str(getattr(item, "id", "")),
                str(getattr(item, "title", "Untitled")),
                str(getattr(item, "description", "") or ""),
            )

        keywords = [word for word in clean_topic.split() if len(word) > 2]
        matches: list[NotebookMatch] = []
        for notebook in catalog:
            notebook_id, notebook_title, notebook_description = _get_notebook_info(notebook)
            searchable = f"{notebook_title} {notebook_description} {notebook_id}".lower()
            matching_key_words = [ keyword for  keyword in keywords if  keyword in searchable ]
            if matching_key_words:
                relevance = min(5, 2 + len(matching_key_words))
                matches.append(
                    NotebookMatch(
                        notebook_id=notebook_id,
                        notebook_title=notebook_title,
                        reason=f"Matches keywords ({', '.join(matching_key_words)}) in title/description",
                        relevance=relevance,
                    )
                )

        matches.sort(key=lambda match: match.relevance, reverse=True)
        return matches[:limit] if limit else matches
