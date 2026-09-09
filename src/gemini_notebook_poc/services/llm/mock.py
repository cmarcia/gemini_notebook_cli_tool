"""Mock offline implementation of ILLMService."""

from __future__ import annotations

import logging
from typing import Any

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.notebook_match import NotebookMatch

logger = logging.getLogger("gemini_notebook_poc.services.llm.mock")


class MockLLMService:
    """Provides deterministic offline LLM synthesis and semantic search."""

    def __init__(self, prefix: str = "Mock-LLM-Synthesizer") -> None:
        self.prefix = prefix

    async def synthesize(self, question: str, answers: list[NotebookAnswer]) -> str:
        valid_answers = [a for a in answers if a.success and a.answer.strip()]
        if not valid_answers:
            return "No valid answers could be retrieved from the selected notebooks."

        if len(valid_answers) == 1:
            return valid_answers[0].answer

        sections = [
            f"### [Grounded in: {ans.notebook_title}]\n{ans.answer}" for ans in valid_answers
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

        def _get_nb_info(item: Any) -> tuple[str, str, str]:
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

        keywords = [w for w in clean_topic.split() if len(w) > 2]
        matches: list[NotebookMatch] = []
        for nb in catalog:
            nid, ntitle, ndesc = _get_nb_info(nb)
            searchable = f"{ntitle} {ndesc} {nid}".lower()
            matching_kw = [kw for kw in keywords if kw in searchable]
            if matching_kw:
                relevance = min(5, 2 + len(matching_kw))
                matches.append(
                    NotebookMatch(
                        notebook_id=nid,
                        notebook_title=ntitle,
                        reason=f"Matches keywords ({', '.join(matching_kw)}) in title/description",
                        relevance=relevance,
                    )
                )

        matches.sort(key=lambda m: m.relevance, reverse=True)
        return matches[:limit] if limit else matches
