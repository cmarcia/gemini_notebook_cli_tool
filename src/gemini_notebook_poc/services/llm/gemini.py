"""Google Gemini LLM service implementation of ILLMService."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.notebook_match import NotebookMatch
from gemini_notebook_poc.orchestrator import NotebookAuthError

logger = logging.getLogger("gemini_notebook_poc.services.llm.gemini")


class GeminiLLMService:
    """Uses Google GenAI SDK (Gemini 3.6 Flash) with exponential backoff for LLM operations."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        max_retries: int = 3,
        base_retry_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key.strip()
        if not self.api_key:
            raise NotebookAuthError("GEMINI_API_KEY must be provided for GeminiLLMService.")
        self.model = model
        self.max_retries = max(1, max_retries)
        self.base_retry_delay = max(0.1, base_retry_delay)

    async def synthesize(self, question: str, answers: list[NotebookAnswer]) -> str:
        valid_answers = [a for a in answers if a.success and a.answer.strip()]
        if not valid_answers:
            return "No valid answers could be retrieved from the selected notebooks."

        if len(valid_answers) == 1:
            return valid_answers[0].answer

        blocks: list[str] = []
        for ans in valid_answers:
            citations_str = "\n  - Citations: " + "; ".join(ans.citations) if ans.citations else ""
            blocks.append(
                f'=== NOTEBOOK: "{ans.notebook_title}" (ID: {ans.notebook_id}) ===\n'
                f"{ans.answer}"
                f"{citations_str}"
            )

        context = "\n\n".join(blocks)
        system_instruction = (
            "You are an expert research and software architecture synthesizer. "
            "The user asked a question across multiple independent notebooks. "
            "Synthesize the answers from these notebooks into a clear, unified, and comprehensive response. "
            "Guidelines:\n"
            "1. Synthesize the overlapping ideas into a cohesive narrative.\n"
            "2. Explicitly attribute unique findings or specific recommendations to the originating notebook by title (e.g. '[In \"Notebook Title\"]').\n"
            "3. If different notebooks offer complementary perspectives or trade-offs, highlight them clearly.\n"
            "4. If a notebook indicates it has no relevant information, omit it or note it briefly.\n"
            "5. Conclude with key takeaways or architectural recommendations."
        )

        prompt = (
            f"User Question:\n{question}\n\n"
            f"Grounded Information from {len(valid_answers)} Notebooks:\n\n"
            f"{context}\n\n"
            "Please provide a comprehensive, well-structured synthesized answer with notebook attributions."
        )

        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                from google import genai

                logger.debug(
                    "Executing Gemini synthesis (attempt %d/%d)...", attempt, self.max_retries
                )
                client = genai.Client(api_key=self.api_key)  # type: ignore[attr-defined]
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "automatic_function_calling": {"disable": True},
                    },
                )
                logger.info("Gemini synthesis succeeded on attempt %d.", attempt)
                return resp.text or "No synthesis generated."
            except Exception as exc:
                last_exception = exc
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                if is_rate_limit and attempt < self.max_retries:
                    delay = (self.base_retry_delay * (2 ** (attempt - 1))) + random.uniform(
                        0.1, 0.5
                    )
                    logger.warning(
                        "Gemini rate limit in synthesize on attempt %d/%d. Backing off %.2fs...",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error("Gemini synthesis failed on attempt %d: %s", attempt, exc)
                break

        # Fallback to direct answers if LLM call failed
        sections = [f"### Notebook: {ans.notebook_title}\n{ans.answer}" for ans in valid_answers]
        err_str = str(last_exception) if last_exception else "Unknown error"
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            notice = "*Note: Gemini API rate limit reached for synthesis. Displaying individual answers directly from each notebook:*"
        else:
            notice = f"*Note: LLM synthesis encountered an issue ({err_str}). Displaying direct answers from each notebook:*"
        return notice + "\n\n" + "\n\n---\n\n".join(sections)

    async def semantic_search(
        self, topic: str, catalog: list[NotebookInfo], limit: int | None = None
    ) -> list[NotebookMatch]:
        clean_topic = topic.strip()
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

        catalog_lines = []
        for nb in catalog:
            nid, ntitle, ndesc = _get_nb_info(nb)
            desc_part = f" - {ndesc}" if ndesc else ""
            catalog_lines.append(f"- [{nid}] {ntitle}{desc_part}")
        catalog_text = "\n".join(catalog_lines)

        system_instruction = (
            "You are an expert knowledge retrieval assistant. "
            "Analyze the notebook catalog and identify all notebooks that are relevant to the user's search topic or theme. "
            "Output your findings strictly as a JSON array of objects with keys: "
            "'id', 'title', 'reason' (1 concise sentence explaining relevance), 'relevance' (integer 1-5, 5 being most relevant). "
            "Sort the array from highest relevance to lowest. "
            "If no notebooks match, return an empty JSON array []."
        )

        prompt = (
            f"Search Topic: '{clean_topic}'\n\n"
            f"Notebook Catalog ({len(catalog)} items):\n\n"
            f"{catalog_text}\n\n"
            "Return the JSON array of matching notebooks."
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                from google import genai

                logger.debug(
                    "Executing Gemini semantic search (attempt %d/%d)...", attempt, self.max_retries
                )
                client = genai.Client(api_key=self.api_key)  # type: ignore[attr-defined]
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "automatic_function_calling": {"disable": True},
                    },
                )

                raw_text = (resp.text or "[]").strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                data = json.loads(raw_text)
                matches: list[NotebookMatch] = []
                for item in data:
                    if isinstance(item, dict) and "id" in item:
                        matches.append(
                            NotebookMatch(
                                notebook_id=str(item.get("id", "")),
                                notebook_title=str(item.get("title", "")),
                                reason=str(item.get("reason", "")),
                                relevance=int(item.get("relevance", 3)),
                            )
                        )
                logger.info("Gemini semantic search found %d matching notebooks.", len(matches))
                return matches[:limit] if limit else matches
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                if is_rate_limit and attempt < self.max_retries:
                    delay = (self.base_retry_delay * (2 ** (attempt - 1))) + random.uniform(
                        0.1, 0.5
                    )
                    logger.warning(
                        "Gemini rate limit in semantic_search on attempt %d/%d. Backing off %.2fs...",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.warning(
                    "Gemini semantic search failed: %s; falling back to keyword search.", exc
                )
                break

        # Fallback to keyword matching
        topic_lower = clean_topic.lower()
        keywords = [w for w in topic_lower.split() if len(w) > 2]
        fallback: list[NotebookMatch] = []
        for nb in catalog:
            nid, ntitle, ndesc = _get_nb_info(nb)
            searchable = f"{ntitle} {ndesc} {nid}".lower()
            if any(kw in searchable for kw in keywords):
                fallback.append(
                    NotebookMatch(
                        notebook_id=nid,
                        notebook_title=ntitle,
                        reason="Matches topic keywords in title or description",
                        relevance=3,
                    )
                )
        return fallback[:limit] if limit else fallback
