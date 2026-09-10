from __future__ import annotations

import asyncio
from typing import Any

from google import genai

from gemini_notebook_poc.backends.base import BaseNotebookBackend
from gemini_notebook_poc.application_configuration import ApplicationConfiguration
from gemini_notebook_poc.model import (
    GroundedAnswer,
    NotebookInfo,
    NotebookQueryAnswer,
    SourceInfo,
)
from gemini_notebook_poc.orchestrator import (
    NotebookAnswer,
    NotebookOrchestrator,
)
from gemini_notebook_poc.services.notebook.mock import MockNotebookService

MOCK_NOTEBOOKS = [
    NotebookInfo(
        id="nb-enterprise-finance-q3",
        title="Enterprise Financial Insights & Risk Assessment Q3",
        description="Comprehensive quarterly financial reports, revenue breakdowns, and risk audit filings.",
        created_at="2026-07-15T10:00:00Z",
        updated_at="2026-08-30T14:20:00Z",
        source_count=3,
    ),
    NotebookInfo(
        id="nb-cloud-architecture-v2",
        title="Cloud Infrastructure & Multi-Region Migration",
        description="Specifications for migrating DocumentUploadService and databases to GCP Spanner and GKE.",
        created_at="2026-08-01T09:30:00Z",
        updated_at="2026-09-02T16:45:00Z",
        source_count=2,
    ),
    NotebookInfo(
        id="nb-ai-governance-cmek",
        title="Gemini Enterprise Governance & Security Policy",
        description="Compliance documentation on Customer-Managed Encryption Keys (CMEK) and VPC-SC.",
        created_at="2026-08-10T11:15:00Z",
        updated_at="2026-09-05T08:10:00Z",
        source_count=2,
    ),
]

MOCK_SOURCES: dict[str, list[SourceInfo]] = {
    "nb-enterprise-finance-q3": [
        SourceInfo(
            id="src-10q-sec-filing",
            title="Q3 10-Q SEC Financial Filing.pdf",
            source_type="PDF Document",
            snippet="Q3 total revenue reached $124.5M, representing a 28% YoY increase. Cloud services segment grew by 42%...",
            full_content=(
                "Q3 Total Revenue: $124.5M (+28% YoY).\n"
                "Operating Expenses: $78.2M, driven primarily by R&D investments in generative AI infrastructure.\n"
                "Net Income: $31.8M compared to $22.1M in Q3 of the prior fiscal year.\n"
                "Cash and cash equivalents: $210M as of September 30.\n"
                "Guidance for Q4 anticipates revenue between $135M and $140M."
            ),
        ),
        SourceInfo(
            id="src-risk-audit-report",
            title="Internal Risk Management Audit 2026.docx",
            source_type="Google Docs",
            snippet="Identified 3 key operational risk factors: multi-region failover latency, vendor lock-in, and compliance...",
            full_content=(
                "Key Risk Areas:\n"
                "1. Multi-region latency: Cross-region replication introduces up to 35ms latency spikes.\n"
                "2. Compliance: Requires strict data residency controls across EU and US jurisdictions.\n"
                "3. Vendor concentration: High reliance on single cloud provider APIs necessitates contingency architecture."
            ),
        ),
        SourceInfo(
            id="src-cfo-call-transcript",
            title="CFO Earnings Call Transcript.txt",
            source_type="Text Content",
            snippet="CFO remarks: 'Our margin expansion this quarter was largely driven by automated document processing pipelines...'",
            full_content=(
                "CFO: 'Our margin expansion this quarter was largely driven by automated document processing pipelines.\n"
                "We expect our gross margin to stabilize around 71% next year while maintaining accelerated hiring in engineering.'"
            ),
        ),
    ],
    "nb-cloud-architecture-v2": [
        SourceInfo(
            id="src-arch-gcp-spanner",
            title="Cloud Architecture Design v2.pdf",
            source_type="PDF Document",
            snippet="Overview of microservices architecture on GKE with Cloud Spanner for multi-region active-active persistence...",
            full_content=(
                "Architecture Architecture:\n"
                "- Microservices hosted on Google Kubernetes Engine (GKE Autopilot).\n"
                "- Database: Google Cloud Spanner multi-region instance (nam3 config).\n"
                "- Ingestion: Cloud Pub/Sub feeding Cloud Run worker nodes with auto-scaling to zero.\n"
                "- Target SLA: 99.99% availability with under 50ms p99 latency."
            ),
        ),
        SourceInfo(
            id="src-sre-runbook",
            title="SRE Incident Response Runbook.md",
            source_type="Markdown Document",
            snippet="Escalation policy: Severity 1 incidents trigger automated PagerDuty alerts to primary on-call within 5 minutes...",
            full_content=(
                "Severity 1 Response Procedure:\n"
                "1. Automated alerts dispatch PagerDuty to on-call SRE within 5 minutes.\n"
                "2. Incident commander creates bridge channel #inc-active.\n"
                "3. Rollback automation can be triggered via `sre-cli rollback --service <name>`."
            ),
        ),
    ],
    "nb-ai-governance-cmek": [
        SourceInfo(
            id="src-cmek-policy",
            title="Customer-Managed Encryption Keys Policy.pdf",
            source_type="PDF Document",
            snippet="All data at rest in Gemini Enterprise notebooks and Discovery Engine datastores must be encrypted with Cloud KMS CMEK...",
            full_content=(
                "Security Mandate:\n"
                "- Cloud KMS keys rotated every 90 days.\n"
                "- VPC Service Controls perimeter restricts API access to corporate VPN IP ranges.\n"
                "- Zero data training guarantee: Enterprise data is excluded from foundation model training."
            ),
        ),
        SourceInfo(
            id="src-vpc-sc-spec",
            title="VPC Service Controls Specification.docx",
            source_type="Google Docs",
            snippet="Perimeter definition enclosing discoveryengine.googleapis.com, aiplatform.googleapis.com, and storage.googleapis.com...",
            full_content=(
                "VPC-SC Configuration:\n"
                "- Protected services: discoveryengine.googleapis.com, aiplatform.googleapis.com, storage.googleapis.com.\n"
                "- Ingress rule allows developer access from designated corporate subnet (10.120.0.0/16)."
            ),
        ),
    ],
}


class MockBackend(BaseNotebookBackend):
    """Mock backend providing sample notebooks and sources."""

    def __init__(self, config: ApplicationConfiguration):
        self.config = config
        self._svc = MockNotebookService()

    async def list_notebooks(self) -> list[NotebookInfo]:
        return await self._svc.list_notebooks()

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        return await self._svc.get_notebook(notebook_id)

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        return await self._svc.create_notebook(title, description)

    async def update_notebook(
        self, notebook_id: str, title: str | None = None, description: str | None = None
    ) -> NotebookInfo:
        return await self._svc.update_notebook(notebook_id, title, description)

    async def delete_notebook(self, notebook_id: str) -> bool:
        return await self._svc.delete_notebook(notebook_id)

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        return await self._svc.list_sources(notebook_id)

    async def get_source(self, notebook_id: str, source_id: str) -> SourceInfo:
        srcs = await self._svc.list_sources(notebook_id)
        for s in srcs:
            if s.id == source_id:
                return s
        raise KeyError(f"Source '{source_id}' not found in notebook '{notebook_id}'.")

    async def add_source(
        self, notebook_id: str, title: str, content: str, source_type: str = "Document"
    ) -> SourceInfo:
        return await self._svc.add_source(notebook_id, title, content, source_type)

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        return await self._svc.delete_source(notebook_id, source_id)

    async def ask_question(
        self,
        notebook_id: str,
        source_id: str | None,
        question: str,
        history: list[dict[str, Any]] | None = None,
    ) -> GroundedAnswer:
        # Check if active source is selected
        target_source: SourceInfo | None = None
        if source_id:
            try:
                target_source = await self.get_source(notebook_id, source_id)
            except KeyError:
                pass

        # If user configured a live GEMINI_API_KEY, use the live Gemini model to ground on the mock source!
        gemini_key = self.config.gemini_api_key.strip()
        if gemini_key:
            try:
                client = genai.Client(api_key=gemini_key)  # type: ignore[attr-defined]
                context = (
                    f"Notebook: {notebook_id}\n"
                    f"Source: {target_source.title if target_source else 'All'}\n"
                    f"Content:\n{target_source.full_content if target_source else 'All sources in notebook'}"
                )
                system_instruction = (
                    "You are the Gemini Notebook assistant. Answer strictly based on the provided source content. "
                    "Always include explicit citations like [Source: <title>]."
                )
                resp = await asyncio.to_thread(
                    client.models.generate_content,
                    model=self.config.gemini_model,
                    contents=f"{context}\n\nQuestion: {question}",
                    config={
                        "system_instruction": system_instruction,
                        "automatic_function_calling": {"disable": True},
                    },
                )
                answer_text = resp.text or "No response generated."
                citations = (
                    [f"Source: {target_source.title}"]
                    if target_source
                    else ["Source: Mock Documents"]
                )
                return GroundedAnswer(
                    answer=answer_text,
                    citations=citations,
                    source_titles=[target_source.title] if target_source else ["Mock Documents"],
                    notebook_id=notebook_id,
                    source_id=source_id,
                    model_name=self.config.gemini_model,
                )
            except Exception:
                # Fall back to simulated response if Gemini API call errors
                pass

        # Simulated response
        src_name = target_source.title if target_source else "Selected Notebook Sources"
        answer = (
            f"Based on **{src_name}**, here are the key findings regarding your question:\n\n"
            f"• Relevant facts retrieved from the source confirm the key metrics and operational guidelines.\n"
            f'• Specifically, the document notes: *"{target_source.snippet if target_source else "Relevant operational data"}"*\n\n'
            f"[Source: {src_name}]"
        )
        return GroundedAnswer(
            answer=answer,
            citations=[f"Source: {src_name}"],
            source_titles=[src_name],
            notebook_id=notebook_id,
            source_id=source_id,
            model_name="Mock-RAG-Engine",
        )

    async def ask_notebooks(
        self,
        notebook_ids: list[str] | str,
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Query one or more mock notebooks and synthesize results."""
        if isinstance(notebook_ids, str):
            ids_list = [s.strip() for s in notebook_ids.split(",") if s.strip()]
        else:
            ids_list = [str(nid).strip() for nid in notebook_ids if str(nid).strip()]

        titles_map = dict(notebook_titles) if notebook_titles else {}

        notebook_answers: list[NotebookAnswer] = []
        for nid in ids_list:
            try:
                title = titles_map.get(nid)
                if not title:
                    nb = await self.get_notebook(nid)
                    title = nb.title
                ans = await self.ask_question(nid, None, question)
                notebook_answers.append(
                    NotebookAnswer(
                        notebook_id=nid,
                        notebook_title=title,
                        answer=ans.answer,
                        citations=ans.citations,
                        success=True,
                    )
                )
            except Exception as e:
                notebook_answers.append(
                    NotebookAnswer(
                        notebook_id=nid,
                        notebook_title=nid,
                        answer=f"Error: {e}",
                        success=False,
                        error_message=str(e),
                    )
                )

        orchestrator = NotebookOrchestrator(
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
        )
        synthesized = await orchestrator.synthesize(question, notebook_answers)
        all_citations: list[str] = []
        for a in notebook_answers:
            for c in a.citations:
                all_citations.append(f"[{a.notebook_title}] {c}")

        return NotebookQueryAnswer(
            synthesized_answer=synthesized,
            notebook_answers=notebook_answers,
            model_name=f"{self.config.gemini_model} + Mock",
            all_citations=all_citations,
        )

    async def ask_multi_notebooks(
        self,
        notebook_ids: list[str],
        question: str,
        notebook_titles: dict[str, str] | None = None,
    ) -> NotebookQueryAnswer:
        """Query multiple mock notebooks (backward-compatible alias)."""
        return await self.ask_notebooks(
            notebook_ids,
            question,
            notebook_titles=notebook_titles,
        )
