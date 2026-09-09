"""Mock in-memory notebook service implementing INotebookService."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from gemini_notebook_poc.model.notebook_answer import NotebookAnswer
from gemini_notebook_poc.model.notebook_info import NotebookInfo
from gemini_notebook_poc.model.source_info import SourceInfo
from gemini_notebook_poc.orchestrator import NotebookNotFoundError

logger = logging.getLogger("gemini_notebook_poc.services.notebook.mock")

INITIAL_MOCK_NOTEBOOKS = [
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

INITIAL_MOCK_SOURCES: dict[str, list[SourceInfo]] = {
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


class MockNotebookService:
    """In-memory implementation of INotebookService with full CRUD operations."""

    def __init__(self) -> None:
        self._notebooks: dict[str, NotebookInfo] = {
            nb.id: NotebookInfo(
                id=nb.id,
                title=nb.title,
                description=nb.description,
                created_at=nb.created_at,
                updated_at=nb.updated_at,
                source_count=nb.source_count,
            )
            for nb in INITIAL_MOCK_NOTEBOOKS
        }
        self._sources: dict[str, list[SourceInfo]] = {
            nid: list(srcs) for nid, srcs in INITIAL_MOCK_SOURCES.items()
        }

    async def create_notebook(self, title: str, description: str = "") -> NotebookInfo:
        now = datetime.now(UTC).isoformat()
        nb_id = f"nb-{uuid.uuid4().hex[:8]}"
        nb = NotebookInfo(
            id=nb_id,
            title=title.strip(),
            description=description.strip(),
            created_at=now,
            updated_at=now,
            source_count=0,
        )
        self._notebooks[nb_id] = nb
        self._sources[nb_id] = []
        logger.info("Created mock notebook '%s' (ID: %s)", nb.title, nb_id)
        return nb

    async def get_notebook(self, notebook_id: str) -> NotebookInfo:
        if notebook_id not in self._notebooks:
            raise NotebookNotFoundError(f"Notebook '{notebook_id}' not found.")
        return self._notebooks[notebook_id]

    async def list_notebooks(self) -> list[NotebookInfo]:
        return list(self._notebooks.values())

    async def update_notebook(
        self,
        notebook_id: str,
        title: str | None = None,
        description: str | None = None,
    ) -> NotebookInfo:
        nb = await self.get_notebook(notebook_id)
        if title is not None:
            nb.title = title.strip()
        if description is not None:
            nb.description = description.strip()
        nb.updated_at = datetime.now(UTC).isoformat()
        logger.info("Updated mock notebook ID %s", notebook_id)
        return nb

    async def delete_notebook(self, notebook_id: str) -> bool:
        if notebook_id not in self._notebooks:
            return False
        del self._notebooks[notebook_id]
        self._sources.pop(notebook_id, None)
        logger.info("Deleted mock notebook ID %s", notebook_id)
        return True

    async def list_sources(self, notebook_id: str) -> list[SourceInfo]:
        await self.get_notebook(notebook_id)
        return list(self._sources.get(notebook_id, []))

    async def add_source(
        self,
        notebook_id: str,
        title: str,
        content: str,
        source_type: str = "Document",
    ) -> SourceInfo:
        nb = await self.get_notebook(notebook_id)
        src_id = f"src-{uuid.uuid4().hex[:8]}"
        src = SourceInfo(
            id=src_id,
            title=title.strip(),
            source_type=source_type,
            snippet=content[:150] + "..." if len(content) > 150 else content,
            full_content=content,
        )
        if notebook_id not in self._sources:
            self._sources[notebook_id] = []
        self._sources[notebook_id].append(src)
        nb.source_count = len(self._sources[notebook_id])
        nb.updated_at = datetime.now(UTC).isoformat()
        logger.info("Added source '%s' to notebook '%s'", title, notebook_id)
        return src

    async def delete_source(self, notebook_id: str, source_id: str) -> bool:
        nb = await self.get_notebook(notebook_id)
        srcs = self._sources.get(notebook_id, [])
        initial_len = len(srcs)
        self._sources[notebook_id] = [s for s in srcs if s.id != source_id]
        deleted = len(self._sources[notebook_id]) < initial_len
        if deleted:
            nb.source_count = len(self._sources[notebook_id])
            nb.updated_at = datetime.now(UTC).isoformat()
            logger.info("Deleted source '%s' from notebook '%s'", source_id, notebook_id)
        return deleted

    async def query_notebook(self, notebook_id: str, question: str) -> NotebookAnswer:
        nb = await self.get_notebook(notebook_id)
        srcs = self._sources.get(notebook_id, [])

        if not srcs:
            return NotebookAnswer(
                notebook_id=notebook_id,
                notebook_title=nb.title,
                answer="No documents are currently ingested in this notebook to answer the question.",
                citations=[],
                success=True,
            )

        citations = [f"Source: {s.title} (ID: {s.id})" for s in srcs[:2]]
        answer_text = (
            f"Based on grounded findings in '{nb.title}', regarding: '{question}':\n"
            f"- Information synthesized from {len(srcs)} source document(s).\n"
            f'- Relevant excerpt: "{srcs[0].snippet}"'
        )
        return NotebookAnswer(
            notebook_id=notebook_id,
            notebook_title=nb.title,
            answer=answer_text,
            citations=citations,
            success=True,
        )
