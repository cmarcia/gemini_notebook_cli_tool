"""Application configuration using Python standard library os.getenv."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Literal

import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
from pydantic import BaseModel, Field


def load_env_file(path: Path | str | None = None) -> None:
    """Populate os.environ from .env file using standard library without overriding existing vars."""
    env_file = Path(path) if path else Path.cwd() / ".env"
    if not env_file.is_file():
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_file.is_file():
        return

    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except OSError:
        pass


class ApplicationConfiguration(BaseModel):
    """Application configuration for Gemini Notebook PoC."""

    backend_mode: Literal["enterprise", "notebooklm", "mock"] = Field(
        default="enterprise",
        description="Backend to use: enterprise (GCP Discovery Engine), notebooklm (NotebookLM web), or mock (offline demo).",
    )
    gcp_project_id: str = Field(
        default="",
        description="Google Cloud Project ID with Gemini Enterprise / Discovery Engine enabled.",
    )
    gcp_location: str = Field(
        default="global",
        description="Location for Google Cloud Discovery Engine (e.g., global, us, eu).",
    )
    gcp_access_token: str = Field(
        default="",
        description="Optional OAuth bearer token for GCP API calls.",
    )
    google_application_credentials: str = Field(
        default="",
        description="Optional path to service account JSON key file.",
    )
    gemini_api_key: str = Field(
        default="",
        description="API Key for Gemini models (Google AI Studio / GCP) used for source-grounded answers.",
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier for chat and grounded synthesis.",
    )
    notebooklm_auth_token: str = Field(
        default="",
        description="Optional session token/cookie string if using notebooklm backend.",
    )

    @classmethod
    def load(cls) -> ApplicationConfiguration:
        """Create config from standard environment variables via os.getenv."""
        load_env_file()
        return cls(
            backend_mode=os.getenv("BACKEND_MODE", "enterprise").lower(),  # type: ignore[arg-type]
            gcp_project_id=os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", "")),
            gcp_location=os.getenv("GCP_LOCATION", "global"),
            gcp_access_token=os.getenv("GCP_ACCESS_TOKEN", ""),
            google_application_credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            notebooklm_auth_token=os.getenv("NOTEBOOKLM_AUTH_TOKEN", ""),
        )

    def get_bearer_token(self) -> str:
        """Resolve a valid GCP bearer token for Discovery Engine calls."""
        # 1. Explicit token in config
        if self.gcp_access_token.strip():
            return self.gcp_access_token.strip()

        # 2. Service account key file
        if (
            self.google_application_credentials
            and Path(self.google_application_credentials).is_file()
        ):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    self.google_application_credentials,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                request = google.auth.transport.requests.Request()
                creds.refresh(request)
                if creds.token:
                    return creds.token
            except Exception:
                pass

        # 3. gcloud auth print-access-token CLI helper
        gcloud_bin = shutil.which("gcloud")
        if gcloud_bin and Path(gcloud_bin).is_file():
            try:
                res = subprocess.run(
                    [gcloud_bin, "auth", "print-access-token"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                token = res.stdout.strip()
                if token.startswith("ya29."):
                    return token
            except Exception:
                pass

        # 4. Standard Application Default Credentials (ADC)
        try:
            creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            request = google.auth.transport.requests.Request()
            creds.refresh(request)
            if creds.token:
                return creds.token
        except Exception:
            pass

        raise RuntimeError(
            "No valid GCP Bearer token found. Please do one of the following:\n"
            "  1. Run `gcloud auth login` or `gcloud auth print-access-token`.\n"
            "  2. Set `GCP_ACCESS_TOKEN=<token>` in your .env file.\n"
            "  3. Set `GOOGLE_APPLICATION_CREDENTIALS=<path-to-sa-key.json>` in your .env file."
        )
