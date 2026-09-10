"""Unit tests for configuration loading and environment parsing."""

import os
from unittest.mock import patch

from gemini_notebook_poc.application_configuration import ApplicationConfiguration


def test_config_defaults():
    config = ApplicationConfiguration()
    assert config.backend_mode == "enterprise"
    assert config.gcp_location == "global"
    assert config.gemini_model == "gemini-2.5-flash"


def test_config_load_from_env():
    with patch.dict(
        os.environ,
        {
            "BACKEND_MODE": "mock",
            "GCP_PROJECT_ID": "test-project-123",
            "GCP_LOCATION": "us",
            "GEMINI_API_KEY": "test-key-abc",
            "GEMINI_MODEL": "gemini-2.5-pro",
        },
    ):
        config = ApplicationConfiguration.load()
        assert config.backend_mode == "mock"
        assert config.gcp_project_id == "test-project-123"
        assert config.gcp_location == "us"
        assert config.gemini_api_key == "test-key-abc"
        assert config.gemini_model == "gemini-2.5-pro"


def test_get_bearer_token_explicit():
    config = ApplicationConfiguration(gcp_access_token="ya29.test-explicit-token")
    assert config.get_bearer_token() == "ya29.test-explicit-token"


def test_load_env_file_basic(tmp_path):
    from gemini_notebook_poc.application_configuration import load_env_file

    env_path = tmp_path / ".env"
    env_path.write_text("TEST_KEY_FOO=bar_value\n# comment\n", encoding="utf-8")

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TEST_KEY_FOO", None)
        load_env_file(env_path)
        assert os.environ.get("TEST_KEY_FOO") == "bar_value"
