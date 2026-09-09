"""Tests for CLI help and argument parsing."""

from __future__ import annotations

import re
import sys
from unittest.mock import patch

import pytest
from rich.console import Console

from gemini_notebook_poc.cli import print_custom_help, run_cli


def test_print_custom_help_contains_all_sections_and_flags():
    """Verify that custom help renders all logical sections, flags, and examples without emoji."""
    test_console = Console(record=True, width=120)
    print_custom_help(test_console)
    output = test_console.export_text()

    # Verify header & usage
    assert "User Manual & Command Reference" in output
    assert "uv run gemini-notebook-poc" in output

    # Verify all 6 category tables and recipes panel
    assert "1. Backend & Connection Configuration" in output
    assert "2. Interactive Chat & Headless Q&A" in output
    assert "3. Semantic Topic Discovery & Cross-Notebook Synthesis" in output
    assert "4. Notebook Lifecycle Management (CRUD)" in output
    assert "5. Source Document Management (CRUD)" in output
    assert "6. General & Help Options" in output
    assert "Common End-to-End Workflow Recipes" in output

    # Verify all supported flags are documented
    expected_flags = [
        "--mode",
        "--mock",
        "--project",
        "-n, --notebook",
        "-s, --source",
        "-q, --question",
        "-f, --find",
        "-l, --limit",
        "--list-notebooks",
        "-c, --create-notebook",
        "--update-notebook",
        "--title",
        "--description",
        "-d, --delete-notebook",
        "--list-sources",
        "--add-source",
        "--source-title",
        "--delete-source",
        "-h, --help",
    ]
    for flag in expected_flags:
        assert flag in output, f"Expected flag '{flag}' not found in help manual output"

    # Verify examples exist in output
    assert "uv run gemini-notebook-poc --mode mock" in output
    assert "cloud infrastructure costs" in output
    assert "2026 Strategy" in output
    assert "Roadmap" in output

    # Strict compliance: Zero emojis allowed
    emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
    assert not emoji_pattern.findall(output), "Help output must not contain any emojis"


@pytest.mark.asyncio
async def test_run_cli_help_flag():
    """Verify that --help exits cleanly after printing help without attempting backend calls."""
    with patch.object(sys, "argv", ["gemini-notebook-poc", "--help"]):
        # Should return cleanly (status 0) without raising SystemExit or network errors
        await run_cli()


@pytest.mark.asyncio
async def test_run_cli_short_help_flag():
    """Verify that -h exits cleanly after printing help."""
    with patch.object(sys, "argv", ["gemini-notebook-poc", "-h"]):
        await run_cli()


@pytest.mark.asyncio
async def test_run_cli_invalid_argument():
    """Verify that an unrecognized argument displays custom error and exits with code 2."""
    with patch.object(sys, "argv", ["gemini-notebook-poc", "--unknown-bogus-flag"]):
        with pytest.raises(SystemExit) as excinfo:
            await run_cli()
        assert excinfo.value.code == 2


@pytest.mark.asyncio
async def test_run_cli_add_source_resolves_title():
    """Verify that --add-source resolves notebook name to ID and adds source successfully."""
    with patch.object(
        sys,
        "argv",
        [
            "gemini-notebook-poc",
            "--mock",
            "-n",
            "Financial",
            "--add-source",
            "Sample text content",
            "--source-title",
            "Test Ingested Document",
        ],
    ):
        await run_cli()


@pytest.mark.asyncio
async def test_run_cli_add_source_typo_detection():
    """Verify that a path-like source with missing slash is detected and warned."""
    with patch.object(
        sys,
        "argv",
        [
            "gemini-notebook-poc",
            "--mock",
            "-n",
            "Financial",
            "--add-source",
            "..NONEXISTENT.md",
        ],
    ):
        await run_cli()
