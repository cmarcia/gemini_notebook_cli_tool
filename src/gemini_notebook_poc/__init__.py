"""Gemini Notebook / NotebookLM Enterprise PoC entry point."""

from __future__ import annotations

import asyncio
import sys

from gemini_notebook_poc.cli import run_cli


def main() -> None:
    """Synchronous entry point for project.scripts."""
    try:
        asyncio.run(run_cli())
    except SystemExit as e:
        sys.exit(e.code)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
