"""Interactive rich console interface for Gemini Notebook PoC."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from gemini_notebook_poc.backends import get_backend
from gemini_notebook_poc.backends.base import BaseNotebookBackend
from gemini_notebook_poc.backends.enterprise import EnterpriseAPIError
from gemini_notebook_poc.config import AppConfig
from gemini_notebook_poc.model import (
    NotebookInfo,
    NotebookMatch,
    NotebookQueryAnswer,
    SourceInfo,
)

warnings.filterwarnings("ignore")

console = Console()


def print_banner(config: AppConfig) -> None:
    """Print the application welcome banner and current configuration status."""
    key_status = (
        "[green]Configured[/green]"
        if config.gemini_api_key
        else "[yellow]Not set (answers will be simulated/raw)[/yellow]"
    )
    mode_color = "cyan" if config.backend_mode == "enterprise" else "magenta"
    details = (
        f"[bold]Backend Mode:[/bold] [{mode_color}]{config.backend_mode.upper()}[/{mode_color}]\n"
        f"[bold]GCP Project ID:[/bold] {config.gcp_project_id or '[dim]Not configured[/dim]'}\n"
        f"[bold]GCP Location:[/bold] {config.gcp_location}\n"
        f"[bold]Gemini Model:[/bold] {config.gemini_model}\n"
        f"[bold]Gemini API Key:[/bold] {key_status}"
    )
    console.print(
        Panel(
            details,
            title="[bold blue]Gemini Notebook / NotebookLM Enterprise PoC[/bold blue]",
            subtitle="Proof of Concept Console",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


def display_notebooks_table(notebooks: list[NotebookInfo]) -> None:
    """Render a styled table of notebooks."""
    table = Table(
        title="Available Notebooks",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Notebook Name", style="bold green", min_width=30)
    table.add_column("Notebook ID", style="cyan", min_width=24)
    table.add_column("Sources", justify="center", width=9)
    table.add_column("Updated", style="dim", width=20)

    for idx, nb in enumerate(notebooks, 1):
        updated = nb.updated_at[:10] if nb.updated_at else "N/A"
        table.add_row(
            str(idx),
            nb.title,
            nb.id,
            str(nb.source_count),
            updated,
        )
    console.print(table)


def display_sources_table(sources: list[SourceInfo], notebook_title: str) -> None:
    """Render a styled table of sources inside a notebook."""
    table = Table(
        title=f"Sources in '{notebook_title}'",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Notebook Name", style="bold green", min_width=26)
    table.add_column("Source Title", style="bold yellow", min_width=28)
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Source ID", style="dim", min_width=20)
    table.add_column("Preview / Snippet", style="white", max_width=35)

    for idx, s in enumerate(sources, 1):
        snippet = s.snippet.replace("\n", " ")[:35] + ("..." if len(s.snippet) > 35 else "")
        table.add_row(
            str(idx),
            notebook_title,
            s.title,
            s.source_type,
            s.id,
            snippet or "[dim]No preview[/dim]",
        )
    console.print(table)


async def chat_loop(
    backend: BaseNotebookBackend,
    notebook: NotebookInfo,
    selected_source: SourceInfo | None,
) -> None:
    """Run the interactive console chat session grounded on a notebook/source."""
    source_label = selected_source.title if selected_source else "All Sources in Notebook"

    console.print(
        Panel(
            f"[bold]Active Notebook:[/bold] [green]{notebook.title}[/green] ([dim]{notebook.id}[/dim])\n"
            f"[bold]Active Grounding Source:[/bold] [yellow]{source_label}[/yellow]\n\n"
            "[dim]Commands: [/dim][cyan]/sources[/cyan] (list sources) | [cyan]/back[/cyan] (change source) | "
            "[cyan]/notebook[/cyan] (change notebook) | [cyan]/clear[/cyan] (reset history) | [cyan]/exit[/cyan] (quit)",
            title="[bold green]Chat Session - Ask Questions About Source[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    history: list[dict[str, Any]] = []

    while True:
        try:
            prompt_text = f"[bold green]{notebook.title[:15]}..[/bold green] [yellow]({source_label[:20]})[/yellow] > "
            user_question = Prompt.ask(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting chat session.[/dim]")
            break

        if not user_question:
            continue

        cmd = user_question.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)
        elif cmd in ("/sources", "/sources-list"):
            with console.status(
                f"[bold cyan]Loading sources for '{notebook.title}'...[/bold cyan]"
            ):
                try:
                    srcs = await backend.list_sources(notebook.id)
                except Exception as e:
                    console.print(f"[red]Error loading sources:[/red] {e}")
                    srcs = []
            display_sources_table(srcs, notebook.title)
            continue
        elif cmd == "/back" or cmd == "/notebook":
            return
        elif cmd == "/clear":
            history.clear()
            console.print("[dim]Chat history cleared.[/dim]")
            continue

        # Ask question with spinner
        with console.status(
            "[bold cyan]Querying notebook and grounding with source...[/bold cyan]"
        ):
            try:
                answer_obj = await backend.ask_question(
                    notebook_id=notebook.id,
                    source_id=selected_source.id if selected_source else None,
                    question=user_question,
                    history=history,
                )
            except Exception as e:
                console.print(f"[bold red]Error querying source:[/bold red] {e}")
                continue

        # Render response
        console.print()
        console.print(
            Panel(
                Markdown(answer_obj.answer),
                title=f"[bold cyan]Gemini Response ({answer_obj.model_name or 'Grounded'})[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

        if answer_obj.citations:
            cit_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
            cit_table.add_column("Bullet", style="cyan", width=3)
            cit_table.add_column("Citation", style="italic dim")
            for c in answer_obj.citations:
                cit_table.add_row("-", c)
            console.print(
                Panel(
                    cit_table,
                    title="[dim]Citations & Grounding Sources[/dim]",
                    border_style="dim",
                    box=box.ROUNDED,
                )
            )
        console.print()

        # Update history
        history.append({"role": "user", "content": user_question})
        history.append({"role": "model", "content": answer_obj.answer})


def resolve_notebook_selection(
    user_input: str | list[str], notebooks: list[NotebookInfo]
) -> list[NotebookInfo]:
    """Parse notebook identifiers (titles with spaces, numbers, or IDs) into NotebookInfo list.

    Supports:
    - Repeated flags: ['Cloudinary Architecture', 'Event-Driven Microservices']
    - Comma-separated strings: 'Cloudinary Architecture, Event-Driven Microservices'
    - Index numbers: '1, 2' or ['1', '2']
    - Direct notebook IDs
    """
    if isinstance(user_input, str):
        raw_items = [user_input]
    else:
        raw_items = list(user_input)

    tokens: list[str] = []
    for item in raw_items:
        item_clean = item.strip()
        if not item_clean:
            continue
        # If item has a comma, check if it matches a single notebook title directly first
        direct_match = any(
            item_clean.lower() == nb.title.lower() or item_clean == nb.id for nb in notebooks
        )
        if direct_match or "," not in item_clean:
            tokens.append(item_clean)
        else:
            tokens.extend([t.strip() for t in item_clean.split(",") if t.strip()])

    matched: list[NotebookInfo] = []
    seen_ids: set[str] = set()

    for token in tokens:
        match_nb: NotebookInfo | None = None
        if token.isdigit():
            idx = int(token) - 1
            if 0 <= idx < len(notebooks):
                match_nb = notebooks[idx]
        if not match_nb:
            token_lower = token.lower()
            for nb in notebooks:
                if (
                    nb.id == token
                    or nb.id.lower().startswith(token_lower)
                    or token_lower in nb.title.lower()
                ):
                    match_nb = nb
                    break
        if match_nb and match_nb.id not in seen_ids:
            matched.append(match_nb)
            seen_ids.add(match_nb.id)

    return matched


def display_multi_answer(multi_ans: NotebookQueryAnswer) -> None:
    """Render a synthesized cross-notebook response and attribution breakdown."""
    console.print()
    console.print(
        Panel(
            Markdown(multi_ans.synthesized_answer),
            title=f"[bold magenta]Cross-Notebook Synthesized Answer ({multi_ans.model_name})[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        )
    )

    table = Table(
        title="Notebook Contributions",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Notebook Name", style="bold green", min_width=25)
    table.add_column("Notebook ID", style="cyan", min_width=20)
    table.add_column("Status", justify="center", width=12)
    table.add_column("Citations", justify="center", width=10)
    table.add_column("Answer Snippet", style="dim", max_width=35)

    for idx, ans in enumerate(multi_ans.notebook_answers, 1):
        status = "[green]Success[/green]" if ans.success else "[red]Error[/red]"
        clean_text = ans.answer.replace("\n", " ").strip()
        preview = clean_text[:35] + "..." if len(clean_text) > 35 else clean_text
        table.add_row(
            str(idx),
            ans.notebook_title,
            ans.notebook_id,
            status,
            str(len(ans.citations)),
            preview or "[dim]None[/dim]",
        )
    console.print(table)

    if multi_ans.all_citations:
        cit_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
        cit_table.add_column("Bullet", style="cyan", width=3)
        cit_table.add_column("Citation", style="italic dim")
        for c in multi_ans.all_citations:
            cit_table.add_row("-", c)
        console.print(
            Panel(
                cit_table,
                title="[dim]Attributed Grounding Citations Across Notebooks[/dim]",
                border_style="dim",
                box=box.ROUNDED,
            )
        )
    console.print()


def display_multi_sources(
    notebooks: list[NotebookInfo], sources_map: dict[str, list[SourceInfo]]
) -> None:
    """Render styled tables of sources across multiple notebooks."""
    total_sources = sum(len(sources_map.get(nb.id, [])) for nb in notebooks)
    console.print()
    console.print(
        Panel(
            f"Found a total of [bold cyan]{total_sources} source(s)[/bold cyan] across [bold green]{len(notebooks)} notebook(s)[/bold green].",
            title="[bold cyan]Notebook Sources Inventory[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )
    table = Table(
        title="All Sources Across Selected Notebooks",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Notebook Name", style="bold green", min_width=24)
    table.add_column("Source Title", style="bold yellow", min_width=28)
    table.add_column("Type", style="magenta", width=16)
    table.add_column("Source ID", style="dim", min_width=18)
    table.add_column("Preview / Snippet", style="white", max_width=35)

    counter = 1
    for nb in notebooks:
        sources = sources_map.get(nb.id, [])
        for s in sources:
            snippet = s.snippet.replace("\n", " ")[:35] + ("..." if len(s.snippet) > 35 else "")
            table.add_row(
                str(counter),
                nb.title,
                s.title,
                s.source_type,
                s.id,
                snippet or "[dim]No preview[/dim]",
            )
            counter += 1
    console.print(table)


def display_search_results(matches: list[NotebookMatch], topic: str) -> None:
    """Render a styled table of topic search matches."""
    table = Table(
        title=f"Topic Search: '{topic}' ({len(matches)} matches)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Notebook Name", style="bold green", min_width=28)
    table.add_column("Notebook ID", style="cyan", min_width=24)
    table.add_column("Relevance", style="bold yellow", justify="center", width=11)
    table.add_column("Why It Matches", style="white", min_width=32)

    for idx, m in enumerate(matches, 1):
        score_str = f"{m.relevance}/5"
        table.add_row(
            str(idx),
            m.notebook_title,
            m.notebook_id,
            f"[yellow]{score_str}[/yellow]",
            m.reason,
        )
    console.print()
    console.print(table)


async def multi_chat_loop(
    backend: BaseNotebookBackend,
    notebooks: list[NotebookInfo],
) -> None:
    """Run interactive chat session across multiple notebooks."""
    console.print(
        Panel(
            f"[bold]Active Multi-Notebook Group ({len(notebooks)} notebooks):[/bold]\n"
            + "\n".join(
                f"  • [green]{nb.title}[/green] ([dim]{nb.id[:12]}...[/dim])" for nb in notebooks
            )
            + "\n\n[dim]Commands: [/dim][cyan]/sources[/cyan] (list all sources) | [cyan]/notebook[/cyan] (change notebooks) | [cyan]/exit[/cyan] (quit)",
            title="[bold magenta]Multi-Notebook Chat Session (Cross-Notebook Synthesis)[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        )
    )

    while True:
        try:
            prompt_text = f"[bold magenta]Multi-NB ({len(notebooks)})[/bold magenta] > "
            user_question = Prompt.ask(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Exiting chat session.[/dim]")
            break

        if not user_question:
            continue

        cmd = user_question.lower()
        if cmd in ("/exit", "/quit", "exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            sys.exit(0)
        elif cmd in ("/sources", "/source", "/sources-list"):
            with console.status(
                f"[bold cyan]Fetching sources for {len(notebooks)} notebooks...[/bold cyan]"
            ):
                sources_map = await backend.list_multi_sources([nb.id for nb in notebooks])
            display_multi_sources(notebooks, sources_map)
            continue
        elif cmd in ("/back", "/notebook"):
            return

        with console.status(
            f"[bold cyan]Querying {len(notebooks)} notebooks in parallel and synthesizing...[/bold cyan]"
        ):
            try:
                multi_ans = await backend.ask_notebooks(
                    notebook_ids=[nb.id for nb in notebooks],
                    question=user_question,
                    notebook_titles={nb.id: nb.title for nb in notebooks},
                )
            except Exception as e:
                console.print(f"[bold red]Error querying notebooks:[/bold red] {e}")
                continue

        display_multi_answer(multi_ans)


def print_custom_help(target_console: Console | None = None) -> None:
    """Display comprehensive command-line reference manual and usage instructions."""
    out = target_console or console

    # Header Panel
    out.print(
        Panel(
            "[bold white]A unified CLI for Google Cloud Discovery Engine NotebookLM Enterprise, personal NotebookLM,\n"
            "and offline mock simulation backends. Supports semantic topic discovery, document grounding,\n"
            "parallel multi-notebook synthesis, and complete notebook/source lifecycle management.[/bold white]\n\n"
            "[bold yellow]Syntax:[/bold yellow]\n"
            "  [bold green]uv run gemini-notebook-poc[/bold green] [cyan][OPTIONS][/cyan]\n"
            "  [bold green]gemini-notebook-poc[/bold green] [cyan][OPTIONS][/cyan]\n\n"
            "[dim]Note: Running without arguments launches the interactive console navigation interface.[/dim]",
            title="[bold blue]Gemini Notebook / NotebookLM Enterprise PoC - User Manual & Command Reference[/bold blue]",
            subtitle="CLI Documentation & Usage Guide",
            border_style="blue",
            box=box.ROUNDED,
        )
    )

    # 1. Backend & Connection Configuration
    backend_table = Table(
        title="1. Backend & Connection Configuration",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    backend_table.add_column("Option / Flag", style="bold cyan", width=22)
    backend_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    backend_table.add_column("Example Usage", style="green", min_width=40)

    backend_table.add_row(
        "--mode <MODE>",
        "Select the operational backend engine.\n"
        "[bold]Allowed values:[/bold] enterprise, notebooklm, mock\n"
        "[bold]Default:[/bold] Loaded from .env (BACKEND_MODE) or 'enterprise'\n"
        "[bold]Constraints:[/bold] enterprise requires GCP ADC; notebooklm requires 'uv run notebooklm login'.",
        "uv run gemini-notebook-poc --mode mock\n"
        "uv run gemini-notebook-poc --mode enterprise\n"
        "uv run gemini-notebook-poc --mode notebooklm",
    )
    backend_table.add_row(
        "--mock",
        "Fast shorthand flag for '--mode mock'.\n"
        "Instantly starts the CLI in offline simulation mode with pre-seeded\n"
        "sample notebooks and synthesized responses. Does not require GCP\n"
        "credentials, .env configuration, or internet access.",
        "uv run gemini-notebook-poc --mock\n"
        "uv run gemini-notebook-poc --mock --list\n"
        "uv run gemini-notebook-poc --mock -n 'Financial Reports' -q 'Q3 revenue?'",
    )
    backend_table.add_row(
        "--project <ID>",
        "Specify or override the Google Cloud Project ID.\n"
        "[bold]Default:[/bold] Loaded from .env (GCP_PROJECT_ID) or GCP ADC.\n"
        "[bold]Constraints:[/bold] Applied only when running with '--mode enterprise'.",
        "uv run gemini-notebook-poc --project my-company-prod-123\n"
        "uv run gemini-notebook-poc --project dev-analytics-456 --list-notebooks",
    )
    out.print(backend_table)
    out.print()

    # 2. Interactive Chat & Headless Q&A
    chat_table = Table(
        title="2. Interactive Chat & Headless Q&A",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    chat_table.add_column("Option / Flag", style="bold cyan", width=22)
    chat_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    chat_table.add_column("Example Usage", style="green", min_width=40)

    chat_table.add_row(
        "-n, --notebook <NAME|ID>",
        "Target one or more notebooks directly by title substring, ID, or index #.\n"
        "[bold]Multi-notebook:[/bold] Repeat flag (-n 'NB 1' -n 'NB 2'), use comma separation\n"
        "(-n 'NB 1, NB 2'), or pass 'all' to select all available notebooks.\n"
        "Single notebook starts single-notebook session; multiple notebooks triggers\n"
        "parallel cross-notebook synthesis.",
        "uv run gemini-notebook-poc -n 'Q3 Financials'\n"
        "uv run gemini-notebook-poc -n 1\n"
        "uv run gemini-notebook-poc -n nb-101-alpha\n"
        "uv run gemini-notebook-poc -n 'Q3 Financials' -n 'Roadmap'\n"
        "uv run gemini-notebook-poc -n 'Q3 Financials, Roadmap'\n"
        "uv run gemini-notebook-poc -n all --sources",
    )
    chat_table.add_row(
        "-s, --source <NAME|ID>",
        "Filter and ground questions on a specific source document within a notebook.\n"
        "[bold]Allowed values:[/bold] Source title substring, source ID, 1-based index, or 'all'.\n"
        "[bold]Constraints:[/bold] Requires a single notebook via -n or interactive selection.",
        "uv run gemini-notebook-poc -n 'Architecture' -s 'System Specs'\n"
        "uv run gemini-notebook-poc -n 'Architecture' -s src-401 -q 'Explain data flow'\n"
        "uv run gemini-notebook-poc -n 'Architecture' -s 2",
    )
    chat_table.add_row(
        "-q, --question <TEXT>",
        "Directly ask a question non-interactively and exit.\n"
        "Outputs the grounded Gemini answer with attributed citations.\n"
        "[bold]Modes:[/bold] Grounds against single notebook/source, synthesizes across\n"
        "multiple notebooks in parallel, or pairs with -f/--find for automatic synthesis.",
        "uv run gemini-notebook-poc -n 'HR Policies' -q 'PTO carry over policy?'\n"
        "uv run gemini-notebook-poc -n 'Specs' -n 'Roadmap' -q 'Identify conflicts'\n"
        "uv run gemini-notebook-poc --mock -n 'Financial Reports' -q 'Net margin?'",
    )
    out.print(chat_table)
    out.print()

    # 3. Semantic Topic Discovery & Cross-Notebook Synthesis
    search_table = Table(
        title="3. Semantic Topic Discovery & Cross-Notebook Synthesis",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    search_table.add_column("Option / Flag", style="bold cyan", width=22)
    search_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    search_table.add_column("Example Usage", style="green", min_width=40)

    search_table.add_row(
        "-f, --find <TOPIC>",
        "Perform semantic topic discovery across all notebooks using Gemini reasoning.\n"
        "Matches conceptual themes beyond simple keyword lookup.\n"
        "[bold]Standalone:[/bold] Renders ranked matches with relevance scores (1-5) and rationale.\n"
        "[bold]With -q:[/bold] Queries top matching notebooks in parallel and synthesizes a unified answer.",
        "uv run gemini-notebook-poc -f 'enterprise security compliance'\n"
        "uv run gemini-notebook-poc -f 'cloud infrastructure costs' -q 'Cost drivers?'\n"
        "uv run gemini-notebook-poc --mock -f 'financial performance'",
    )
    search_table.add_row(
        "-l, --limit <NUMBER>",
        "Maximum number of top-ranked notebooks to synthesize across when combining\n"
        "-f/--find with -q/--question.\n"
        "[bold]Default:[/bold] 3\n"
        "[bold]Constraints:[/bold] Must be a positive integer.",
        "uv run gemini-notebook-poc -f 'budget' -q 'Summarize variances' -l 2\n"
        "uv run gemini-notebook-poc -f 'security' -q 'Open vulnerabilities' -l 5",
    )
    out.print(search_table)
    out.print()

    # 4. Notebook Lifecycle Management (CRUD)
    nb_table = Table(
        title="4. Notebook Lifecycle Management (CRUD)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    nb_table.add_column("Option / Flag", style="bold cyan", width=22)
    nb_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    nb_table.add_column("Example Usage", style="green", min_width=40)

    nb_table.add_row(
        "--list-notebooks, --list",
        "Display all available notebooks in the active backend in a formatted table\n"
        "showing Index (#), Notebook Name, Notebook ID, Source Count, and Updated Date, then exit.",
        "uv run gemini-notebook-poc --list-notebooks\n"
        "uv run gemini-notebook-poc --list\n"
        "uv run gemini-notebook-poc --mock --list",
    )
    nb_table.add_row(
        "-c, --create-notebook <TITLE>",
        "Create a new notebook with the given title and exit.\n"
        "Optionally pair with '--description' to supply notebook metadata.",
        "uv run gemini-notebook-poc -c '2026 Strategy Plan'\n"
        "uv run gemini-notebook-poc -c '2026 Strategy Plan' --description 'Strategic goals'",
    )
    nb_table.add_row(
        "--update-notebook <ID>",
        "Update the title and/or description of an existing notebook by its ID.\n"
        "[bold]Constraints:[/bold] Requires '--title' and/or '--description'.",
        "uv run gemini-notebook-poc --update-notebook nb-101 --title 'New Title'\n"
        "uv run gemini-notebook-poc --update-notebook nb-101 --description 'Updated desc'\n"
        "uv run gemini-notebook-poc --update-notebook nb-101 --title 'Title' --description 'Desc'",
    )
    nb_table.add_row(
        "--title <TEXT>",
        "Specify the new title when updating a notebook with '--update-notebook'.",
        "uv run gemini-notebook-poc --update-notebook nb-101 --title 'Q4 Review'",
    )
    nb_table.add_row(
        "--description <TEXT>",
        "Specify description text when creating (-c) or updating (--update-notebook) a notebook.",
        "uv run gemini-notebook-poc -c 'Incidents' --description 'Post-mortem reports'",
    )
    nb_table.add_row(
        "-d, --delete-notebook <ID>",
        "Permanently delete a notebook by its unique ID and exit.\n"
        "[bold]Caution:[/bold] Removes the notebook and all associated sources from backend.",
        "uv run gemini-notebook-poc -d nb-101\n"
        "uv run gemini-notebook-poc --delete-notebook nb-archive-test",
    )
    out.print(nb_table)
    out.print()

    # 5. Source Document Management (CRUD)
    src_table = Table(
        title="5. Source Document Management (CRUD)",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    src_table.add_column("Option / Flag", style="bold cyan", width=22)
    src_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    src_table.add_column("Example Usage", style="green", min_width=40)

    src_table.add_row(
        "--list-sources, --sources",
        "List all source documents inside specified notebook(s) in a table and exit.\n"
        "[bold]Constraints:[/bold] Requires -n/--notebook (single notebook, multiple, or 'all').",
        "uv run gemini-notebook-poc -n 'Financial Reports' --sources\n"
        "uv run gemini-notebook-poc -n 'Financial Reports' -n 'Legal' --list-sources\n"
        "uv run gemini-notebook-poc -n all --sources",
    )
    src_table.add_row(
        "--add-source <FILE|TEXT>",
        "Add a new document source to a notebook. Accepts a local file path (.md, .txt,\n"
        ".json, .csv) or an inline raw text string.\n"
        "[bold]Constraints:[/bold] Requires -n/--notebook. If a file path is given, its filename is used\n"
        "as default title unless '--source-title' is specified.",
        "uv run gemini-notebook-poc -n 'Strategy' --add-source ./notes/q3.txt\n"
        "uv run gemini-notebook-poc -n 'Strategy' --add-source ./data/q3.md --source-title 'Q3 Exec'\n"
        "uv run gemini-notebook-poc -n 'Strategy' --add-source 'ARR up 25% YoY.' --source-title 'ARR'",
    )
    src_table.add_row(
        "--source-title <TITLE>",
        "Set a custom title for the source being added via '--add-source'.\n"
        "[bold]Constraints:[/bold] Used in combination with '--add-source'.",
        "uv run gemini-notebook-poc -n 'Legal' --add-source ./tos.txt --source-title 'Terms of Service v3'",
    )
    src_table.add_row(
        "--delete-source <ID>",
        "Delete a specific source document by ID from a notebook and exit.\n"
        "[bold]Constraints:[/bold] Requires -n/--notebook to identify parent notebook.",
        "uv run gemini-notebook-poc -n 'Financial Reports' --delete-source src-401",
    )
    out.print(src_table)
    out.print()

    # 6. General & Help
    help_table = Table(
        title="6. General & Help Options",
        box=box.ROUNDED,
        header_style="bold cyan",
        title_style="bold white",
        expand=True,
    )
    help_table.add_column("Option / Flag", style="bold cyan", width=22)
    help_table.add_column("Description, Values & Constraints", style="white", min_width=44)
    help_table.add_column("Example Usage", style="green", min_width=40)

    help_table.add_row(
        "-h, --help",
        "Display this comprehensive user manual with in-depth descriptions,\n"
        "accepted options, constraints, and copy-pasteable CLI command recipes, then exit.",
        "uv run gemini-notebook-poc --help\nuv run gemini-notebook-poc -h",
    )
    out.print(help_table)
    out.print()

    # 7. Common End-to-End Workflow Recipes
    recipes = (
        "[bold yellow]1. Semantic Discovery and Synthesis:[/bold yellow]\n"
        "   Find notebooks discussing cloud costs and immediately synthesize an answer across the top 2 matches:\n"
        "   [bold green]uv run gemini-notebook-poc -f 'cloud infrastructure costs' -q 'What were our biggest cost drivers?' -l 2[/bold green]\n\n"
        "[bold yellow]2. Multi-Notebook Parallel Comparison:[/bold yellow]\n"
        "   Ask a question across two specific notebooks to compare technical alignment:\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Product Spec' -n 'Sales Playbook' -q 'Does the pitch match the spec?'[/bold green]\n\n"
        "[bold yellow]3. Targeted Document Deep Dive:[/bold yellow]\n"
        "   Ground questions strictly on a single document within a notebook:\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Architecture' -s 'System Specs' -q 'What is the cache policy?'[/bold green]\n\n"
        "[bold yellow]4. Ingestion and Verification Workflow:[/bold yellow]\n"
        "   Add a document to a notebook and verify it in the sources inventory:\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Roadmap' --add-source ./notes/2026_q1.md --source-title '2026 Q1 OKRs'[/bold green]\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Roadmap' --sources[/bold green]\n\n"
        "[bold yellow]5. Offline Development & Testing (Mock Mode):[/bold yellow]\n"
        "   Test CLI features, queries, and multi-notebook synthesis offline without GCP credentials:\n"
        "   [bold green]uv run gemini-notebook-poc --mock -n 'Financial Reports' -q 'What was the Q3 revenue growth?'[/bold green]\n\n"
        "[bold yellow]6. Complete Notebook Lifecycle Pipeline:[/bold yellow]\n"
        "   Create a notebook, ingest notes, ask questions, and clean up:\n"
        "   [bold green]uv run gemini-notebook-poc -c 'Sprint Retrospectives' --description 'Sprint review notes'[/bold green]\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Sprint Retrospectives' --add-source ./retro.md --source-title 'Sprint 42'[/bold green]\n"
        "   [bold green]uv run gemini-notebook-poc -n 'Sprint Retrospectives' -q 'What went well in Sprint 42?'[/bold green]\n"
        "   [bold green]uv run gemini-notebook-poc -d <NOTEBOOK_ID>[/bold green]"
    )
    out.print(
        Panel(
            recipes,
            title="[bold blue]Common End-to-End Workflow Recipes[/bold blue]",
            subtitle="Copy-Pasteable Recipes",
            border_style="blue",
            box=box.ROUNDED,
        )
    )


async def run_cli() -> None:
    """Main CLI interaction flow."""
    parser = argparse.ArgumentParser(
        description="Gemini Notebook / NotebookLM Enterprise PoC",
        add_help=False,
    )

    def custom_error(message: str) -> None:
        console.print(f"[bold red]Error:[/bold red] {message}\n")
        console.print(
            "[dim]Run with [bold cyan]--help[/bold cyan] or [bold cyan]-h[/bold cyan] to see full manual and usage examples.[/dim]"
        )
        sys.exit(2)

    parser.error = custom_error  # type: ignore[assignment]
    parser.print_help = lambda file=None: print_custom_help()  # type: ignore[assignment]

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show comprehensive manual with detailed descriptions and usage examples, then exit",
    )
    parser.add_argument(
        "--mode",
        choices=["enterprise", "notebooklm", "mock"],
        help="Backend mode override (enterprise, notebooklm, mock)",
    )
    parser.add_argument("--project", help="GCP Project ID override")
    parser.add_argument("--mock", action="store_true", help="Shortcut for --mode mock")
    parser.add_argument(
        "--notebook",
        "-n",
        action="append",
        help="Open notebook(s) directly. Can be repeated (-n 'NB 1' -n 'NB 2') or comma-separated (-n 'NB 1, NB 2')",
    )
    parser.add_argument(
        "--source",
        "-s",
        help="Select a source directly by title substring or ID",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Directly ask a question non-interactively and exit",
    )
    parser.add_argument(
        "--list-sources",
        "--sources",
        dest="list_sources",
        action="store_true",
        help="List all sources for the specified notebook(s) and exit",
    )
    parser.add_argument(
        "--find",
        "-f",
        help="Search notebooks by semantic topic or theme using Gemini",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=3,
        help="Max notebooks to synthesize across when using --find with -q (default: 3)",
    )
    # CRUD Notebook Arguments
    parser.add_argument(
        "--create-notebook",
        "-c",
        metavar="TITLE",
        help="Create a new notebook with the specified title and exit",
    )
    parser.add_argument(
        "--delete-notebook",
        "-d",
        metavar="ID",
        help="Delete a notebook by ID and exit",
    )
    parser.add_argument(
        "--update-notebook",
        metavar="ID",
        help="Notebook ID to update (requires --title and/or --description)",
    )
    parser.add_argument(
        "--title",
        help="New title when updating or creating a notebook",
    )
    parser.add_argument(
        "--description",
        help="Description for updating or creating a notebook",
    )
    parser.add_argument(
        "--list-notebooks",
        "--list",
        dest="list_notebooks",
        action="store_true",
        help="List all notebooks and exit",
    )

    # CRUD Source Arguments
    parser.add_argument(
        "--add-source",
        metavar="FILE_OR_TEXT",
        help="Add a source to a notebook (requires -n/--notebook)",
    )
    parser.add_argument(
        "--source-title",
        help="Title for the source being added via --add-source",
    )
    parser.add_argument(
        "--delete-source",
        metavar="SOURCE_ID",
        help="Delete a source by ID from a notebook (requires -n/--notebook)",
    )
    args = parser.parse_args()

    if args.help:
        print_custom_help()
        return

    config = AppConfig.load()
    if args.mock:
        config.backend_mode = "mock"
    elif args.mode:
        config.backend_mode = args.mode  # type: ignore[assignment]
    if args.project:
        config.gcp_project_id = args.project

    print_banner(config)
    backend = get_backend(config)

    # Dispatch CRUD Notebook commands
    if args.create_notebook:
        with console.status(
            f"[bold cyan]Creating notebook '{args.create_notebook}'...[/bold cyan]"
        ):
            nb = await backend.create_notebook(
                title=args.create_notebook,
                description=args.description or "",
            )
        console.print(
            Panel(
                f"[bold green]Notebook Created Successfully![/bold green]\n\n"
                f"[bold]Notebook Name:[/bold] {nb.title}\n"
                f"[bold]Notebook ID:[/bold]   {nb.id}\n"
                f"[bold]Description:[/bold]   {nb.description or '[dim]None[/dim]'}",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        return

    if args.delete_notebook:
        target_input = args.delete_notebook
        target_id = target_input
        target_title = target_input
        try:
            all_nbs = await backend.list_notebooks()
            resolved = resolve_notebook_selection(target_input, all_nbs)
            if resolved:
                target_id = resolved[0].id
                target_title = resolved[0].title
        except Exception:
            pass

        with console.status(f"[bold cyan]Deleting notebook '{target_title}'...[/bold cyan]"):
            try:
                success = await backend.delete_notebook(target_id)
            except Exception as e:
                console.print(f"[bold red]Error deleting notebook:[/bold red] {e}")
                return

        if success:
            console.print(
                f"[bold green]Notebook '{target_title}' ({target_id}) deleted successfully.[/bold green]"
            )
        else:
            console.print(
                f"[bold red]Failed to delete notebook '{target_title}' (not found or error).[/bold red]"
            )
        return

    if args.update_notebook:
        if not args.title and not args.description:
            console.print(
                "[bold red]Error: --update-notebook requires --title and/or --description.[/bold red]"
            )
            return

        target_input = args.update_notebook
        target_id = target_input
        try:
            all_nbs = await backend.list_notebooks()
            resolved = resolve_notebook_selection(target_input, all_nbs)
            if resolved:
                target_id = resolved[0].id
        except Exception:
            pass

        with console.status(f"[bold cyan]Updating notebook '{target_id}'...[/bold cyan]"):
            try:
                nb = await backend.update_notebook(
                    notebook_id=target_id,
                    title=args.title,
                    description=args.description,
                )
            except Exception as e:
                console.print(f"[bold red]Error updating notebook:[/bold red] {e}")
                return

        console.print(
            Panel(
                f"[bold green]Notebook Updated Successfully![/bold green]\n\n"
                f"[bold]Notebook Name:[/bold] {nb.title}\n"
                f"[bold]Notebook ID:[/bold]   {nb.id}\n"
                f"[bold]Description:[/bold]   {nb.description or '[dim]None[/dim]'}",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        return

    # Dispatch CRUD Source commands
    if args.add_source:
        if not args.notebook:
            console.print(
                "[bold red]Error: Adding a source requires specifying the notebook via -n / --notebook.[/bold red]"
            )
            return

        with console.status(
            f"[bold cyan]Resolving notebook '{args.notebook[0]}' on {config.backend_mode.upper()} backend...[/bold cyan]"
        ):
            try:
                all_notebooks = await backend.list_notebooks()
            except Exception as e:
                console.print(f"[bold red]Error fetching notebooks:[/bold red] {e}")
                return

        resolved = resolve_notebook_selection(args.notebook[0], all_notebooks)
        if not resolved:
            console.print(
                f"[bold red]Error: Notebook '{args.notebook[0]}' could not be resolved. Run with --list to view available notebooks.[/bold red]"
            )
            return

        target_notebook = resolved[0]
        source_title = args.source_title

        raw_source = args.add_source
        is_path_like = raw_source.startswith((".", "/", "~")) or any(
            raw_source.lower().endswith(ext)
            for ext in (
                ".md",
                ".txt",
                ".json",
                ".csv",
                ".pdf",
                ".html",
                ".xml",
                ".yaml",
                ".yml",
                ".py",
                ".ts",
                ".js",
                ".cs",
            )
        )

        if is_path_like and not os.path.isfile(raw_source):
            suggest = None
            if raw_source.startswith("..") and not raw_source.startswith("../"):
                fixed = "../" + raw_source[2:]
                if os.path.isfile(fixed):
                    suggest = fixed
            err_msg = f"[bold red]Error: File not found:[/bold red] '{raw_source}'"
            if suggest:
                err_msg += f"\n[yellow]Did you mean '[bold cyan]{suggest}[/bold cyan]'?[/yellow]"
            console.print(err_msg)
            return

        if os.path.isfile(raw_source):
            try:
                with open(raw_source, encoding="utf-8") as f:
                    content = f.read()
                if not source_title:
                    source_title = os.path.basename(raw_source)
            except Exception as e:
                console.print(f"[bold red]Error reading file '{raw_source}':[/bold red] {e}")
                return
        else:
            content = raw_source

        if not source_title:
            source_title = "Uploaded Document"

        with console.status(
            f"[bold cyan]Adding source '{source_title}' to notebook '{target_notebook.title}'...[/bold cyan]"
        ):
            try:
                src = await backend.add_source(
                    notebook_id=target_notebook.id,
                    title=source_title,
                    content=content,
                )
            except Exception as e:
                console.print(
                    f"[bold red]Error adding source to notebook '{target_notebook.title}':[/bold red] {e}"
                )
                return

        console.print(
            Panel(
                f"[bold green]Source Added Successfully![/bold green]\n\n"
                f"[bold]Source Title:[/bold]  {src.title}\n"
                f"[bold]Source ID:[/bold]     {src.id}\n"
                f"[bold]Notebook Name:[/bold] {target_notebook.title}\n"
                f"[bold]Notebook ID:[/bold]   {target_notebook.id}",
                border_style="green",
                box=box.ROUNDED,
            )
        )
        return

    if args.delete_source:
        if not args.notebook:
            console.print(
                "[bold red]Error: Deleting a source requires specifying the notebook via -n / --notebook.[/bold red]"
            )
            return

        with console.status(
            f"[bold cyan]Resolving notebook '{args.notebook[0]}' on {config.backend_mode.upper()} backend...[/bold cyan]"
        ):
            try:
                all_notebooks = await backend.list_notebooks()
            except Exception as e:
                console.print(f"[bold red]Error fetching notebooks:[/bold red] {e}")
                return

        resolved = resolve_notebook_selection(args.notebook[0], all_notebooks)
        if not resolved:
            console.print(
                f"[bold red]Error: Notebook '{args.notebook[0]}' could not be resolved. Run with --list to view available notebooks.[/bold red]"
            )
            return

        target_notebook = resolved[0]
        with console.status(
            f"[bold cyan]Deleting source '{args.delete_source}' from notebook '{target_notebook.title}'...[/bold cyan]"
        ):
            try:
                success = await backend.delete_source(
                    notebook_id=target_notebook.id, source_id=args.delete_source
                )
            except Exception as e:
                console.print(f"[bold red]Error deleting source:[/bold red] {e}")
                return

        if success:
            console.print(
                f"[bold green]Source '{args.delete_source}' deleted successfully from notebook '{target_notebook.title}'.[/bold green]"
            )
        else:
            console.print(
                f"[bold red]Failed to delete source '{args.delete_source}' (not found or error).[/bold red]"
            )
        return

    if args.list_notebooks:
        with console.status(
            f"[bold cyan]Fetching notebooks from {config.backend_mode.upper()} backend...[/bold cyan]"
        ):
            all_notebooks = await backend.list_notebooks()
        display_notebooks_table(all_notebooks)
        return

    initial_notebook_arg = args.notebook
    initial_source_arg = args.source

    while True:
        # Step 1: Fetch and list notebooks
        notebooks: list[NotebookInfo] | None = None
        fetch_error: Exception | None = None

        with console.status(
            f"[bold cyan]Fetching notebooks from {config.backend_mode.upper()} backend...[/bold cyan]"
        ):
            try:
                notebooks = await backend.list_notebooks()
            except Exception as e:
                fetch_error = e

        if fetch_error is not None:
            if isinstance(fetch_error, EnterpriseAPIError):
                console.print(
                    f"\n[bold red]Google Cloud Enterprise API Notice:[/bold red]\n{fetch_error}\n"
                )
            else:
                msg = str(fetch_error)
                if "storage_state.json" in msg or "login" in msg:
                    console.print(
                        "\n[bold yellow]NotebookLM Authentication Required:[/bold yellow]\n"
                        "Your local session is not signed in yet. To connect to your real Gemini Pro notebooks, run:\n"
                        "  [bold cyan]uv run notebooklm login[/bold cyan]\n"
                    )
                else:
                    console.print(
                        f"\n[bold red]Failed to load notebooks:[/bold red] {fetch_error}\n"
                    )

            use_mock = Prompt.ask(
                "Would you like to switch to [bold cyan]MOCK mode[/bold cyan] for this session to test the interface?",
                choices=["y", "n"],
                default="y",
            )
            if use_mock == "y":
                config.backend_mode = "mock"
                backend = get_backend(config)
                with console.status("[bold cyan]Loading sample notebooks...[/bold cyan]"):
                    notebooks = await backend.list_notebooks()
            else:
                console.print("[dim]Exiting.[/dim]")
                return

        if not notebooks:
            console.print("[yellow]No notebooks found in this project/account.[/yellow]")
            return

        selected_nbs: list[NotebookInfo] = []
        if initial_notebook_arg:
            selected_nbs = resolve_notebook_selection(initial_notebook_arg, notebooks)
            initial_notebook_arg = None

        if args.find:
            with console.status(
                f"[bold cyan]Searching {len(notebooks)} notebooks for '{args.find}' with Gemini...[/bold cyan]"
            ):
                matches = await backend.find_notebooks(args.find, notebooks=notebooks)

            if not matches:
                console.print(f"\n[yellow]No notebooks matched the topic '{args.find}'.[/yellow]")
                return

            display_search_results(matches, args.find)

            # Option 2: Find and immediately ask question across top matches
            if args.question:
                top_matches = matches[: args.limit]
                target_ids = [m.notebook_id for m in top_matches]
                console.print(
                    f"\n[bold cyan]Synthesizing question across top {len(target_ids)} matched notebooks: "
                    + ", ".join(f"'{m.notebook_title}'" for m in top_matches)
                    + "...[/bold cyan]"
                )
                with console.status(
                    f"[bold cyan]Querying {len(target_ids)} notebooks in parallel and synthesizing...[/bold cyan]"
                ):
                    multi_ans = await backend.ask_notebooks(
                        notebook_ids=target_ids,
                        question=args.question,
                        notebook_titles={m.notebook_id: m.notebook_title for m in top_matches},
                    )
                display_multi_answer(multi_ans)
                return

            # Option 1: Find only - display results and exit cleanly
            return

        if not selected_nbs:
            display_notebooks_table(notebooks)

            # Step 2: Select a notebook or multiple notebooks
            console.print(
                "\n[dim]Select one or more notebooks (e.g. '1' or '1, 2'), or search by topic (e.g. 'find software architecture').[/dim]"
            )
            choice = Prompt.ask(
                "Enter notebook [bold green]#[/bold green] / [bold cyan]ID[/bold cyan] (e.g. '1', '1, 2', 'find <topic>', 'all', or 'q' to quit)",
                default="all" if args.list_sources else "1",
            ).strip()

            if choice.lower() in ("q", "quit", "exit"):
                console.print("[dim]Goodbye![/dim]")
                return

            if choice.lower().startswith("/find ") or choice.lower().startswith("find "):
                topic = choice.split(" ", 1)[1].strip()
                with console.status(
                    f"[bold cyan]Searching {len(notebooks)} notebooks for '{topic}' with Gemini...[/bold cyan]"
                ):
                    matches = await backend.find_notebooks(topic, notebooks=notebooks)
                if not matches:
                    console.print(f"\n[yellow]No notebooks matched the topic '{topic}'.[/yellow]")
                else:
                    display_search_results(matches, topic)
                continue

            if choice.lower() == "all":
                selected_nbs = list(notebooks)
            else:
                selected_nbs = resolve_notebook_selection(choice, notebooks)

            if not selected_nbs:
                console.print(
                    f"[red]Notebook selection '{choice}' could not be resolved. Try again.[/red]"
                )
                continue

        # Check if list sources was requested via CLI flag
        if args.list_sources:
            with console.status(
                f"[bold cyan]Fetching sources for {len(selected_nbs)} notebook(s)...[/bold cyan]"
            ):
                sources_map = await backend.list_multi_sources([nb.id for nb in selected_nbs])
            display_multi_sources(selected_nbs, sources_map)
            if not args.question:
                return

        # Check if Multi-Notebook Mode was chosen
        if len(selected_nbs) > 1:
            if args.question:
                with console.status(
                    f"[bold cyan]Querying {len(selected_nbs)} notebooks in parallel and synthesizing...[/bold cyan]"
                ):
                    multi_ans = await backend.ask_notebooks(
                        notebook_ids=[nb.id for nb in selected_nbs],
                        question=args.question,
                        notebook_titles={nb.id: nb.title for nb in selected_nbs},
                    )
                display_multi_answer(multi_ans)
                return

            await multi_chat_loop(backend, selected_nbs)
            continue

        selected_nb = selected_nbs[0]

        # Step 3: Sources loop for selected notebook
        while True:
            with console.status(
                f"[bold cyan]Loading sources for '{selected_nb.title}'...[/bold cyan]"
            ):
                try:
                    sources = await backend.list_sources(selected_nb.id)
                except Exception as e:
                    console.print(f"[red]Error fetching sources for notebook:[/red] {e}")
                    sources = []

            console.print()
            if sources:
                display_sources_table(sources, selected_nb.title)
            else:
                console.print(
                    f"[yellow]No individual sources found in notebook '{selected_nb.title}'.[/yellow]"
                )

            selected_src: SourceInfo | None = None
            if args.question:
                if initial_source_arg and initial_source_arg.lower() != "all":
                    match_src = initial_source_arg.lower()
                    for s in sources:
                        if (
                            match_src in s.title.lower()
                            or s.id.lower().startswith(match_src)
                            or s.id.lower() == match_src
                        ):
                            selected_src = s
                            break

                with console.status(
                    "[bold cyan]Querying notebook and grounding with source...[/bold cyan]"
                ):
                    answer_obj = await backend.ask_question(
                        notebook_id=selected_nb.id,
                        source_id=selected_src.id if selected_src else None,
                        question=args.question,
                    )

                console.print()
                console.print(
                    Panel(
                        Markdown(answer_obj.answer),
                        title=f"[bold cyan]Gemini Response ({answer_obj.model_name or 'Grounded'})[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    )
                )

                if answer_obj.citations:
                    cit_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
                    cit_table.add_column("Bullet", style="cyan", width=3)
                    cit_table.add_column("Citation", style="italic dim")
                    for c in answer_obj.citations:
                        cit_table.add_row("-", c)
                    console.print(
                        Panel(
                            cit_table,
                            title="[dim]Citations & Grounding Sources[/dim]",
                            border_style="dim",
                            box=box.ROUNDED,
                        )
                    )
                return

            if initial_source_arg:
                match_src = initial_source_arg.lower()
                if match_src != "all":
                    for s in sources:
                        if (
                            match_src in s.title.lower()
                            or s.id.lower().startswith(match_src)
                            or s.id.lower() == match_src
                        ):
                            selected_src = s
                            break
                initial_source_arg = None
            else:
                console.print("\n[dim]Choose which source you want to ask questions about.[/dim]")
                src_choice = Prompt.ask(
                    "Enter source [bold yellow]#[/bold yellow] / [bold cyan]ID[/bold cyan], '[bold green]all[/bold green]', or '[bold]b[/bold]' (back to notebooks)",
                    default="1" if sources else "all",
                ).strip()

                if src_choice.lower() in ("b", "back"):
                    break
                if src_choice.lower() in ("q", "quit", "exit"):
                    console.print("[dim]Goodbye![/dim]")
                    return

                if src_choice.lower() != "all" and sources:
                    if src_choice.isdigit():
                        s_idx = int(src_choice) - 1
                        if 0 <= s_idx < len(sources):
                            selected_src = sources[s_idx]
                    if not selected_src:
                        for s in sources:
                            if (
                                s.id == src_choice
                                or s.id.startswith(src_choice)
                                or src_choice.lower() in s.title.lower()
                            ):
                                selected_src = s
                                break

            # Step 4: Launch chat loop
            await chat_loop(backend, selected_nb, selected_src)
