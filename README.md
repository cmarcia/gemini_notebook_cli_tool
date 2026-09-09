# Gemini Notebook / NotebookLM Enterprise PoC

A production-oriented Python client and orchestration interface for Google Gemini Notebooks (formerly known as NotebookLM) and Google Cloud Discovery Engine.

This repository provides:
- An interactive Rich terminal console for conversational exploration of enterprise notebooks and grounded sources.
- Non-interactive command-line operations for headless Q&A, document ingestion, and notebook management.
- Multi-notebook scatter-gather querying with parallel retrieval and unified LLM synthesis.
- Semantic topic discovery with automated fallback to keyword search under offline or quota-constrained conditions.
- Switchable execution backends: Enterprise (Google Cloud Discovery Engine v1alpha), Mock (offline simulation with pre-seeded data), and NotebookLM (web/browser session authentication via `notebooklm-py`).

---

## Architecture Overview

```
                               +------------------------------------------------+
                               |   Interactive Rich Console CLI (cli.py)        |
                               +-----------------------+------------------------+
                                                       |
                       +-------------------------------+-------------------------------+
                       |                               |                               |
                       v                               v                               v
          +-------------------------+    +---------------------------+    +-------------------------+
          |   Enterprise Backend    |    |     Mock / Demo Backend   |    |   NotebookLM Backend    |
          | (Google Cloud Discovery |    |   (Deterministic offline  |    |  (notebooklm-py web /   |
          |    Engine v1alpha)      |    |      data store)          |    |  browser session auth)  |
          +------------+------------+    +-------------+-------------+    +-------------------------+
                       |                               |
                       +---------------+---------------+
                                       |
                                       v
                         +---------------------------+
                         |   Gemini Grounding API    |
                         |  (google-genai SDK via    |
                         |      GEMINI_API_KEY)      |
                         +---------------------------+
```

### Supported Operational Backends
- **Enterprise (`enterprise`)**: Connects to the Google Cloud Discovery Engine API (`discoveryengine.googleapis.com/v1alpha/projects/{project}/locations/{location}`). Manages enterprise data stores and grounds answers on ingested enterprise documents.
- **Mock (`mock`)**: Fully offline testing and simulation backend. Loads realistic corporate notebooks (financial audits, cloud architecture specifications, AI governance policies) without network or GCP dependencies.
- **NotebookLM (`notebooklm`)**: Uses the `notebooklm-py` client library to interface with personal or Google Workspace NotebookLM accounts through authenticated browser sessions.

---

## Prerequisites and System Requirements

Before setting up the project, ensure your environment meets the following specifications:

- **Operating System**: macOS (macOS 12 Monterey or later) or Windows (Windows 10, Windows 11, or Windows Server 2019/2022).
- **Python**: Version `3.13` or higher.
- **Package and Environment Manager**: `uv` version `0.5.0` or higher.
- **Google Cloud SDK (`gcloud`)** *(Optional, required for Enterprise mode)*: For generating Application Default Credentials (ADC) or bearer access tokens.

---

## Installing uv

`uv` is an extremely fast Python package and environment manager built in Rust. It replaces `pip`, `pip-tools`, `virtualenv`, and `poetry` workflows.

### Installing uv on macOS

#### Option 1: Via Homebrew (Recommended)
```bash
brew install uv
```

#### Option 2: Via the Official Standalone Shell Script
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
After installation, verify that `uv` is available in your shell:
```bash
uv --version
```

### Installing uv on Windows

#### Option 1: Via PowerShell Installer (Recommended)
Open PowerShell (as a standard user or administrator) and execute:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Option 2: Via Windows Package Manager (`winget`)
```cmd
winget install --id=astral-sh.uv -e
```

#### Option 3: Via Scoop
```powershell
scoop install uv
```

After installation, restart your terminal or refresh environment variables, then verify:
```powershell
uv --version
```

---

## Project Setup and Virtual Environment Management

This project uses `uv` native project workspaces defined in `pyproject.toml`.

### 1. Clone the Repository

#### macOS / Linux
```bash
git clone <repository-url>
cd gemini-notebook-poc
```

#### Windows (PowerShell or Command Prompt)
```powershell
git clone <repository-url>
cd gemini-notebook-poc
```

---

### 2. Managing the Virtual Environment with uv

`uv` provides two ways to manage environments:
1. **Implicit / Transient Execution**: When using `uv run <command>`, `uv` automatically creates a virtual environment inside `.venv` if one does not exist and ensures all dependencies match `uv.lock`. You never need to manually create or activate an environment if you use `uv run`.
2. **Explicit Virtual Environment**: You can explicitly create and activate `.venv` for IDE integration (such as VS Code, PyCharm, or Antigravity) or traditional shell workflows.

#### Creating the Virtual Environment

##### macOS
```bash
# Create a virtual environment targeting Python 3.13
uv venv --python 3.13
```

##### Windows (PowerShell or Command Prompt)
```powershell
# Create a virtual environment targeting Python 3.13
uv venv --python 3.13
```

`uv` will locate an existing Python 3.13 installation on your system. If Python 3.13 is not installed, `uv` will automatically download and install an isolated Python 3.13 binary for you.

---

#### Activating the Virtual Environment

##### macOS (Zsh / Bash)
```bash
source .venv/bin/activate
```
To deactivate:
```bash
deactivate
```

##### Windows (PowerShell)
```powershell
# If script execution is restricted, enable RemoteSigned for the current session:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the environment:
.venv\Scripts\Activate.ps1
```
To deactivate:
```powershell
deactivate
```

##### Windows (Command Prompt `cmd.exe`)
```cmd
.venv\Scripts\activate.bat
```
To deactivate:
```cmd
deactivate
```

---

### 3. Synchronizing Dependencies with uv

Package dependencies are specified in `pyproject.toml` and pinned deterministically in `uv.lock`. The `uv sync` command reconciles the virtual environment with the lockfile, installing missing packages and removing undeclared ones.

#### Production Dependencies Only

##### macOS
```bash
uv sync --no-dev
```

##### Windows (PowerShell or Command Prompt)
```powershell
uv sync --no-dev
```

#### Complete Dependencies (Production + Development Tools)

To include development tools (`pytest`, `pytest-asyncio`, `mypy`, `ruff`):

##### macOS
```bash
uv sync --all-groups
```

##### Windows (PowerShell or Command Prompt)
```powershell
uv sync --all-groups
```

#### Upgrading or Updating Dependencies

To upgrade all locked packages to their latest compatible versions and regenerate `uv.lock`:

##### macOS / Windows
```bash
uv lock --upgrade
uv sync --all-groups
```

---

## Configuration and Environment Variables

Configuration is loaded from standard environment variables or a local `.env` file via `gemini_notebook_poc.config.AppConfig`.

### 1. Initialize the Configuration File

#### macOS
```bash
cp .env.example .env
```

#### Windows (PowerShell)
```powershell
Copy-Item .env.example .env
```

#### Windows (Command Prompt)
```cmd
copy .env.example .env
```

### 2. Configuration Parameters

Open `.env` in a text editor and adjust the settings:

```env
# Operational backend: 'enterprise', 'mock', or 'notebooklm'
BACKEND_MODE=enterprise

# Google Cloud Project ID with Discovery Engine enabled
GCP_PROJECT_ID=my-gcp-project-id

# Google Cloud Location (e.g., 'global', 'us', 'eu')
GCP_LOCATION=global

# Optional explicit bearer token (if not using gcloud CLI or service account keys)
# GCP_ACCESS_TOKEN=ya29.your-access-token

# Optional service account key path
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Gemini API Key (required for live document synthesis and semantic search)
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini Model identifier
GEMINI_MODEL=gemini-3.6-flash

# Optional NotebookLM consumer session token (if using BACKEND_MODE=notebooklm)
# NOTEBOOKLM_AUTH_TOKEN=
```

---

## How to Run the Application

You can execute the application in two ways:
1. **Via `uv run` (Recommended)**: Executes commands within the project context without needing to manually activate the virtual environment.
2. **Via Activated Virtual Environment**: Using the installed script entry point directly (`gemini-notebook-poc`).

### 1. Interactive Console Mode

Launches the full interactive terminal application with navigable menus and chat sessions:

#### macOS
```bash
# Using uv run (auto-wires environment)
uv run gemini-notebook-poc

# In Mock mode (offline demonstration without GCP credentials)
uv run gemini-notebook-poc --mock
```

#### Windows (PowerShell or Command Prompt)
```powershell
# Using uv run
uv run gemini-notebook-poc

# In Mock mode
uv run gemini-notebook-poc --mock
```

---

### 2. Command-Line Reference and Arguments Manual

To display the built-in reference manual directly in your terminal:

#### macOS
```bash
uv run gemini-notebook-poc --help
```

#### Windows
```powershell
uv run gemini-notebook-poc --help
```

---

## Complete CLI Arguments Reference

The CLI supports flags for backend selection, non-interactive queries, semantic discovery, and full CRUD operations on notebooks and source documents.

### Summary Table

| Category | Flag / Option | Arguments | Description |
| :--- | :--- | :--- | :--- |
| **Backend** | `--mode` | `enterprise \| mock \| notebooklm` | Explicitly sets the operational backend engine. |
| | `--mock` | *None* | Shortcut for `--mode mock`. Starts offline simulation mode. |
| | `--project` | `<PROJECT_ID>` | Overrides the Google Cloud Project ID for Discovery Engine. |
| **Interactive & Q&A** | `-n`, `--notebook` | `<NAME \| ID \| INDEX \| all>` | Selects one or more notebooks. Can be repeated or comma-separated. |
| | `-s`, `--source` | `<NAME \| ID \| INDEX \| all>` | Filters grounding to a specific source within a notebook. |
| | `-q`, `--question` | `"<TEXT>"` | Headless Q&A. Queries source material and outputs the answer directly. |
| **Discovery** | `-f`, `--find` | `"<TOPIC>"` | Performs semantic topic search across the notebook catalog. |
| | `-l`, `--limit` | `<INT>` | Maximum number of notebooks to query when combining `-f` with `-q` (default: 3). |
| **Notebook CRUD** | `--list`, `--list-notebooks` | *None* | Displays all available notebooks in a formatted table and exits. |
| | `-c`, `--create-notebook` | `"<TITLE>"` | Creates a new notebook with the given title. |
| | `--update-notebook` | `<ID>` | Updates an existing notebook by ID (requires `--title` or `--description`). |
| | `--title` | `"<TITLE>"` | New title used when creating or updating a notebook. |
| | `--description` | `"<TEXT>"` | Description used when creating or updating a notebook. |
| | `-d`, `--delete-notebook` | `<ID>` | Permanently deletes a notebook by ID. |
| **Source CRUD** | `--sources`, `--list-sources` | *None* | Displays all ingested sources in the specified notebook(s). |
| | `--add-source` | `<FILE_PATH \| TEXT>` | Ingests a document from a local file or inline text string. |
| | `--source-title` | `"<TITLE>"` | Custom title for the document added via `--add-source`. |
| | `--delete-source` | `<SOURCE_ID>` | Deletes a specific source document from a notebook. |
| **General** | `-h`, `--help` | *None* | Prints the comprehensive user manual and exits. |

---

### Detailed Argument Descriptions and Examples

#### 1. Backend Configuration Flags

##### `--mode <MODE>`
Overrides the `BACKEND_MODE` variable set in `.env`. Allowed values:
- `enterprise`: Google Cloud Discovery Engine v1alpha API.
- `mock`: In-memory deterministic simulation.
- `notebooklm`: Browser-authenticated NotebookLM web backend.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mode enterprise --list
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mode mock --list
```

##### `--mock`
A convenience flag that instantly forces `--mode mock`. Does not require network access, GCP credentials, or Gemini API keys.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock
```

##### `--project <PROJECT_ID>`
Overrides the `GCP_PROJECT_ID` configuration setting. Applies only when operating under `enterprise` mode.

*macOS Example:*
```bash
uv run gemini-notebook-poc --project my-production-project-42 --list
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --project my-production-project-42 --list
```

---

#### 2. Querying and Grounded Q&A Flags

##### `-n, --notebook <NAME | ID | INDEX | all>`
Directly selects one or multiple target notebooks. Supports:
- Single notebook by title substring: `-n "Financial"`
- Single notebook by exact ID: `-n "nb-enterprise-finance-q3"`
- Single notebook by 1-based index from `--list`: `-n 1`
- Multiple notebooks via repeated flag: `-n "Financial" -n "Cloud Infrastructure"`
- Multiple notebooks via comma-separated string: `-n "Financial, Cloud Infrastructure"`
- All available notebooks: `-n all`

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -n "Financial" -q "What was the Q3 net income?"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -n "Financial" -q "What was the Q3 net income?"
```

##### `-s, --source <NAME | ID | INDEX | all>`
Limits grounding to an individual document source within the notebook selected via `-n`.
- By title substring: `-s "10-Q"`
- By source ID: `-s "src-10q-sec-filing"`
- By 1-based index from `--sources`: `-s 1`
- All sources: `-s all`

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -n "Financial" -s "10-Q" -q "What are the primary expense drivers?"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -n "Financial" -s "10-Q" -q "What are the primary expense drivers?"
```

##### `-q, --question "<TEXT>"`
Executes a non-interactive query against the target notebook(s) or source, outputs the grounded answer with citations, and exits.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -n "Cloud Infrastructure" -q "What is our database SLA target?"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -n "Cloud Infrastructure" -q "What is our database SLA target?"
```

---

#### 3. Semantic Topic Search and Cross-Notebook Synthesis

##### `-f, --find "<TOPIC>"`
Searches across the entire notebook catalog using Gemini semantic understanding. Matches conceptual topics even when literal keywords differ.

*macOS Example (Search Only):*
```bash
uv run gemini-notebook-poc --mock -f "cloud migration and infrastructure costs"
```

*Windows Example (Search Only):*
```powershell
uv run gemini-notebook-poc --mock -f "cloud migration and infrastructure costs"
```

##### Combining `-f` with `-q` and `-l`
When `-f` is combined with `-q`, the CLI identifies the top matching notebooks, retrieves grounded answers from each in parallel, and synthesizes a unified answer across them.

##### `-l, --limit <NUMBER>`
Controls how many matching notebooks are queried during automated search and synthesis (default: 3).

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -f "infrastructure" -q "What are the reliability risks?" -l 2
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -f "infrastructure" -q "What are the reliability risks?" -l 2
```

---

#### 4. Notebook Lifecycle Management (CRUD)

##### `--list-notebooks, --list`
Displays a table of all notebooks containing index, title, ID, source count, and last update timestamp.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock --list
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock --list
```

##### `-c, --create-notebook "<TITLE>"`
Creates a new notebook. Can be accompanied by `--description`.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -c "2027 Engineering Strategy" --description "Roadmap and architectural milestones"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -c "2027 Engineering Strategy" --description "Roadmap and architectural milestones"
```

##### `--update-notebook <ID>`
Updates the title or description of an existing notebook. Requires `--title` and/or `--description`.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock --update-notebook "nb-cloud-architecture-v2" --title "Cloud Architecture v3" --description "Updated multi-region plan"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock --update-notebook "nb-cloud-architecture-v2" --title "Cloud Architecture v3" --description "Updated multi-region plan"
```

##### `-d, --delete-notebook <ID>`
Deletes the specified notebook and all associated documents.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -d "nb-enterprise-finance-q3"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -d "nb-enterprise-finance-q3"
```

---

#### 5. Source Document Management (CRUD)

##### `--list-sources, --sources`
Lists all documents ingested in the specified notebook(s). Requires `-n/--notebook`.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -n "Financial" --sources
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -n "Financial" --sources
```

##### `--add-source <FILE_PATH | TEXT>`
Ingests a source document into the notebook specified by `-n`. Accepts a local file path (`.md`, `.txt`, `.json`, `.csv`, `.pdf`) or an inline raw text string.

*macOS File Path Example:*
```bash
uv run gemini-notebook-poc --mock -n "Financial" --add-source ./notes/audit.txt --source-title "Internal Audit Notes"
```

*Windows File Path Example (PowerShell):*
```powershell
uv run gemini-notebook-poc --mock -n "Financial" --add-source .\notes\audit.txt --source-title "Internal Audit Notes"
```

*Inline Text Example (Cross-Platform):*
```bash
uv run gemini-notebook-poc --mock -n "Financial" --add-source "Q4 projected growth stands at 14% based on early pipeline velocity." --source-title "Q4 Forecast Memo"
```

##### `--delete-source <SOURCE_ID>`
Deletes a source document from a notebook. Requires `-n/--notebook`.

*macOS Example:*
```bash
uv run gemini-notebook-poc --mock -n "Financial" --delete-source "src-cfo-call-transcript"
```

*Windows Example:*
```powershell
uv run gemini-notebook-poc --mock -n "Financial" --delete-source "src-cfo-call-transcript"
```

---

## Interactive Console Navigation Commands

When running the interactive chat console (`uv run gemini-notebook-poc` or `uv run gemini-notebook-poc --mock`), you can enter specialized slash commands at the prompt:

| Command | Action |
| :--- | :--- |
| `/sources` | Displays a formatted table of all sources available in the current notebook. |
| `/back` | Returns to the source selection menu (or notebook selection menu). |
| `/notebook` | Exits the active session and returns directly to notebook selection. |
| `/clear` | Clears conversation context history for the current session. |
| `/exit` or `/quit` | Exits the console application. |

---

## Portable Standalone Orchestrator

The file `src/gemini_notebook_poc/orchestrator.py` contains a self-contained orchestrator that can be executed directly with standard Python:

### macOS
```bash
# Semantic search
python src/gemini_notebook_poc/orchestrator.py --find "architecture"

# Cross-notebook query
python src/gemini_notebook_poc/orchestrator.py -n "nb-cloud-architecture-v2" -q "Explain Spanner config"

# List sources
python src/gemini_notebook_poc/orchestrator.py -n "nb-cloud-architecture-v2" --sources
```

### Windows (PowerShell or Command Prompt)
```powershell
# Semantic search
python src\gemini_notebook_poc\orchestrator.py --find "architecture"

# Cross-notebook query
python src\gemini_notebook_poc\orchestrator.py -n "nb-cloud-architecture-v2" -q "Explain Spanner config"

# List sources
python src\gemini_notebook_poc\orchestrator.py -n "nb-cloud-architecture-v2" --sources
```

---

## Google Cloud Enterprise Setup and IAM

To connect this application to the live **Google Cloud Discovery Engine** API:

1. **Enable the Discovery Engine API** in your GCP project:
   ```bash
   gcloud services enable discoveryengine.googleapis.com --project=YOUR_PROJECT_ID
   ```

2. **Assign Appropriate IAM Roles**:
   - `roles/discoveryengine.viewer` (Read-only access to search engines and data stores)
   - `roles/discoveryengine.admin` (Full administrative CRUD over data stores and schemas)
   - `roles/aiplatform.user` (If utilizing Vertex AI endpoints)

3. **Authenticate**:
   - **Local Developer Login**:
     ```bash
     gcloud auth application-default login
     ```
   - **Access Token Export**:
     ```bash
     # macOS / Linux
     export GCP_ACCESS_TOKEN=$(gcloud auth print-access-token)

     # Windows (PowerShell)
     $env:GCP_ACCESS_TOKEN = (gcloud auth print-access-token)
     ```
   - **Service Account Key**:
     Place the service account JSON key on your system and set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` in `.env`.

---

## Testing and Code Quality

The project includes an automated test suite and strict static analysis configurations.

### Running Tests

#### macOS
```bash
uv run pytest
```

#### Windows (PowerShell or Command Prompt)
```powershell
uv run pytest
```

To run tests with detailed verbosity:
```bash
uv run pytest -v
```

### Running Static Type Checking (mypy)

#### macOS
```bash
uv run mypy src
```

#### Windows (PowerShell or Command Prompt)
```powershell
uv run mypy src
```

### Running Code Quality and Linter Checks (ruff)

#### macOS
```bash
uv run ruff check
```

#### Windows (PowerShell or Command Prompt)
```powershell
uv run ruff check
```

To automatically apply formatting and lint fixes:
```bash
uv run ruff format
uv run ruff check --fix
```

---

## Troubleshooting and Platform-Specific Notes

### Windows Execution Policy Restriction
If PowerShell reports:
```text
File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system.
```
Run the following in PowerShell before activating:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Windows Path Separators
When supplying file paths to `--add-source`, you may use either forward slashes (`./path/to/file.txt`) or Windows backslashes (`.\path\to\file.txt`). Both are normalized automatically by `pathlib.Path`.

### Line Endings (CRLF vs LF)
Git on Windows may check out files with CRLF line endings. The `.env` parser and file readers handle both CRLF (`\r\n`) and LF (`\n`) transparently.

### Rate Limiting and Quota Fallback
If the Gemini API returns HTTP 429 (`RESOURCE_EXHAUSTED`), the internal LLM service will automatically perform exponential backoff retries. If the quota remains exhausted, the semantic topic search mechanism seamlessly falls back to local keyword matching across notebook titles and metadata.
