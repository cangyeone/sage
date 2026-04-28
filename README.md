<p align="center">
  <img src="logo.png" alt="SeismicX logo" width="180"/>
</p>

<h1 align="center">SAGE — Seismology AI-Guided Engine</h1>

<p align="center">
  Conversational AI Platform for Seismology Research
</p>



<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-Flask-lightgrey" alt="Flask"/>
  <img src="https://img.shields.io/badge/LLM-Ollama%20%7C%20OpenAI%20Compatible-green" alt="LLM"/>
  <img src="https://img.shields.io/badge/RAG-BGE--M3%20%2B%20FAISS-orange" alt="RAG"/>
  <img src="https://img.shields.io/badge/License-GPLv3-blue" alt="License"/>
</p>

---

SAGE is an earthquake science AI platform integrating **natural language interaction**, **intelligent phase picking**, **statistical analysis**, **code generation and execution**, **GMT map drawing**, and **literature interpretation**. Users can drive complete analysis workflows through bilingual conversations without memorizing command-line parameters or writing boilerplate code.


**This project is actively evolving. Contributions via patches are welcome, and users are encouraged to stay up to date with ongoing repository updates.**
---

## Table of Contents

- [Features Overview](#features-overview)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Basic Installation](#basic-installation)
  - [pnsn Phase Picking Module](#pnsn-phase-picking-module-installation)
  - [RAG Dependencies](#rag-dependencies)
- [Configuring LLM Backend](#configuring-llm-backend)
- [Web Interface](#web-interface)
- [Command Line Tools](#command-line-tools)
- [Conversation Routing Mechanism](#conversation-routing-mechanism)
- [seismo_skill Skill System](#seismo_skill-skill-system)
- [seismo_script Workflow System](#seismo_script-workflow-system)
- [GMT Map Drawing](#gmt-map-drawing)
- [EvidenceDrivenGeoAgent — Geoscience Interpretation Agent](#evidencedrivengeoagent--geoscience-interpretation-agent)
  - [Design Principles](#design-principles)
  - [Architecture](#architecture)
  - [Agent Reasoning Loop](#agent-reasoning-loop)
  - [Evidence Record Schema](#evidence-record-schema)
  - [Nine Built-in Tools](#nine-built-in-tools)
  - [Data Source Priority](#data-source-priority)
  - [Convergence Conditions](#convergence-conditions)
  - [Web UI](#geo-agent-web-ui)
  - [File Upload for Research Data](#file-upload-for-research-data)
  - [Inline Web Literature Search](#inline-web-literature-search)
  - [CLI Command](#geo-agent-cli-command)
  - [Flask API](#geo-agent-flask-api)
  - [Python Programmatic Usage](#python-programmatic-usage)
  - [Output Schema](#output-schema)
- [Core Modules Details](#core-modules-details)
- [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
- [FAQ](#faq)

---

## Features Overview

| Module | Function Description |
|------|---------|
| 💬 **Intelligent Conversation Routing** | LLM automatically identifies intent (knowledge Q&A / code execution / chatting), no manual mode switching required |
| 🔍 **Phase Picking** | Single station online picking / batch directory picking, supporting various deep learning models in JIT and ONNX formats |
| 🔗 **Event Association** | Multiple methods including FastLink / REAL / Gamma, automatically associating station picking results into earthquake events |
| 🧭 **Polarity Analysis** | Automatic determination of P-wave first motion polarity |
| 📊 **Seismic Statistics** | b-value estimation (MLE/LSQ), F-M distribution plots, temporal and spatial distribution analysis |
| 🧑‍💻 **Code Generation and Execution** | LLM generates Python code + sandbox secure execution + built-in seismology toolkit, connecting multiple skill steps |
| 🗺️ **GMT Map Drawing** | Calls GMT6 to draw epicenter maps, station maps, topographic maps, focal mechanism diagrams, with downloadable images and scripts |
| 🤖 **Autonomous Agent** | Reads papers → understands methods → autonomous planning → progressive programming implementation, with automatic retries at each step |
| 📚 **Knowledge Base RAG** | BGE-M3 vectorization + FAISS retrieval, persistent storage, batch PDF ingestion and literature Q&A |
| 📖 **Literature Interpretation** | Temporary PDF upload → deep interpretation of methods/formulas/conclusions, multi-round questioning |
| 🗂 **Local File Access** | After authorizing specified directory, LLM can directly read file lists to assist analysis |
| ⚡ **Skill System** | Markdown format skill documents (7 built-in + unlimited custom), automatically retrieved and injected during conversation and code generation |
| 🔄 **Workflow System** | Declarative multi-step analysis pipelines (`.md` + YAML frontmatter); agent dispatches workflows to Code Engine step-by-step with shared execution directory and per-step debug loop |
| 📈 **Waveform Visualization** | Waveform diagrams embedded in conversation window (with phase annotation overlay), images can be clicked to enlarge or download |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Web UI (Flask + JS)                          │
│   /chat  ·  /knowledge  ·  /skills  ·  /llm-settings            │
└──────────────┬──────────────────────────────────────────────────┘
               │ HTTP REST API
┌──────────────▼──────────────────────────────────────────────────┐
│   /api/chat/route (LLM Intent Router)  │  /api/chat/workflow     │
│      code ──────┬──── qa ──── chat     │   (workflow endpoint)   │
└────────┬────────┼────────┬─────────────┴──────────┬─────────────┘
         │        │        │                         │
  ┌──────▼─────┐ ┌▼──────┐ ┌▼──────────┐    ┌───────▼──────────┐
  │ CodeEngine │ │RAG Q&A│ │General    │    │ CodeEngine       │
  │ + Toolkit  │ │BGE-M3 │ │Chat       │    │ .run_workflow()  │
  │ + GMT      │ │+FAISS │ │           │    └───────┬──────────┘
  └──────┬─────┘ └───────┘ └───────────┘            │
         │                                  ┌────────▼──────────────────────┐
         │                                  │   seismo_script Workflow      │
         │                                  │   Runner + Step DAG Executor  │
         │                                  │   builtin + ~/.seismicx/      │
         │                                  │   workflows/                  │
         │                                  └────────┬──────────────────────┘
         └─────────────────┬───────────────────────── ┘
                           │
  ┌────────────────────────▼──────────────────────────────────────┐
  │            seismo_skill Skill Retrieval                        │
  │    7 Built-in Skills  +  User Custom Skills                    │
  │    (~/.seismicx/skills/)                                       │
  └────────────────────────┬──────────────────────────────────────┘
                           │ Automatic injection of function descriptions + code examples
  ┌────────────────────────▼──────────────────────────────────────┐
  │            LLM Backend                                         │
  │   Ollama (local)  ·  vLLM  ·  OpenAI Compatible               │
  └───────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────┐
  │           pnsn/ Phase Picking Engine        │
  │    PhaseNet / EQTransformer / JIT / ONNX    │
  │    FastLink / Gamma Event Association       │
  └─────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the main repository
git clone https://github.com/yourname/sage.git
cd sage

# 2. Clone the pnsn phase picking module (must be placed under sage/ directory)
git clone https://github.com/cangyeone/pnsn.git

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama and pull model (choose one)
ollama serve &
ollama pull qwen3:8b          # Lightweight, ~6 GB

# 5. Start Web service
python web_app/app.py --port 5010

# 6. Access via browser
open http://localhost:5010
```

On first access, select and save the pulled model on the **LLM Settings page** to start using all features.

---

## Installation

### System Requirements

| Resource | Minimum Requirements | Recommended Configuration |
|------|---------|---------|
| **Operating System** | macOS / Linux / Windows | macOS 13+ / Ubuntu 22.04+ |
| **Python** | 3.9 | 3.10 / 3.11 |
| **Memory (RAM)** | 8 GB | 16 GB+ (for running local LLM) |
| **Storage Space** | 5 GB | 30 GB+ (models + knowledge base) |
| **GPU** | Optional | CUDA 11.8+ or Apple Metal (for accelerated inference) |

### Basic Installation

```bash
git clone https://github.com/yourname/sage.git
cd sage

# Complete installation (recommended)
pip install -r requirements.txt

# Or install parts on demand
pip install flask flask-cors                          # Web services
pip install obspy torch scipy numpy pandas            # Seismic data processing
pip install matplotlib plotly                         # Visualization
pip install FlagEmbedding faiss-cpu pdfminer.six PyMuPDF  # RAG Knowledge Base
```

### pnsn Phase Picking Module Installation

pnsn is a deep learning model library specifically for phase picking, developed by [cangyeone](https://github.com/cangyeone). **Must clone it to the `sage/` main directory**, SAGE calls it through relative paths.

```bash
# Execute in sage/ directory
git clone https://github.com/cangyeone/pnsn.git

# Install pnsn dependencies
cd pnsn
pip install -r requirements.txt
cd ..
```

**Directory Structure Confirmation:**

```
sage/
├── pnsn/               ← Must be in this location
│   ├── sage_picker.py
│   ├── fastlinker.py
│   ├── gammalink.py
│   ├── pickers/        ← JIT / ONNX model files
│   └── config/
├── web_app/
└── ...
```

Main models provided by pnsn:

| Model | Purpose | Format |
|------|------|------|
| **PhaseNet** | P/S wave arrival picking | JIT / ONNX |
| **EQTransformer** | Event detection + phase picking integration | JIT / ONNX |
| **JMA Picker** | Picker based on JMA algorithm | JIT |

### RAG Dependencies

Knowledge base RAG functionality requires the `tokenizers` library, which on some systems requires Rust compilation environment:

```bash
# Install Rust (only needed when pip install reports compilation errors)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Reinstall embedding models library
pip install FlagEmbedding sentence-transformers

# On first use, BGE-M3 model (~2 GB) will automatically download from HuggingFace
# Domestic network can set mirror:
export HF_ENDPOINT=https://hf-mirror.com
```

#### Alternative: Download BGE-M3 via ModelScope (recommended for users in China)

If HuggingFace is inaccessible, use ModelScope to download the model locally first:

```bash
pip install modelscope

python -c "
from modelscope import snapshot_download
snapshot_download('AI-ModelScope/bge-m3', local_dir='open_models/bge-m3')
"
```

Then configure the local path in SAGE so it uses the downloaded model instead of downloading from the internet. There are two ways:

**Option 1 — Web Interface** (Recommended):
Open the **Knowledge Base page** (`/knowledge`) → click the ⚙ gear icon next to "Embedding Model" → paste the absolute path (e.g. `/Users/yourname/open_models/bge-m3`) → click Save.

**Option 2 — Edit config directly**:
Add an `embedding` section to `~/.seismicx/config.json`:

```json
{
  "llm": { "...": "..." },
  "embedding": {
    "model_path": "/Users/yourname/open_models/bge-m3"
  }
}
```

Leave `model_path` as an empty string or omit the field entirely to revert to HuggingFace auto-download. The setting takes effect on the next document build — no restart required.

---

## Configuring LLM Backend

All AI functions require an LLM backend. Configuration is done through **Web Interface → LLM Settings Page**, or via command line, uniformly stored in `~/.seismicx/config.json`. Changes **take effect immediately without restart**.

### Method 1: Ollama (Recommended, local, no internet required)

```bash
# 1. Install Ollama
# macOS / Linux:
curl -fsSL https://ollama.ai/install.sh | sh
# Or visit https://ollama.ai/download

# 2. Start service
ollama serve

# 3. Pull model (select based on VRAM / Memory)
ollama pull qwen3:8b         # ~6 GB, suitable for daily use
ollama pull qwen3:30b        # ~20 GB, comprehensive capabilities
ollama pull deepseek-r1:8b   # ~9 GB, strong reasoning capability
ollama pull llama3.3:latest  # ~40 GB, strong English capability
```

Select model on LLM settings page and click "Save Configuration" to complete setup.

### Method 2: Online API (OpenAI Compatible Format)

On LLM settings page → Select "Custom API" and fill in:

| Field | Example (DeepSeek) | Example (SiliconFlow) |
|------|----------------|-------------------|
| **API Base URL** | `https://api.deepseek.com/v1` | `https://api.siliconflow.cn/v1` |
| **API Key** | `sk-xxxxxxxx` | `sk-xxxxxxxx` |
| **Model Name** | `deepseek-chat` | `Qwen/Qwen2.5-72B-Instruct` |

Supports any OpenAI compatible interface, including DeepSeek, SiliconFlow, Moonshot (Moonshot), Alibaba Tongyi (DashScope), Zhipu GLM, Anthropic, etc.

### Method 3: Command Line Configuration

```bash
# Ollama local model
python seismic_cli.py backend use ollama --model qwen3:30b

# Online API
python seismic_cli.py backend use online \
    --provider deepseek \
    --api-key sk-xxx \
    --model deepseek-chat

# View all backend status
python seismic_cli.py backend status

# Auto-detect available backends
python seismic_cli.py backend auto
```

---

## Web Interface

After startup, visit `http://localhost:5010`, containing four main pages.

### 🗨 Conversation Page (/chat)

Main interaction interface. **No mode switching required** — system automatically determines intent of each message through LLM and routes to the most appropriate processor:

| Content Sent | Automatically Routes To |
|-----------|----------|
| "What is the Q-filter algorithm?" | Knowledge Q&A (RAG retrieval) |
| "Help me do 1-10 Hz bandpass filtering on /data/wave.mseed and plot" | Code generation and execution |
| "Help me draw a Chinese topographic map with GMT" | GMT skill execution |
| "Hello" | General conversation |

**Sidebar:**
- 📎 Upload PDF (temporary session use)
- 🗂 Authorize local working directory (LLM can read file lists from specified path)
- Knowledge base document count / snippet count status display

**Image Display and Download:**
- Images generated by code execution are directly embedded in conversation bubbles
- Toolbar displayed below each image: **⬇ Image** downloads PNG, **⬇ GMT Script** downloads reproducible `.sh` script (for GMT images only)
- Click image to view full screen in new window

**Typical conversation examples:**

```
# Knowledge Q&A (automatic knowledge base retrieval)
> What is the Q-filter algorithm?
> Explain the principle of HVSR spectrum ratio method

# Data processing (automatic code execution)
> Check files in directory /data/seismic/waveform
> Draw the waveforms
> Filter the waveforms with 1-10 Hz bandpass and plot
> Calculate power spectral density of vertical component

# GMT maps
> Help me draw a Chinese topographic map with GMT
> Draw epicenter distribution map for 90-120°E, 20-45°N

# Literature interpretation
> What are the core methods of this paper? (ask after uploading PDF)
```

### 📚 Knowledge Base Page (/knowledge)

- Drag-and-drop upload multiple PDFs, automatically vectorized with **BGE-M3** and ingested
- Real-time indexing progress display (text extraction → chunking → embedding → FAISS write)
- Document management: view page count/snippet count/file size, support single deletion or bulk clearing
- **Persistent storage**: Knowledge base automatically loads after service restart, no re-upload required

> Storage path: `~/.seismicx/knowledge/`

### ⚡ Skill Management Page (/skills)

Expand AI capabilities without restart. The page has two tabs: **Skills** and **Workflows**.

**Skills tab:**
- Left: Group display of built-in skills (read-only) and user custom skills (editable/deletable)
- Right: Markdown editor + real-time preview, with syntax highlighting
- Support creating, editing, deleting custom skills
- Takes effect immediately for next conversation or code generation after saving

> Custom skill storage path: `~/.seismicx/skills/`

**Workflows tab:**
- List of built-in and user-defined workflows with title, version, and skill dependency badges
- Step DAG preview panel: visualizes dependency graph between workflow steps
- Markdown editor for `.md` workflow files (YAML frontmatter + guide body)
- Support creating, editing, deleting custom workflows

> Custom workflow storage path: `~/.seismicx/workflows/`

### ⚙️ LLM Settings Page (/llm-settings)

- Online detection of installed Ollama models, one-click selection
- Supports configuring any OpenAI compatible API
- Takes effect immediately on all functions after saving
- Top badge displays currently used model in real-time

---

## Command Line Tools

`seismic_cli.py` provides complete command line interface, suitable for scripted and batch processing scenarios.

### Conversation Mode

```bash
python seismic_cli.py chat
```

### Phase Picking

```bash
# Single station picking
python seismic_cli.py pick \
    -i /data/station/ \
    -m pnsn/pickers/pnsn.v3.jit

# Batch picking (all waveform files in directory)
python seismic_cli.py pick \
    -i /data/seismic/2024/ \
    --batch \
    -o results/picks.csv

# Specify compute device
python seismic_cli.py pick -i /data/ --device cuda
```

### Event Association

```bash
python seismic_cli.py associate \
    -i results/picks.csv \
    -s station_list.csv \
    --method fastlink \
    -o results/events.txt
```

### Seismic Statistics

```bash
# Calculate b-value
python seismic_cli.py stats bvalue -i catalog.csv --mc auto

# Draw F-M distribution plot
python seismic_cli.py stats plot-gr -i catalog.csv -o fmd.png

# Generate complete statistical report (b-value + temporal + spatial distribution)
python seismic_cli.py stats report -i catalog.csv
```

### LLM Code Generation and Execution

```bash
python seismic_cli.py run "do 1-10Hz bandpass filtering on /data/wave.mseed and plot"
python seismic_cli.py run "calculate source parameters, epicentral distance 50km" -d /data/waves/
python seismic_cli.py run "draw travel time curve, distance 0-30°, depth 10km" --show-code
```

### Autonomous Agent

```bash
# Implement algorithm from local PDF
python seismic_cli.py agent \
    "implement the travel time residual correction method in the paper" \
    --paper /papers/velest_method.pdf \
    --data /data/picks.csv \
    --output results/agent_run/

# Implement from arXiv paper ID
python seismic_cli.py agent \
    "reproduce the b-value temporal analysis method in the paper" \
    --arxiv 2309.12345

# Implement from DOI
python seismic_cli.py agent \
    "implement HVSR spectrum ratio method" \
    --doi 10.1785/0220230045 \
    --max-steps 6
```

### Skill Management

```bash
python seismic_cli.py skill list                     # List all skills
python seismic_cli.py skill search "bandpass filter"        # Keyword search
python seismic_cli.py skill show waveform_processing # View complete documentation
python seismic_cli.py skill new my_tool              # Create custom skill
python seismic_cli.py skill edit my_tool             # Edit existing skill
python seismic_cli.py skill delete my_tool           # Delete skill
python seismic_cli.py skill dir                      # View skill directory path
```

### LLM Backend Management

```bash
python seismic_cli.py backend status          # View current status
python seismic_cli.py backend setup           # Interactive configuration wizard
python seismic_cli.py backend auto            # Auto-detect and select
python seismic_cli.py backend models          # List locally downloaded models
python seismic_cli.py backend pull qwen3:8b   # Pull Ollama model
```

---

## Conversation Routing Mechanism

SAGE automatically determines intent of each message through dedicated LLM routing calls, avoiding keyword mis-matching (for example, "Q-filter **algorithm**" will not be incorrectly routed to code execution).

### Routing Flow

```
User message
   │
   ├─ Fast path: message contains absolute path (/data/...) and is not a question
   │              └─→ code (execute directly)
   │
   └─ LLM routing call (max_tokens=10, approximately <1s)
          │
          ├─ code  → CodeEngine generates and executes Python / GMT code
          ├─ qa    → RAG retrieves knowledge base + LLM response
          └─ chat  → General conversation
```

### Three Types of Routing

| Route | Trigger Condition | Example |
|------|---------|------|
| `code` | Data processing, plotting, file operations, GMT maps | "Filter waveform with bandpass and plot", "Help me draw a Chinese topographic map with GMT" |
| `qa` | Concept explanation, method introduction, literature retrieval | "What is Q-filter?", "Explain the principle of HVSR" |
| `chat` | Greetings, chatting, non-seismological content | "Hello", "How is the weather today" |

**Fallback rules when LLM is unavailable:**

- Message contains `drawing/plotting/filtering/spectrum/waveform/.sac/.mseed` → `code`
- Others → `qa`

---

## seismo_skill Skill System

The skill system is SAGE's core extension mechanism. Each skill is a Markdown document describing function usage and code examples. **Skills documents are automatically retrieved and injected during AI conversation and code generation**, significantly improving the accuracy and standardization of generated code.

### Working Principle

```
User message (natural language)
       │
       ▼
  seismo_skill keyword retrieval
  (Chinese-English mixed TF-IDF scoring)
       │
       ├─ Matched skill → inject function signature + example code into LLM system prompt
       │
       ▼
  LLM generates code / responds
  (prioritizes standardized writing in skill documents)
```

Retrieval points integrated into:
- `/api/chat/rag` (Web knowledge Q&A)
- `seismo_code/code_engine.py` (code generation engine)
- `seismo_agent/agent_loop.py` (autonomous agent code generation at each step)

### Built-in Skills (7)

| Skill File | Category | Main Functions |
|----------|------|---------|
| `waveform_io.md` | waveform | `read_stream`, `read_stream_from_dir`, `stream_info`, `picks_to_dict` |
| `waveform_processing.md` | waveform | `detrend_stream`, `taper_stream`, `filter_stream`, `resample_stream`, `trim_stream`, `remove_response` |
| `waveform_visualization.md` | visualization | `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion` |
| `spectral_analysis.md` | analysis | `compute_spectrum`, `compute_hvsr` |
| `b_value_analysis.md` | statistics | `load_catalog_file`, `calc_mc_*`, `calc_bvalue_mle`, `plot_gr` |
| `source_parameters.md` | analysis | `estimate_magnitude_ml`, `estimate_corner_freq`, `estimate_seismic_moment`, `moment_to_mw`, `estimate_stress_drop` |
| `gmt_plotting.md` | visualization | `run_gmt` (epicenter maps, station maps, topographic maps, focal mechanisms, cross-sections) |

### Creating Custom Skills

**Method 1: Web Interface** (Recommended)

Visit `/skills` → Click "Create Custom Skill" → Fill in basic information → Complete documentation in editor.

**Method 2: Command Line**

```bash
python seismic_cli.py skill new my_hypodd_tool \
    --title "HypoDD Double Difference Location Tool" \
    --keywords "double difference location, HypoDD, precise location, relocation" \
    --desc "Package HypoDD input file generation and result parsing"
```

**Method 3: Directly Write Markdown File**

Create `.md` file under `~/.seismicx/skills/`:

```markdown
---
name: my_skill_name
category: custom
keywords: keyword1, keyword2, english_keyword
related_skills:            # optional — bidirectional skill expansion
  - waveform_io
  - tabular_io
workflow: seismicity_analysis   # optional — linked workflow name
---

# Skill Title

## Description

Tool function description (one or two sentences).

---

## Main Functions

### `function_name(param1, param2=default)`

**Parameters:**
- `param1` : type — description
- `param2` : type — description, default default

**Returns:** type — description

```python
# Minimal runnable example
result = function_name("input", param2=42)
print(result)
```

---

## Notes

- Note 1
```

> **Override Rules:** When custom skill has same name as built-in skill, custom version takes priority automatically.

### Building Documentation and Skills

SAGE supports automatic building of documentation and skills from external repositories. You can integrate third-party documentation into the knowledge base and generate corresponding skills.

#### Building from GitHub Repositories

1. **Clone Repository to Local Directory**
   ```bash
   # Example: Clone GMT Chinese documentation
   git clone https://github.com/gmt-china/GMT_docs.git
   ```

2. **Place in seismo_skill/docs Directory**
   ```bash
   # Move to SAGE project directory
   mv GMT_docs ~/path/to/sage/seismo_skill/docs/
   ```

3. **Access Knowledge Base Page**
   - Open web interface: `http://localhost:5000/knowledge`
   - Click the "Build Skills" button in the main knowledge base card
   - The system will automatically:
     - Scan `seismo_skill/docs/` directory
     - Index all documentation files (PDF, MD, RST, HTML, etc.)
     - Generate FAISS vector index for RAG retrieval
     - Create corresponding skill documents based on documentation content

4. **Supported Documentation Formats**
   - PDF documents
   - Markdown files (.md)
   - reStructuredText files (.rst)
   - HTML files (.html, .htm)
   - Text files (.txt)

#### Automatic Skill Generation

When building documentation, SAGE automatically analyzes the content and generates skill documents that include:

- Function signatures and usage examples
- Code snippets from documentation
- Parameter descriptions
- Best practices and notes

**Example Generated Skill Structure:**
```markdown
---
name: gmt_basemap
category: generated
keywords: GMT, basemap, map frame, projection
---

# GMT Basemap Drawing

## Description

Create map frames and coordinate systems using GMT's basemap module.

## Main Functions

### `psbasemap -R -J -B`

**Parameters:**
- `-R`: Region specification (e.g., -R0/10/0/10)
- `-J`: Projection type (e.g., -JM10c for Mercator)
- `-B`: Frame and annotation settings

**Example:**
```bash
gmt psbasemap -R0/360/-90/90 -JM10c -Bafg -P > map.ps
```

## Notes

- Use appropriate projection for your data region
- Frame annotations (-B) control tick marks and labels
```

#### Building Progress Monitoring

The build process runs in the background and provides real-time progress updates:

- **Scanning Phase**: Detects new/modified/deleted files
- **Indexing Phase**: Processes documents and builds vector embeddings
- **Skill Generation Phase**: Creates skill documents from indexed content
- **Completion**: Updates knowledge base statistics

You can monitor progress through the web interface or check logs in the terminal.

---

## seismo_script Workflow System

The workflow system lets you define multi-step analysis pipelines as declarative `.md` files. Each workflow specifies which skills to load, which steps to execute, and how those steps depend on each other. The Code Engine handles all code generation and execution — the workflow simply acts as the coordination blueprint.

### Role Distribution

| Role | Responsibility |
|------|---------------|
| **Workflow** | Process blueprint: what steps to run, which skills to use, in what order |
| **Skill** | Specialist manual: how to use a specific tool or method |
| **Agent** | Dispatcher: matches user request to workflow, loads skills, decomposes task |
| **Code Engine** | Programmer: generates and fixes Python/GMT/Shell code for each step |
| **Tool** | Executor: Python sandbox, GMT, Shell |

### Workflow File Format

Workflows use `.md` files with YAML frontmatter — the same format as skills:

```markdown
---
name: seismicity_analysis
title: Seismicity Analysis Workflow
version: "1.0"
description: Complete seismicity analysis including catalog loading, spatial/temporal distribution, and b-value estimation
keywords:
  - seismicity
  - b-value
  - epicenter map
skills:
  - name: tabular_io
    role: catalog loading and parsing
  - name: gmt_plotting
    role: epicenter map rendering
  - name: b_value_analysis
    role: b-value estimation and GR plots
steps:
  - id: load_catalog
    skill: tabular_io
    description: Load earthquake catalog from file
  - id: epicenter_map
    skill: gmt_plotting
    description: Draw epicenter distribution map
    depends_on: [load_catalog]
  - id: b_value
    skill: b_value_analysis
    description: Calculate b-value and plot GR distribution
    depends_on: [load_catalog]
---

## Seismicity Analysis Workflow Guide

Step 1: Load the catalog using `load_catalog_file()`...
```

**Frontmatter fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Workflow identifier |
| `title` | str | Human-readable title |
| `description` | str | One-line summary |
| `keywords` | list[str] | Used for relevance search |
| `skills` | list[{name, role}] | Required skills and their roles in this workflow |
| `steps` | list[{id, skill, description, depends_on}] | Execution DAG |

The Markdown body is the **workflow guide** — injected into the LLM context to direct code generation at each step.

### Storage

| Location | Contents |
|----------|----------|
| `seismo_script/workflows/` | Built-in workflows (shipped with SAGE) |
| `~/.seismicx/workflows/` | User-defined workflows (higher priority, override built-ins) |

### Built-in Workflows

| Workflow | Description | Skills |
|----------|-------------|--------|
| `gmt_terrain_map` | Full GMT terrain map pipeline (7 steps: CPT → DEM cut → render → coast → contours → scale/legend → export) | `gmt_plotting`, `_gen_gmt_docs_6_5` |
| `seismicity_analysis` | Seismicity analysis (catalog → epicenter map → time series → b-value → cross-section) | `tabular_io`, `gmt_plotting`, `b_value_analysis` |

### `CodeEngine.run_workflow()` API

```python
result: WorkflowRunResult = engine.run_workflow(
    workflow_name    = "seismicity_analysis",
    user_request     = "Analyze the 2024 catalog at /data/catalog.csv",
    data_hint        = "/data/catalog.csv",   # optional path hint injected into step prompts
    max_debug_rounds = 3,                     # retries per step on failure
    timeout          = 120,                   # per-step execution timeout (seconds)
    skip_on_failure  = False,                 # if True, skip failed steps instead of aborting
    on_progress      = callback_fn,           # optional: called with progress dicts
)
```

`run_workflow()` topo-sorts the step DAG, then for each step:
1. Checks all `depends_on` predecessors have succeeded
2. Scans the shared execution directory for available output files
3. Calls `build_skill_context_with_rag()` for the step's declared skill
4. Generates code via LLM (skill context + completed-steps summary injected)
5. Executes code in a shared directory (so step N+1 can read files written by step N)
6. On failure: re-queries RAG with the error text appended, retries up to `max_debug_rounds`
7. Records a `StepResult` and appends it to the shared conversation history

**`WorkflowRunResult`:**

```python
@dataclass
class WorkflowRunResult:
    workflow_name: str
    steps:         List[StepResult]   # one entry per executed step
    shared_dir:    str                # directory where all step output files live
    total_time:    float              # total wall-clock time (seconds)

    @property
    def failed_steps(self)  -> List[StepResult]: ...
    @property
    def skipped_steps(self) -> List[StepResult]: ...
```

**`StepResult`:**

```python
@dataclass
class StepResult:
    step_id:      str
    skill:        str
    description:  str
    success:      bool
    code:         str
    stdout:       str = ""
    stderr:       str = ""
    figures:      List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    attempts:     int = 1
    diagnosis:    str = ""
    skipped:      bool = False
```

### Web API

**Trigger a workflow run:**

```
POST /api/chat/workflow
Content-Type: application/json

{
  "workflow_name":   "seismicity_analysis",
  "message":         "Analyze the 2024 Sichuan catalog at /data/catalog.csv",
  "session_id":      "optional-session-id",
  "data_hint":       "/data/catalog.csv",
  "skip_on_failure": false
}

Response: { "ok": true, "job_id": "wf_xxxx" }
```

**Poll for results** (same endpoint as single-step code jobs):

```
GET /api/chat/code/poll/<job_id>

Response (completed):
{
  "status": "completed",
  "result": {
    "step_results": [
      { "step_id": "load_catalog", "success": true,  "figures": [...], "stdout": "..." },
      { "step_id": "epicenter_map","success": true,  "figures": ["/path/map.png"], "stdout": "" },
      { "step_id": "b_value",      "success": false, "diagnosis": "mc too high", "attempts": 3 }
    ],
    "shared_dir": "/tmp/sage_wf_xxxxx"
  }
}
```

### Creating Custom Workflows

**Method 1: Web Interface** (Recommended)

Visit `/skills` → **Workflows** tab → Click "New Workflow" → Fill in metadata → Edit the Markdown guide body. The step DAG preview updates live as you edit the frontmatter.

**Method 2: Write `.md` File Directly**

Save to `~/.seismicx/workflows/<name>.md` using the frontmatter format shown above. The file is picked up immediately (no restart needed).

---

## GMT Map Drawing

SAGE directly calls GMT6 through the `run_gmt()` utility function to generate professional-grade seismological maps.

### Installing GMT

```bash
# macOS
brew install gmt

# Linux (Conda environment)
conda install -c conda-forge gmt

# Linux (apt)
sudo apt install gmt
```

### Usage

Directly describe requirements in conversation, SAGE automatically generates and executes GMT script:

```
> Help me draw a Chinese topographic map with GMT
> Draw epicenter distribution map for 90-120°E, 20-45°N
> Draw station distribution map with GMT, data in /data/stations.txt
```

Or call in code (`run_gmt` is pre-injected, no import needed):

```python
gmt_script = """
gmt begin china_topo PNG
  gmt grdcut @earth_relief_01m -R70/140/15/55 -Gtopo.grd
  gmt grdimage topo.grd -JM16c -Cetopo1 -I+d
  gmt coast -W0.5p,gray40 -N1/0.8p -Baf -BWSne+t"China Topographic Map"
  gmt colorbar -DJBC+w8c/0.4c -Baf+l"Elevation (m)"
gmt end
"""

run_gmt(gmt_script, outname="china_topo", title="China Topographic Map")
```

### Automatic Chinese Title Processing

GMT's PostScript engine does not support CJK characters. SAGE automatically handles this issue:
1. Extract Chinese titles/labels from script before execution
2. Replace with empty placeholders, allowing GMT to render map content without garbled characters
3. After execution, overlay Chinese titles back onto PNG with matplotlib

> **User does not need to care about this detail**, just write Chinese titles directly in the script.

### Image and Script Download

Toolbar below each GMT image provides:
- **⬇ Image**: Download PNG file
- **⬇ GMT Script**: Download `.sh` script file, can independently run in terminal to completely reproduce the map

---

## EvidenceDrivenGeoAgent — Geoscience Interpretation Agent

`EvidenceDrivenGeoAgent` is an autonomous geoscience interpretation agent that follows an evidence-first, anti-hallucination design philosophy. Given a geological research question, it autonomously retrieves data from multiple sources, extracts structured evidence, generates and scores competing hypotheses, and finally produces a fully traceable interpretation report.

### Design Principles

| Principle | Implementation |
|---|---|
| **Evidence-first** | Every claim must reference at least one evidence record with source, confidence, and polarity |
| **Anti-hallucination** | Cannot assert any conclusion not supported by retrieved evidence |
| **Convergence detection** | Stops automatically when new evidence no longer changes hypothesis scores (Δ < 0.05) |
| **Competing hypotheses** | Always maintains ≥ 2 competing hypotheses until evidence definitively rules one out |
| **Full traceability** | Every step of the reasoning chain is logged in the tool call history |

### Architecture

```
EvidenceDrivenGeoAgent
├── AgentConfig              # All parameters: workspace, LLM, loop limits, capability flags
├── EvidenceRecord           # Structured evidence: content / source / confidence / polarity / data_type
├── GeoHypothesis            # Hypothesis: description / support_score / evidence_ids / status
├── AgentState               # Running state: question / evidence list / hypotheses / iteration counter
└── Tool Registry (9 tools)
    ├── retrieve_local_literature   # FAISS semantic search in local PDFs
    ├── retrieve_rag_chunks         # RAG vector database search
    ├── web_search                  # DuckDuckGo HTML + Semantic Scholar API
    ├── read_local_file             # Read CSV / GeoJSON / text data files
    ├── run_python_analysis         # Sandboxed Python execution (optional)
    ├── generate_hypothesis         # Propose new competing hypotheses
    ├── score_hypothesis            # Score and rank hypotheses based on evidence
    ├── write_report                # Generate structured Markdown interpretation report
    └── request_missing_info        # Document data gaps and additional information needed
```

### Agent Reasoning Loop

```
Question → Initialization
     ↓
┌─── Iteration N ────────────────────────────────────┐
│  1. Plan: which tools to call this round           │
│  2. Execute tools (≤ max_tool_calls_per_iter)      │
│  3. Extract evidence → append to evidence list     │
│  4. Update hypothesis scores                       │
│  5. Convergence check (Δscore < 0.05 for 2 iters) │
└─────────────────────────────────────────────────────┘
     ↓  (converged or max_iterations reached)
Write Report → Return AgentOutput
```

### Evidence Record Schema

```python
@dataclass
class EvidenceRecord:
    evidence_id:  str    # Unique ID, e.g. "ev_001"
    content:      str    # Evidence text content
    source:       str    # Source: filename / URL / "user_upload"
    source_type:  str    # "literature" | "rag" | "web" | "data" | "user_upload"
    confidence:   float  # 0.0 – 1.0
    polarity:     str    # "support" | "contradict" | "neutral"
    data_type:    str    # "text" | "numeric" | "geospatial" | "figure"
    timestamp:    str    # ISO 8601
```

### Nine Built-in Tools

| Tool | Description | Required Config |
|---|---|---|
| `retrieve_local_literature` | Semantic search in local PDF literature directory | `literature_root` set, PyPDF2 installed |
| `retrieve_rag_chunks` | Search FAISS / ChromaDB vector database | `use_rag=True`, RAG engine initialized |
| `web_search` | DuckDuckGo web search + Semantic Scholar academic search | `allow_web_search=True` |
| `read_local_file` | Read CSV / GeoJSON / TXT data files | `use_local_files=True`, `workspace_root` set |
| `run_python_analysis` | Execute Python code in a sandbox for data processing | `allow_python=True` |
| `generate_hypothesis` | Propose ≥ 2 competing interpretations for the question | Always available |
| `score_hypothesis` | Score hypotheses based on current evidence | Always available |
| `write_report` | Generate structured Markdown report with evidence citations | Always available |
| `request_missing_info` | Document data gaps and additional information needed | Always available |

### Data Source Priority

The agent searches data sources in the following priority order:

1. **User-uploaded files** (uploaded via the web UI in the current session) — highest priority, most relevant to the current task
2. **Local workspace files** (`workspace_root`) — project data files
3. **Local literature directory** (`literature_root`) — locally stored PDF papers
4. **RAG vector database** — indexed knowledge base (all uploaded historical documents)
5. **Web search** — DuckDuckGo + Semantic Scholar (only when `allow_web_search=True`)

### Convergence Conditions

The agent stops the reasoning loop when **any** of the following conditions is met:

- `max_iterations` reached (default: 3)
- Hypothesis score changes across two consecutive iterations all < 0.05 (convergence)
- A `write_report` tool call has been made
- All required tools have been called and evidence is sufficient

### Geo Agent Web UI

Access the web interface at `http://localhost:5000/evidence-geo-agent`.

**Left sidebar — Configuration panel:**

| Section | Controls |
|---|---|
| Workspace | Workspace path / Literature directory / Output directory |
| Loop Parameters | Max iterations / Max tool calls / RAG top-k / Score threshold |
| Capabilities | Python execution / RAG search / Local files / Multimodal / Web search |
| File Upload | Drag-and-drop upload of images, PDFs, CSVs (auto-classified) |
| Web Search | Keyword search across DuckDuckGo + Semantic Scholar |

**Main area — Interpretation task:**

| Field | Description |
|---|---|
| Research Question | The geological question to interpret, e.g. "What is the seismogenic structure of the 2023 M7.8 Turkey earthquake?" |
| Study Area | Geographic location (optional), e.g. "Kahramanmaraş, Turkey" |
| Example buttons | Click to auto-fill a typical question |

**Results tabs:**

| Tab | Content |
|---|---|
| Report | Full Markdown interpretation report, rendered to HTML |
| Evidence Table | All evidence records with source, confidence, polarity, collapsible |
| Hypotheses | All competing hypotheses with final scores and ranking |
| Figures | All figures output by the agent, displayed as a thumbnail grid |
| Tool Log | Complete tool call history per iteration, collapsible |
| Missing Info | Data gaps and additional information the agent could not obtain |

**Bilingual support:** Click the `中/EN` button in the top-right to toggle between Chinese and English. The preference is saved in `localStorage` and persists across sessions.

### File Upload for Research Data

The web UI supports drag-and-drop or click-to-select upload of multiple files simultaneously:

```
Supported formats:
  PDF              → auto-classified to literature/
  PNG / JPG / SVG  → auto-classified to figures/
  CSV              → auto-classified to data/
  TXT / JSON / XML → auto-classified to misc/
```

Each upload session gets an isolated workspace directory:

```
uploads/geo_workspaces/<session_id>/
├── literature/   ← uploaded PDF papers
├── figures/      ← uploaded images / diagrams
├── data/         ← uploaded CSV / data files
└── misc/         ← other file types
```

The agent automatically discovers and reads files from this workspace, with user uploads taking highest priority in the data source hierarchy.

**API endpoint:**

```
POST /api/evidence_geo_agent/upload
Content-Type: multipart/form-data

Fields:
  files[]     — one or more files
  session_id  — session identifier (generated by frontend if not provided)

Response:
  { "ok": true, "uploaded": [...], "session_id": "...", "workspace": "..." }
```

### Inline Web Literature Search

The web UI includes an instant web search panel in the left sidebar. Without starting a full agent run, you can directly search for relevant papers and web resources:

```
Search modes:
  scholar_search  — Semantic Scholar API, returns title / authors / year / abstract / citation count
  web_search      — DuckDuckGo HTML scraping, returns title / URL / snippet
```

**API endpoint:**

```
POST /api/evidence_geo_agent/web_search
Content-Type: application/json

Body:
  { "query": "East Anatolian Fault seismicity", "search_type": "scholar_search" }

Response:
  { "ok": true, "results": [ { "title": "...", "url": "...", "snippet": "..." }, ... ] }
```

### Geo Agent CLI Command

```bash
# Basic usage
sage-geo "What is the seismogenic mechanism of the 2021 Maduo earthquake?"

# Full options
sage-geo "Analyze the seismotectonics of Sichuan Basin" \
  --workspace /path/to/project \
  --literature /path/to/papers \
  --output outputs/my_report \
  --max-iter 5 \
  --web-search \
  --rag \
  --multimodal

# Available options
  --workspace       Project workspace root directory
  --literature      Local PDF literature directory
  --output          Output directory for results
  --max-iter        Maximum reasoning iterations (default: 3)
  --max-tools       Maximum tool calls per iteration (default: 8)
  --rag-k           RAG retrieval top-k (default: 8)
  --web-search      Enable web search (DuckDuckGo + Semantic Scholar)
  --no-rag          Disable RAG vector database search
  --no-local-files  Disable local file reading
  --multimodal      Enable multimodal analysis (requires vision-capable LLM)
  --allow-shell     Allow shell command execution
  --provider        LLM provider (ollama/openai/custom)
  --model           Model name
  --api-key         API key (for online providers)
  --api-base        API base URL (for custom/compatible providers)
```

**Output files:**

```
outputs/evidence_driven_geo_agent/
├── report_<timestamp>.md           # Full Markdown interpretation report
├── evidence_table_<timestamp>.json # All structured evidence records
├── hypotheses_<timestamp>.json     # All hypotheses with scores
└── agent_state_<timestamp>.json    # Complete agent state snapshot
```

### Geo Agent Flask API

**Start an interpretation task:**

```
POST /api/evidence_geo_agent
Content-Type: application/json

{
  "question":                "Analyze the seismogenic structure of the 2023 M7.8 Turkey earthquake",
  "study_area":              "Kahramanmaraş, Turkey",
  "session_id":              "optional-session-id",
  "workspace_root":          "/path/to/workspace",
  "literature_root":         "/path/to/papers",
  "output_dir":              "outputs/evidence_driven_geo_agent",
  "allow_web_search":        true,
  "use_rag":                 true,
  "use_local_files":         true,
  "use_multimodal":          false,
  "max_iterations":          3,
  "max_tool_calls_per_iter": 8,
  "rag_top_k":               8,
  "score_threshold":         0.35
}

Response: { "job_id": "geo_xxxx", "status": "started" }
```

**Poll for results:**

```
GET /api/evidence_geo_agent/poll/<job_id>

Response (running):
  { "status": "running", "progress": [...log lines...], "result": null }

Response (completed):
  {
    "status": "completed",
    "progress": [...],
    "result": {
      "report":          "# Interpretation Report\n...",
      "evidence_list":   [ { "evidence_id": "ev_001", ... }, ... ],
      "hypotheses":      [ { "description": "...", "support_score": 0.82, ... }, ... ],
      "figures":         [ "/api/evidence_geo_agent/figure?path=...", ... ],
      "tool_log":        [ { "iter": 1, "tool": "web_search", ... }, ... ],
      "missing_info":    [ "High-resolution focal mechanism data", ... ],
      "iterations_used": 3,
      "converged":       true
    }
  }
```

**Serve a figure:**

```
GET /api/evidence_geo_agent/figure?path=<relative-or-absolute-path>

Returns the image file (PNG/JPG/SVG) with appropriate Content-Type.
Path is sandbox-checked to stay within the project root or geo_workspaces directory.
```

### Python Programmatic Usage

```python
from sage_agents import EvidenceDrivenGeoAgent, AgentConfig

cfg = AgentConfig(
    workspace_root          = "/path/to/project",
    literature_root         = "/path/to/papers",
    output_dir              = "outputs/my_analysis",
    allow_web_search        = True,
    use_rag                 = True,
    use_multimodal          = False,
    max_iterations          = 5,
    max_tool_calls_per_iter = 10,
    rag_top_k               = 8,
    score_threshold         = 0.35,
    allow_python            = True,
    allow_shell             = False,
    code_timeout_s          = 60,
)

agent = EvidenceDrivenGeoAgent(config=cfg)

output = agent.run(
    question   = "What is the seismogenic structure of the 2021 Maduo earthquake?",
    study_area = "Maduo County, Qinghai Province, China",
)

print(output.report)
print(f"Converged in {output.iterations_used} iterations")
print(f"Evidence records: {len(output.evidence_list)}")
print(f"Top hypothesis: {output.hypotheses[0].description} (score={output.hypotheses[0].support_score:.2f})")
```

### Output Schema

```python
@dataclass
class AgentOutput:
    report:          str                    # Full Markdown interpretation report
    evidence_list:   List[EvidenceRecord]   # All structured evidence records
    hypotheses:      List[GeoHypothesis]    # All hypotheses, sorted by score descending
    figures:         List[str]              # Paths to output figures
    tool_log:        List[dict]             # Detailed tool call log per iteration
    missing_info:    List[str]              # Data gaps the agent could not fill
    iterations_used: int                    # Actual iterations completed
    converged:       bool                   # Whether stopped due to convergence
    error:           Optional[str]          # Error message if run failed
```

---

## Core Modules Details

### `seismo_script/` — Workflow System

```
seismo_script/
├── workflow_runner.py  # Workflow discovery, search, CRUD, and context building
├── workflows/          # Built-in workflow .md files (gmt_terrain_map, seismicity_analysis, ...)
└── __init__.py         # Public API: list_workflows, search_workflows, load_workflow,
                        #   save_user_workflow, delete_user_workflow, build_workflow_context
```

**Public API summary:**

| Function | Description |
|----------|-------------|
| `list_workflows()` | Return all workflow metadata (no guide body) |
| `search_workflows(query, top_k)` | Rank workflows by keyword relevance |
| `load_workflow(name)` | Return full workflow entry including guide text |
| `save_user_workflow(name, text)` | Save a `.md` file to `~/.seismicx/workflows/` |
| `delete_user_workflow(name)` | Delete a user-defined workflow |
| `build_workflow_context(query)` | Return `(context_str, skill_names)` for LLM injection |

### `seismo_code/` — Code Generation and Execution Engine

```
seismo_code/
├── code_engine.py      # LLM code generation (skill injection, multi-round history, error retry,
│                       #   run_workflow() multi-step DAG execution)
├── safe_executor.py    # Sandbox execution (independent subprocess, 120s timeout, automatic image collection)
├── toolkit.py          # Built-in seismological utility functions (no import needed, direct call)
└── doc_parser.py       # Extract context snippets related to code tasks from PDF
```

**Built-in Toolkit (`toolkit.py`, automatically injected during code execution):**

| Category | Functions |
|------|------|
| Data Reading | `read_stream`, `read_stream_from_dir` |
| Waveform Processing | `detrend_stream`, `taper_stream`, `filter_stream`, `resample_stream`, `trim_stream`, `remove_response` |
| Visualization | `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion`, `plot_travel_time_curve` |
| Travel Time Calculation | `taup_arrivals`, `p_travel_time`, `s_travel_time` |
| Spectrum Analysis | `compute_spectrum`, `compute_hvsr` |
| Source Parameters | `estimate_magnitude_ml`, `estimate_corner_freq`, `estimate_seismic_moment`, `moment_to_mw`, `estimate_stress_drop` |
| GMT Plotting | `run_gmt` |
| Utility Functions | `stream_info`, `picks_to_dict`, `savefig` |

**Sandbox Execution Mechanism:**
- Code runs in independent subprocess, main process unaffected by crashes
- Timeout protection (default 120 seconds)
- Generated images automatically collected via `[FIGURE] /path` marker and sent to frontend
- GMT scripts separately collected via `[GMT_SCRIPT] /path` marker, for frontend download provision

### `seismo_agent/` — Autonomous Agent

Complete automatic implementation flow from literature to code:

```
seismo_agent/
├── paper_reader.py   # Literature loading (PDF / arXiv ID / DOI / plain text)
├── memory.py         # Cross-step work memory (literature content, step results, generated variables)
├── planner.py        # LLM task planning (goal + literature summary → JSON step list)
└── agent_loop.py     # Main loop (planning → code → execution → failure retry → summary)
```

Execution flow:

```
User goal + literature source (PDF / arXiv / DOI)
       │
  Load and extract core literature content
       │
  LLM plans execution steps (3–8 steps, JSON format)
       │
  ┌─── Each Step ───────────────────────────┐
  │  Retrieve relevant skill documents (seismo_skill)     │
  │  LLM generates code (skill context injection)      │  ← Retry up to 2 times on failure
  │  Sandbox secure execution                        │
  │  Record results and generated images                   │
  └──────────────────────────────────────┘
       │ Loop through all steps
  Summary report + output directory
```

### `web_app/rag_engine.py` — Knowledge Base RAG Engine

| Stage | Implementation |
|------|------|
| PDF Parsing | pdfminer.six (priority) / PyMuPDF (fallback) |
| Text Chunking | 500 chars/chunk, 50 char sliding overlap |
| Vectorization | BGE-M3 (1024 dimensions, L2 normalized, Chinese-English bilingual) |
| Indexing | FAISS `IndexFlatIP` (inner product = cosine similarity) |
| Retrieval | Top-K recall + similarity threshold filtering, only showing truly matched literature |
| Persistence | `~/.seismicx/knowledge/`, automatic load on startup; automatic cleanup of orphaned vectors from deleted files on startup |
| Fallback | Automatic downgrade to TF-IDF cosine similarity retrieval when BGE-M3 unavailable |

### `seismo_stats/` — Seismic Statistical Analysis

```
seismo_stats/
├── bvalue.py         # Mc (maximum curvature / goodness-of-fit) + b-value (MLE / LSQ) + σ_b uncertainty
├── catalog_loader.py # Directory loading: CSV / JSON / picks.txt, automatic column name recognition
└── plotting.py       # F-M distribution plots, temporal activity plots, epicenter distribution plots
```

### `seismo_tools/` — External Tool Registry

Unified management of third-party seismological tools such as HypoDD, VELEST, HASH. Supports automatic control file generation, calling external executables, parsing output results, and can be triggered via conversation commands.

---

## Directory Structure

```
sage/
├── web_app/                      # Web service
│   ├── app.py                    # Flask main application (40+ API routes)
│   ├── rag_engine.py             # BGE-M3 + FAISS knowledge base engine
│   ├── simple_rag.py             # TF-IDF fallback RAG
│   ├── simple_vector_db.py       # Lightweight vector database (pickle persistence)
│   └── templates/
│       ├── chat.html             # Conversation page (main interface)
│       ├── knowledge.html        # Knowledge base management
│       ├── skills.html           # Skill management
│       └── llm_settings.html     # LLM configuration
│
├── seismo_skill/                 # Skill documentation system
│   ├── skill_loader.py           # Parse, retrieve, inject (Chinese-English mixed retrieval)
│   ├── __init__.py
│   ├── waveform_io.md            # Waveform reading
│   ├── waveform_processing.md    # Waveform preprocessing
│   ├── waveform_visualization.md # Waveform visualization
│   ├── spectral_analysis.md      # Spectrum analysis & HVSR
│   ├── b_value_analysis.md       # b-value statistical analysis
│   ├── source_parameters.md      # Source parameter estimation
│   ├── tabular_io.md             # CSV / TXT data reading
│   └── gmt_plotting.md           # GMT map drawing
│
├── seismo_script/                # Workflow system
│   ├── workflow_runner.py        # Workflow discovery, search, CRUD, context building
│   ├── workflows/                # Built-in workflow .md files
│   │   ├── gmt_terrain_map.md    # GMT terrain map 7-step pipeline
│   │   └── seismicity_analysis.md # Seismicity analysis pipeline
│   └── __init__.py
│
├── seismo_code/                  # Code generation and execution engine
│   ├── code_engine.py            # LLM code generation (multi-round history + error retry
│   │                             #   + run_workflow() DAG execution)
│   ├── safe_executor.py          # Sandbox execution (subprocess + timeout protection)
│   ├── toolkit.py                # Built-in seismological utility functions
│   └── doc_parser.py             # PDF content extraction
│
├── seismo_agent/                 # Autonomous Agent
│   ├── agent_loop.py             # Main loop (SeismoAgent class)
│   ├── planner.py                # Task planning (TaskPlanner)
│   ├── memory.py                 # Work memory (AgentMemory)
│   └── paper_reader.py           # Literature loading (load_paper)
│
├── seismo_stats/                 # Seismic statistical analysis
│   ├── bvalue.py                 # b-value / Mc calculation
│   ├── catalog_loader.py         # Earthquake catalog loading
│   └── plotting.py               # Statistical chart plotting
│
├── seismo_tools/                 # External tool registry
│   └── tool_registry.py          # HypoDD / VELEST / HASH etc.
│
├── pnsn/                         # ← Needs separate clone (see installation instructions)
│   ├── sage_picker.py            # Batch picking main class (SagePicker)
│   ├── fastlinker.py             # FastLink event association
│   ├── gammalink.py              # Gamma event association
│   ├── pickers/                  # JIT / ONNX model files
│   └── config/                   # Picker parameter configuration
│
├── conversational_agent.py       # Conversation Agent core (intent classification + skill execution)
├── config_manager.py             # LLM configuration management
├── backend_manager.py            # Multi-backend support (Ollama / vLLM / online API)
├── seismic_cli.py                # Command line entry point
├── requirements.txt              # Python dependencies
└── logo.png

~/.seismicx/                      # User data directory (automatically created on first run)
├── config.json                   # LLM and workspace configuration
├── knowledge/                    # Knowledge base vector index (FAISS + metadata)
│   ├── faiss_index.bin
│   ├── metadata.json
│   └── pdfs/                     # PDF copies
├── skills/                       # User custom skill documents
│   └── my_custom_skill.md
└── workflows/                    # User custom workflow .md files (override built-ins)
    └── my_custom_workflow.md
```

---

## Configuration Files

Configuration is unified in `~/.seismicx/config.json`, maintained automatically via Web interface or CLI, no manual editing required.

```json
{
  "llm": {
    "provider": "ollama",
    "model": "qwen3:30b",
    "api_base": "http://localhost:11434",
    "api_key": ""
  },
  "workspace": {
    "enabled": true,
    "path": "/data/seismic"
  }
}
```

| Field | Description | Optional Values |
|------|------|--------|
| `llm.provider` | LLM provider | `ollama` / `openai` / `custom` |
| `llm.model` | Model name | Ollama tag or API model name |
| `llm.api_base` | API endpoint address | `http://localhost:11434` (Ollama default) |
| `llm.api_key` | API key | Not required for Ollama |
| `workspace.enabled` | Whether to allow LLM to access local file lists | `true` / `false` |
| `workspace.path` | Authorized root directory (LLM cannot access content outside this path) | Absolute path string |

---

## FAQ

**Q: Conversation returns "No available LLM model configured"**

Go to `/llm-settings` to select an installed Ollama model, or configure an online API and click "Save Configuration".

**Q: English questions like "what is filter algorithm?" are incorrectly routed to code execution**

Fixed. SAGE uses LLM rather than keyword regex to determine intent, conceptual questions (containing technical terms like filter, spectrum) will be correctly routed to knowledge Q&A, not code execution.

**Q: Knowledge base PDF vectorization is slow after upload**

First run will download BGE-M3 model (~2 GB) from HuggingFace. Speed will be normal after completion. Domestic network can set mirror acceleration:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

If HuggingFace is completely inaccessible, use ModelScope to download the model locally (see [Alternative: Download BGE-M3 via ModelScope](#alternative-download-bge-m3-via-modelscope-recommended-for-users-in-china)) and then configure the local path in the Knowledge Base page settings.

**Q: Chinese titles in GMT images show as garbled characters**

No special handling required. SAGE has built-in CJK automatic processing: GMT execution stage replaces Chinese with empty placeholders, after execution matplotlib overlays Chinese titles back to PNG, ensuring correct Chinese display.

**Q: GMT plotting fails, prompting "GMT not installed"**

Install GMT >= 6.0:

```bash
# macOS
brew install gmt

# Linux (conda environment)
conda install -c conda-forge gmt
```

**Q: Batch picking is slow**

Default uses CPU. Add `--device cuda` to enable GPU acceleration (requires CUDA environment and corresponding PyTorch version).

**Q: Agent step execution fails**

Agent by default retries up to 2 times per step, failed steps will be skipped and subsequent steps continued. Can increase `--max-steps` limit, or check logs in output directory for details.

**Q: How to make AI use my own function library?**

Create a `.md` skill file under `~/.seismicx/skills/`, according to [skill file format](#creating-custom-skills) write function signatures, parameter explanations and minimal examples. No restart required after saving, takes effect immediately for next conversation.

**Q: RAG function reports error "embedding model library not found"**

```bash
# 1. Confirm installation
pip list | grep -E "(FlagEmbedding|sentence-transformers)"

# 2. Try upgrade
pip install --upgrade FlagEmbedding sentence-transformers

# 3. If Rust compiler needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
pip install FlagEmbedding sentence-transformers
```

If none of the above methods can solve, the project's built-in lightweight TF-IDF vector database will automatically serve as fallback solution, basic RAG functionality still available.

---

## Contact
SeismicX is developed by:
- **Yuqi Cai** - caiyuqiming@foxmail.com
- **Xin Liu** - xinliu_geo@outlook.com
- **Ziye Yu** - yuziye@cea-igp.ac.cn

---

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

**Q: How to add AI support for external tools like HypoDD?**

Call `register_tool()` in `seismo_tools/tool_registry.py` to register tool parameter templates and calling commands; simultaneously create corresponding skill document in `seismo_skill/`, describing input file format, allowing AI to automatically reference during code generation.

---

<p align="center">
  <sub>Built with ❤️ for the seismology community</sub>
</p>
