# EvidenceDrivenGeoAgent

An autonomous, multimodal, evidence-driven geoscience interpretation agent for SAGE.

The agent does **not** generate answers directly. It builds a structured evidence table first, then reasons.

---

## Design Principles

| Principle | How it is enforced |
|---|---|
| Evidence before reasoning | Every claim in the report cites a `[evidence_id]` |
| Observation ≠ interpretation | `observation` and `interpretation` are separate fields in every evidence record |
| Source priority | local data → RAG → literature → (web only if justified) |
| Traceable tool use | Every tool call is logged with `tool`, `method`, `args`, `reason`, `result_summary` |
| Path sandboxing | All file access is validated against `workspace_root` / `literature_root` |
| Anti-hallucination | Evidence extractor rejects any claim not directly traceable to tool output |
| Uncertainty flagging | `confidence`, `uncertainty`, `assumption` fields in every evidence record |

---

## Architecture

```
EvidenceDrivenGeoAgent
│
├── AgentConfig        — workspace roots, capability gates, loop limits
│
├── ToolRegistry       — 9 sandboxed, logged tools
│   ├── LocalFileSearchTool    list_dir / read_file / search_files / grep / get_file_metadata
│   ├── LiteratureLibraryTool  search_papers / read_pdf / extract_pdf_metadata / extract_references / search_bibtex
│   ├── RAGIndexTool           search_rag / add_document / index_documents / rebuild_index
│   ├── SeismoDataTool         read_catalog / read_station_file / read_waveform / read_velocity_model / read_focal_mechanisms
│   ├── GeoPlotTool            plot_catalog_map / plot_depth_section / plot_velocity_slice / plot_fault_distance / plot_evidence_map
│   ├── CodeExecutionTool      run_python / run_shell
│   ├── StateMemoryTool        save_state / load_state / save_report / save_evidence_table / save_hypotheses
│   ├── ImageAnalysisTool      analyze_image / extract_table   [requires use_multimodal=true + vision LLM]
│   └── WebSearchTool          web_search / scholar_search / download_pdf   [requires allow_web_search=true]
│
├── AgentLogger        — immutable ToolCall records for every invocation
│
└── LoopController     — orchestrates iterations
    │
    └── per iteration:
        ├─ Tool planner (LLM selects next tool call)
        ├─ Execute tool → log result
        ├─ Extract evidence (LLM parses output → GeoEvidence records)
        ├─ Update evidence table (dedup + conflict detection)
        ├─ Generate competing hypotheses
        ├─ Evaluate hypotheses
        ├─ Update report
        └─ Convergence check
```

---

## Agent Reasoning Loop

```
PLAN
  ↓
SEARCH (data priority: local structured data → RAG → literature → web)
  ↓
READ (text / PDF / figures / tables)
  ↓
EXTRACT EVIDENCE  ← NEVER skipped
  source_type: literature | local_data | model_derived | inference | speculation
  evidence_type: text | figure | table | data
  ↓
STRUCTURE EVIDENCE (evidence table, conflict detection)
  ↓
GENERATE HYPOTHESES (competing, each citing evidence IDs)
  ↓
VALIDATE (evaluate hypotheses, plan validation checks)
  ↓
UPDATE REPORT (every claim cited)
  ↓
DECIDE NEXT STEP → LOOP or CONVERGE
```

---

## Evidence Record Schema

```python
@dataclass
class GeoEvidence:
    evidence_id:              str   # e.g. "a3f9c12d01"
    source:                   str   # file path, doc name, URL
    source_type:              str   # literature | local_data | model_derived | inference | speculation
    evidence_type:            str   # text | figure | table | data
    observation:              str   # verbatim / close paraphrase — FACTUAL ONLY
    data_type:                str   # seismicity | velocity_model | focal_mechanism | ...
    spatial_scale:            str   # local | regional | crustal | lithospheric
    depth_range:              str   # e.g. "0–15 km"
    geological_structure:     str   # "Molingchang fault", "Sichuan Basin", ...
    interpretation:           str   # source's conclusion from observation
    alternative_interpretation: str # agent-generated alternative reading
    assumption:               str   # key assumptions
    confidence:               str   # high | medium | low
    uncertainty:              str   # stated source of uncertainty
    supports:                 list  # hypothesis_ids this evidence supports
    contradicts:              list  # hypothesis_ids this evidence contradicts
    conflict_with:            list  # evidence_ids that conflict
    citation:                 str   # short citation if available
    iteration:                int   # which loop iteration added this
    tool_call_id:             str   # which tool call produced this
```

**Source type definitions:**

| source_type | When to assign |
|---|---|
| `literature` | Extracted from indexed PDF, BibTeX, RAG KB, or downloaded paper |
| `local_data` | Computed or read from a local seismic catalog, waveform, velocity model, station file |
| `model_derived` | Output of a Python/shell code execution |
| `inference` | LLM-reasoned statement not directly stated in a source |
| `speculation` | Explicitly low-confidence; source or agent used hedging words ("may", "possibly") |

---

## Data Source Priority

The agent **must** follow this priority before escalating:

1. **Local structured data** — catalogs, velocity models, focal mechanisms (`seismo_data`)
2. **RAG knowledge base** — indexed PDFs and documents (`rag_index.search_rag`)
3. **Local PDF/literature library** — (`literature_library.search_papers`, `read_pdf`)
4. **Image analysis** — figures and maps in the workspace (`image_analysis.analyze_image`)
5. **Web search** — only when all local sources are insufficient AND the agent explicitly states why

---

## Tool Usage Rules

- **Every tool call must have a `reason`** — why this tool, why now
- **Every tool call is logged** — `tool`, `method`, `args`, `reason`, `result_summary`, `duration_s`
- **No blind tool calls** — the tool planner LLM must select based on evidence gaps
- **Web search requires justification** — the agent must state "local + RAG + literature insufficient because…"

---

## Output Schema

```json
{
  "question":             "...",
  "study_area":           "...",
  "iterations_run":       3,
  "final_report":         "# Interpretation Report\n...",
  "evidence_table": [
    {
      "evidence_id": "a3f9c12d01",
      "source": "weiyuan_catalog.csv",
      "source_type": "local_data",
      "evidence_type": "data",
      "observation": "31 events with M>4 within 5 km of Molingchang fault trace",
      "interpretation": "Spatial proximity suggests fault-controlled seismicity",
      "alternative_interpretation": "Proximity may reflect catalog completeness bias",
      "confidence": "high",
      "uncertainty": "Hypocenter location uncertainty ±0.5 km",
      "supports": ["H1"],
      "contradicts": []
    }
  ],
  "hypotheses": [
    {
      "hypothesis_id": "H1",
      "statement": "M>4 seismicity is controlled by reactivation of the Molingchang reverse fault",
      "supporting_evidence": ["a3f9c12d01", "b7e2a14f03"],
      "contradicting_evidence": [],
      "data_types_needed": ["focal_mechanism", "stress_field"],
      "confidence": "medium",
      "status": "active"
    }
  ],
  "tool_log": [
    {
      "call_id": "3a9f12bc",
      "iteration": 1,
      "tool": "seismo_data",
      "method": "read_catalog",
      "args": {"path": "data/weiyuan_catalog.csv"},
      "reason": "Read local catalog to extract seismicity statistics",
      "result_summary": "Read 128 rows from weiyuan_catalog.csv",
      "evidence_added": ["a3f9c12d01"],
      "duration_s": 0.12
    }
  ],
  "retrieved_sources":    ["weiyuan_catalog.csv", "field_notes.md"],
  "generated_figures":    ["outputs/catalog_map.png"],
  "missing_information":  ["Focal mechanism solutions for M>3 events", "..."],
  "suggested_validation": [...],
  "convergence_reason":   "no_new_evidence"
}
```

---

## Quick Start

### Programmatic

```python
from sage_agents import EvidenceDrivenGeoAgent, AgentConfig

config = AgentConfig(
    workspace_root  = "./examples/weiyuan",
    literature_root = "./papers/weiyuan",
    allow_python    = True,
    max_iterations  = 3,
)

agent  = EvidenceDrivenGeoAgent(config)
result = agent.run(
    question   = "Why are M>4 earthquakes near the Molingchang fault?",
    study_area = "Weiyuan, Sichuan Basin",
    on_progress = lambda d: print(f"[{d['phase']}] {d['message']}"),
)

print(result["final_report"])
```

### CLI

```bash
python seismic_cli.py evidence-geo-agent \
  --question "Why are M>4 earthquakes near the Molingchang fault?" \
  --study-area "Weiyuan, Sichuan Basin" \
  --workspace-root ./examples/weiyuan \
  --literature-root ./papers/weiyuan \
  --max-iterations 3 \
  --allow-python \
  --output results/weiyuan_interpretation.md \
  --verbose
```

### Flask API

```bash
# Start agent job
curl -X POST http://localhost:5000/api/evidence_geo_agent \
  -H "Content-Type: application/json" \
  -d '{
    "question":        "Why are M>4 earthquakes near the Molingchang fault?",
    "study_area":      "Weiyuan, Sichuan Basin",
    "workspace_root":  "./examples/weiyuan",
    "literature_root": "./papers/weiyuan",
    "max_iterations":  3,
    "allow_python":    true
  }'
# Returns: {"ok": true, "job_id": "geo_abc123"}

# Poll for result
curl http://localhost:5000/api/evidence_geo_agent/poll/geo_abc123
# Returns: {"ok": true, "status": "running|done|error", "progress": [...], "result": {...}}
```

### Example workflow

```bash
python examples/evidence_geo_agent/weiyuan_example.py
```

---

## Multimodal Mode

Set `use_multimodal=True` in `AgentConfig` (requires a **vision LLM** such as GPT-4o, LLaVA, or Qwen-VL):

```python
config = AgentConfig(
    workspace_root = "./examples/weiyuan",
    use_multimodal = True,     # enables analyze_image + extract_table
)
```

The agent will automatically call `image_analysis.analyze_image` on any PNG/JPG/JPEG files found in the workspace, extracting:

- What is plotted (data type, axes)
- Spatial distribution of features
- Key anomalies (clusters, velocity gradients, discontinuities)
- Depth and magnitude scales
- Stated interpretation vs. alternative interpretations

For tables in images, `image_analysis.extract_table` returns structured headers, rows, units, and pre-built evidence entries.

---

## Web Search Mode

Web search is **disabled by default**. Enable only when local sources are insufficient:

```python
config = AgentConfig(
    workspace_root  = "./examples/weiyuan",
    allow_web_search = True,  # enables web_search + scholar_search + download_pdf
)
```

The tool planner LLM is instructed to escalate to web search only after confirming that RAG + local files + literature library are all insufficient, and it must include the reason in every web search call log.

Available methods:
- `web_search(query)` — DuckDuckGo HTML scrape (no API key required)
- `scholar_search(query)` — Semantic Scholar API (no key required)
- `download_pdf(url, filename)` — fetch and save a PDF to `output_dir/downloaded_pdfs/`

---

## Path Sandboxing

All file-access tools enforce that every path resolves to within `workspace_root` or `literature_root`. Any attempt to access `/etc/passwd`, `../../secret`, or an absolute path outside the configured roots returns an `{"error": "Path '...' is outside workspace root."}` response without raising an exception.

Extension allowlist (`allowed_extensions` in `AgentConfig`) prevents reading binary executables, system libraries, or other non-data files.

---

## Convergence Conditions

The loop stops early when any of:

| Condition | `convergence_reason` |
|---|---|
| No new evidence added in iteration > 1 | `no_new_evidence` |
| Single high-confidence hypothesis remains active after iteration > 1 | `hypothesis_convergence` |
| `max_iterations` reached | `max_iterations_reached` |

---

## Running Tests

```bash
python -m pytest tests/test_evidence_driven_geo_agent.py -v
```

Covers: path sandboxing, LocalFileSearchTool, GeoEvidenceTableBuilder, ToolRegistry dispatch, AgentConfig, loop convergence (mocked LLM), GeoEvidence schema, WebSearchTool gates.

---

## Module Reference

```
sage_agents/
├── __init__.py                       exports all public classes
├── evidence_driven_geo_agent.py      main module (~1200 lines)
│   ├── AgentConfig                   dataclass
│   ├── GeoEvidence                   evidence record dataclass
│   ├── GeoHypothesis                 hypothesis dataclass
│   ├── GeoAgentResult                full result dataclass
│   ├── ToolCall                      tool log entry dataclass
│   ├── AgentLogger                   call recorder
│   ├── ToolRegistry                  dispatcher + sandbox enforcer
│   ├── LocalFileSearchTool           (Tool 1)
│   ├── LiteratureLibraryTool         (Tool 2)
│   ├── RAGIndexTool                  (Tool 3)
│   ├── SeismoDataTool                (Tool 4)
│   ├── GeoPlotTool                   (Tool 5)
│   ├── CodeExecutionTool             (Tool 6)
│   ├── StateMemoryTool               (Tool 7)
│   ├── ImageAnalysisTool             (Tool 8, multimodal)
│   ├── WebSearchTool                 (Tool 9, optional)
│   ├── GeoEvidenceTableBuilder       dedup + conflict detection
│   └── LoopController                main agent loop
└── literature_loop_agent.py          simpler RAG-only predecessor
```
