"""
Weiyuan Sichuan Basin — Evidence-Driven Geo Agent Example
==========================================================
This example demonstrates the EvidenceDrivenGeoAgent on the Weiyuan
induced-seismicity problem.

The agent will:
  1. Search local catalog and velocity model files
  2. Query the RAG knowledge base for relevant papers
  3. Search the PDF library for studies on Molingchang fault seismicity
  4. Extract structured evidence (text, figures, tables)
  5. Generate and evaluate competing hypotheses
  6. Write a traceable interpretation report

Run:
    python examples/evidence_geo_agent/weiyuan_example.py

Requires:
  - LLM configured (run:  python seismic_cli.py llm setup)
  - Optional: place seismic catalog CSVs in examples/evidence_geo_agent/data/
  - Optional: place PDFs in examples/evidence_geo_agent/literature/
"""

from __future__ import annotations

import sys
import os
import json
import time
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sage_agents import EvidenceDrivenGeoAgent, AgentConfig


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLE_DIR   = Path(__file__).parent
DATA_DIR      = EXAMPLE_DIR / "data"
LITERATURE_DIR = EXAMPLE_DIR / "literature"
OUTPUT_DIR    = EXAMPLE_DIR / "outputs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LITERATURE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Write sample catalog if none present ─────────────────────────────────────
SAMPLE_CATALOG = DATA_DIR / "weiyuan_catalog.csv"
if not SAMPLE_CATALOG.exists():
    SAMPLE_CATALOG.write_text(
        "lon,lat,depth,mag,time\n"
        "104.67,29.52,3.2,3.1,2019-01-15\n"
        "104.70,29.55,4.5,4.2,2019-02-03\n"
        "104.68,29.53,5.1,2.8,2019-02-20\n"
        "104.72,29.57,3.8,3.5,2019-03-10\n"
        "104.65,29.50,6.0,4.9,2019-04-01\n"
        "104.71,29.54,4.2,3.3,2019-04-18\n"
        "104.69,29.56,5.5,2.5,2019-05-07\n"
        "104.73,29.51,7.0,3.8,2019-06-12\n",
        encoding="utf-8",
    )
    print(f"[sample] Created sample catalog: {SAMPLE_CATALOG}")

# ── Write sample velocity model if none present ───────────────────────────────
SAMPLE_VMODEL = DATA_DIR / "weiyuan_vp_model.csv"
if not SAMPLE_VMODEL.exists():
    SAMPLE_VMODEL.write_text(
        "# Simple 1-D P-wave velocity model for Weiyuan area\n"
        "depth_km,vp_kms,vs_kms,density\n"
        "0,5.0,2.9,2.5\n"
        "5,5.8,3.3,2.7\n"
        "10,6.0,3.5,2.8\n"
        "15,6.2,3.6,2.9\n"
        "20,6.5,3.8,3.0\n"
        "30,6.8,3.9,3.1\n"
        "40,7.5,4.2,3.2\n",
        encoding="utf-8",
    )
    print(f"[sample] Created sample velocity model: {SAMPLE_VMODEL}")

# ── Write sample notes file ───────────────────────────────────────────────────
SAMPLE_NOTES = DATA_DIR / "field_notes.md"
if not SAMPLE_NOTES.exists():
    SAMPLE_NOTES.write_text(
        "# Weiyuan Field Notes\n\n"
        "## Observations\n"
        "- The Molingchang fault strikes NW-SE with a dip of ~70° to the SW.\n"
        "- Shale gas hydraulic fracturing operations started in Weiyuan in 2015.\n"
        "- Seismicity rate increased sharply after 2018 injection operations.\n"
        "- Focal depths cluster predominantly between 2–8 km (sedimentary cover).\n"
        "- Some M>4 events show focal depths of 4–6 km near known injection wells.\n\n"
        "## Fault Geometry\n"
        "- Molingchang fault is a reverse fault with minor left-lateral component.\n"
        "- The fault intersects the primary injection horizon at ~3.5 km depth.\n\n"
        "## Uncertainty\n"
        "- Hypocenter location uncertainty is estimated at ±0.5 km horizontal.\n"
        "- The connection between injection and seismicity timing is suggestive "
        "but not definitively proven.\n",
        encoding="utf-8",
    )
    print(f"[sample] Created sample field notes: {SAMPLE_NOTES}")


# ─────────────────────────────────────────────────────────────────────────────
# Agent configuration
# ─────────────────────────────────────────────────────────────────────────────

config = AgentConfig(
    workspace_root=str(EXAMPLE_DIR),
    literature_root=str(LITERATURE_DIR),
    output_dir=str(OUTPUT_DIR),

    # Enable Python for validation analyses (plots, statistics)
    allow_python=True,
    allow_shell=False,

    # Web search disabled by default — enable with allow_web_search=True
    # and justify: "RAG + local sources insufficient"
    allow_web_search=False,

    # Multimodal: enable if using a vision LLM (e.g. GPT-4o, LLaVA)
    use_multimodal=False,

    # Loop parameters
    max_iterations=3,
    max_tool_calls_per_iter=8,
    rag_top_k=8,
    score_threshold=0.30,
)


# ─────────────────────────────────────────────────────────────────────────────
# Progress callback
# ─────────────────────────────────────────────────────────────────────────────

PHASE_LABELS = {
    "iteration_start": "━━ Iteration",
    "tool_call":       "  ⚙  Tool",
    "tool_done":       "  ✓  Tool loop done",
    "evidence":        "  📋 Evidence",
    "hypothesising":   "  🔬 Hypotheses",
    "evaluating":      "  ⚖  Evaluation",
    "writing":         "  📝 Report",
    "converged":       "  ✓  Converged",
    "warning":         "  ⚠  Warning",
    "done":            "✓  Done",
}


def progress_cb(d: dict) -> None:
    phase = d.get("phase", "")
    msg   = d.get("message", "")
    label = PHASE_LABELS.get(phase, f"[{phase}]")
    print(f"{label}: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

QUESTION   = (
    "Why are M>4 earthquakes in Weiyuan spatially localised near the "
    "Molingchang fault, and what is the dominant mechanism — fault "
    "reactivation, fluid pressure diffusion, or stress transfer?"
)
STUDY_AREA = "Weiyuan, Sichuan Basin, SW China"

print("=" * 72)
print("EvidenceDrivenGeoAgent — Weiyuan Example")
print("=" * 72)
print(f"Question   : {QUESTION[:80]}…")
print(f"Study area : {STUDY_AREA}")
print(f"Workspace  : {config.workspace_root}")
print(f"Literature : {config.literature_root}")
print(f"Output     : {config.output_dir}")
print()

# Check LLM config
try:
    from config_manager import LLMConfigManager
    llm_cfg = LLMConfigManager().get_llm_config()
    if not llm_cfg.get("api_base"):
        print("⚠  LLM not configured. Run:  python seismic_cli.py llm setup")
        print("   Running in demo mode (no LLM calls will succeed).")
except Exception:
    llm_cfg = {}

t0 = time.time()

agent  = EvidenceDrivenGeoAgent(config=config)
result = agent.run(
    question=QUESTION,
    study_area=STUDY_AREA,
    on_progress=progress_cb,
)

elapsed = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
# Results summary
# ─────────────────────────────────────────────────────────────────────────────

print()
print("─" * 72)
print(f"✓  Completed in {elapsed:.1f}s")
print(f"   Iterations      : {result['iterations_run']}")
print(f"   Evidence records: {len(result['evidence_table'])}")
print(f"   Hypotheses      : {len(result['hypotheses'])}")
print(f"   Tool calls      : {len(result['tool_log'])}")
print(f"   Figures         : {len(result['generated_figures'])}")
print(f"   Sources         : {len(result['retrieved_sources'])}")
print(f"   Convergence     : {result['convergence_reason']}")
print()

# Evidence breakdown by source type
if result["evidence_table"]:
    from collections import Counter
    src_types = Counter(e["source_type"] for e in result["evidence_table"])
    ev_types  = Counter(e["evidence_type"] for e in result["evidence_table"])
    print("   Evidence breakdown:")
    for k, v in src_types.most_common():
        print(f"     source_type={k}: {v}")
    for k, v in ev_types.most_common():
        print(f"     evidence_type={k}: {v}")
    print()

# Print hypotheses
if result["hypotheses"]:
    print("   Hypotheses:")
    for h in result["hypotheses"]:
        status = h.get("status", "active")
        conf   = h.get("confidence", "?")
        stmt   = h.get("statement", "")[:80]
        print(f"     [{h['hypothesis_id']}] ({conf}, {status}) {stmt}")
    print()

# Missing information
if result["missing_information"]:
    print("   Missing information:")
    for m in result["missing_information"][:5]:
        print(f"     - {m[:80]}")
    print()

# Figures generated
if result["generated_figures"]:
    print("   Figures saved:")
    for fig in result["generated_figures"]:
        print(f"     {fig}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

report_path = OUTPUT_DIR / "weiyuan_interpretation.md"
report_path.write_text(result["final_report"], encoding="utf-8")
print(f"   Report   → {report_path}")

json_path = OUTPUT_DIR / "weiyuan_full_result.json"
json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"   Full JSON → {json_path}")

print()
print("Done. Open the report to review the evidence-driven interpretation.")
