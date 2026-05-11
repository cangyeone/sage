"""
agent_loop.py — 地震学自主 Agent 主循环

工作流程
--------
1. 加载文献（PDF / arXiv / DOI / 文本）
2. 用 LLM 理解文献，提取核心方法
3. 规划实现步骤（TaskPlanner）
4. 逐步执行（代码生成 + 沙箱执行）
5. 验证每步结果，失败时重试或重规划
6. 汇总所有结果、图像，输出摘要报告

Agent 模式
----------
- interactive  每步执行前征求用户确认（默认）
- autonomous   全自动，不打断用户
- single_step  只执行一步，返回结果
"""

from __future__ import annotations

import json
import os
import re
import shutil
import csv
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .memory import AgentMemory, StepResult
from .paper_reader import PaperStore, load_paper
from .planner import PlanStep, TaskPlanner

# seismo_skill 技能文档检索（可选依赖）
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from seismo_skill import build_skill_context as _build_skill_context
except Exception:
    def _build_skill_context(query: str, **_kw) -> str:  # type: ignore
        return ""


# ---------------------------------------------------------------------------
# Code generation prompt for individual steps
# ---------------------------------------------------------------------------

_STEP_SYSTEM = """\
You are a professional scientific researcher and Python engineer.
You are incrementally completing an autonomous scientific analysis task.

Pre-injected seismology toolkit (call directly — no import needed):
  read_stream(path)  read_stream_from_dir(directory)
  detrend_stream(st)  taper_stream(st)  filter_stream(st, type, freqmin, freqmax)
  plot_stream(st, title, outfile)  plot_psd(tr, outfile)
  plot_spectrogram(tr, outfile)  plot_particle_motion(st, outfile)
  taup_arrivals(dist_deg, depth_km, model)  plot_travel_time_curve(...)
  compute_spectrum(tr)  compute_hvsr(st, ...)
  estimate_magnitude_ml(tr, dist_km)  estimate_corner_freq(tr, ...)
  estimate_seismic_moment(tr, dist_km)  moment_to_mw(M0)  estimate_stress_drop(M0, fc)
  stream_info(st)  picks_to_dict(picks_file)  run_gmt(script, outname, title)

Rules:
1. Output ONLY ```python ... ``` code blocks — no explanations
2. Print numerical results with print() using clear labels
3. Save images via plot_* functions or savefig('name.png') — NEVER use plt.show()
4. Wrap critical steps in try/except with informative error messages
5. Variables computed in prior steps are available (listed in context)
6. Do not assume data schemas. First inspect files, delimiters, columns, units, and row examples.
7. Use os.environ["SAGE_OUTDIR"] for all outputs. Save intermediate CSV/JSON/Markdown notes when useful.
8. If fixing a failed attempt, explicitly diagnose the error in comments and change the approach.
9. The process cwd is os.environ["SAGE_WORKSPACE_ROOT"]; project-relative paths like data/file.txt are valid there.
10. Before reparsing raw files, check prior artifacts listed in context. If a validated CSV such as events_all.csv/events_A.csv exists, prefer reading it with pandas/csv and verify its columns/row count.
11. Every code step must include a small smoke test: print dataframe shapes, columns, and basic non-empty checks before downstream statistics or plotting.
12. This is scientific research, not a data-quality dashboard. Generate only figures/tables that directly test the current hypothesis. Put QC counts, schema checks, and generic histograms into a supplementary Markdown/CSV note instead of making them main figures.
"""

_TABLE_EXTS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".md", ".txt"}

_SCIENCE_FIGURE_POLICY = """\
Scientific artifact policy:
- Start from a mechanism-oriented research question and an explicit testable hypothesis.
- Use a two-stage artifact strategy:
  1) exploratory stage: generate a broad, evidence-rich candidate figure/table set
     (normally 6-10 candidate figures and 3-5 candidate tables/notes when data support it);
  2) convergence stage: select a smaller main-text set and demote the rest to supplement/QC.
- Main-paper figures should be few and evidence-rich after convergence: normally 3-5
  main figures plus 1-3 main tables, with additional useful artifacts kept as supplement.
- Avoid generic data-quality or descriptive plots as final main figures: raw count charts,
  quality-class bar charts, parameter-only histograms, and column/schema inventories
  belong in supplementary/QC notes unless they directly test the hypothesis.
- Each main figure must answer: what mechanism is being tested, what data support it,
  what alternative explanation it compares against, and what uncertainty remains.
- Prefer composite, hypothesis-driven figures over many standalone diagnostic plots.
- Every executable analysis step must state and verify its input/output contract:
  input files and fields, output artifacts, smoke-test checks, and evidence status.
- If evidence is insufficient, write missing_information.md instead of producing
  decorative or weakly motivated plots.
- Every CodeEngine script must print at least one line beginning with [SAGE_TEST]
  summarizing non-empty data checks and created artifact paths.
"""

_RECON_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "node_modules",
    "outputs",
}

_RECON_TEXT_EXTS = {
    ".txt",
    ".dat",
    ".csv",
    ".tsv",
    ".md",
    ".rst",
    ".py",
    ".sh",
    ".tex",
    ".bib",
    ".json",
    ".html",
    ".xml",
    ".yaml",
    ".yml",
}

_RECON_DOC_EXTS = {".doc", ".docx", ".pdf"}
_RECON_DATA_EXTS = {".txt", ".dat", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".h5", ".hdf5", ".nc", ".sac", ".mseed"}


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.as_posix()


def _read_text_preview(path: Path, max_chars: int = 12000) -> str:
    suffix = path.suffix.lower()
    if suffix in _RECON_TEXT_EXTS:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            try:
                return path.read_text(encoding="gb18030", errors="ignore")[:max_chars]
            except Exception:
                return ""
    if suffix == ".doc":
        try:
            proc = subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
            return (proc.stdout or proc.stderr or "")[:max_chars]
        except Exception:
            return ""
    if suffix == ".docx":
        try:
            import docx  # type: ignore

            document = docx.Document(str(path))
            return "\n".join(p.text for p in document.paragraphs if p.text.strip())[:max_chars]
        except Exception:
            return ""
    if suffix == ".pdf":
        try:
            from pdfminer.high_level import extract_text  # type: ignore

            return (extract_text(str(path), maxpages=2) or "")[:max_chars]
        except Exception:
            try:
                import fitz  # type: ignore

                doc = fitz.open(str(path))
                text = "\n".join(page.get_text("text") for page in doc[:2])
                doc.close()
                return text[:max_chars]
            except Exception:
                return ""
    return ""


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False


def _to_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except Exception:
        return None


def _split_data_line(line: str, delimiter: str = "") -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if delimiter == "comma":
        return [p.strip() for p in stripped.split(",")]
    if delimiter == "tab":
        return [p.strip() for p in stripped.split("\t")]
    if "," in stripped and stripped.count(",") >= stripped.count(" "):
        return [p.strip() for p in stripped.split(",")]
    if "\t" in stripped:
        return [p.strip() for p in stripped.split("\t")]
    return stripped.split()


def _detect_delimiter(lines: List[str]) -> str:
    comma = sum(1 for line in lines if line.count(",") >= 2)
    tab = sum(1 for line in lines if line.count("\t") >= 2)
    if comma and comma >= tab:
        return "comma"
    if tab:
        return "tab"
    return "whitespace"


def _infer_column_name(index: int, values: List[str], known_layout: bool) -> tuple[str, float, str]:
    numeric_values = [_to_float(v) for v in values]
    numeric_values = [v for v in numeric_values if v is not None]
    unique = {v for v in values if v not in {"", "nan", "NaN", "NA"}}

    if known_layout:
        mapping = {
            0: ("event_id", 0.75, "first integer-like token in focal-mechanism rows"),
            1: ("year", 0.9, "year-like column in data notes/sample rows"),
            2: ("month", 0.85, "calendar month column"),
            3: ("day", 0.85, "calendar day column"),
            4: ("hour", 0.85, "hour column"),
            5: ("minute", 0.85, "minute column"),
            6: ("second", 0.85, "second column"),
            8: ("magnitude_or_size", 0.55, "numeric event-size column near time fields"),
            10: ("latitude", 0.9, "values fall in geographic latitude range and match sample notes"),
            11: ("longitude", 0.9, "values fall in geographic longitude range and match sample notes"),
            12: ("depth_km", 0.85, "positive depth-like column after coordinates"),
            21: ("strike1_deg", 0.65, "focal-mechanism angle column"),
            22: ("dip1_deg", 0.65, "focal-mechanism angle column"),
            23: ("rake1_deg", 0.65, "focal-mechanism angle column"),
            24: ("strike2_or_aux_deg", 0.45, "auxiliary mechanism-angle column"),
            25: ("dip2_or_aux_deg", 0.45, "auxiliary mechanism-angle column"),
            26: ("rake2_or_aux_deg", 0.45, "auxiliary mechanism-angle column"),
            28: ("quality_class", 0.9, "A/B/C quality label column"),
            29: ("fit_or_score", 0.45, "post-quality numeric score column"),
            30: ("station_or_count", 0.35, "post-quality count-like column"),
        }
        if index in mapping:
            return mapping[index]

    if unique and unique <= {"A", "B", "C"}:
        return ("quality_class", 0.8, "low-cardinality A/B/C text values")
    if numeric_values:
        vmin, vmax = min(numeric_values), max(numeric_values)
        if 1900 <= vmin <= 2100 and 1900 <= vmax <= 2100:
            return ("year", 0.75, "year-like numeric range")
        if 1 <= vmin and vmax <= 12 and index <= 6:
            return ("month_or_small_time_field", 0.45, "small calendar/time-like range")
        if -90 <= vmin <= 90 and -90 <= vmax <= 90 and len(numeric_values) >= max(3, len(values) // 2):
            if 20 <= sum(numeric_values) / len(numeric_values) <= 50:
                return ("latitude_candidate", 0.55, "geographic latitude range")
        if -180 <= vmin <= 180 and -180 <= vmax <= 180 and len(numeric_values) >= max(3, len(values) // 2):
            if 70 <= sum(numeric_values) / len(numeric_values) <= 140:
                return ("longitude_candidate", 0.55, "geographic longitude range")
        if 0 <= vmin and vmax <= 700:
            return ("positive_quantity_or_depth_candidate", 0.3, "non-negative geophysical range")
    return (f"col_{index:02d}", 0.0, "not enough evidence to infer semantic meaning")


def _profile_text_data(path: Path, root: Path, preview: str) -> Dict[str, Any]:
    lines = [line for line in preview.splitlines() if line.strip() and not line.lstrip().startswith(("#", "%"))]
    sample_lines = lines[:200]
    delimiter = _detect_delimiter(sample_lines)
    rows = [_split_data_line(line, delimiter) for line in sample_lines]
    rows = [row for row in rows if row]
    ncols_counter = Counter(len(row) for row in rows)
    ncols = ncols_counter.most_common(1)[0][0] if ncols_counter else 0
    consistent_rows = [row for row in rows if len(row) == ncols][:100]
    known_maduo_mechanism = False
    if ncols >= 29 and consistent_rows:
        row = consistent_rows[0]
        known_maduo_mechanism = (
            len(row) > 28
            and _looks_numeric(row[1])
            and 1900 <= float(row[1]) <= 2100
            and row[28] in {"A", "B", "C"}
        )

    schema: List[Dict[str, Any]] = []
    for index in range(ncols):
        values = [row[index] for row in consistent_rows if index < len(row)]
        numeric_values = [_to_float(v) for v in values]
        numeric_values = [v for v in numeric_values if v is not None]
        inferred, confidence, evidence = _infer_column_name(index, values, known_maduo_mechanism)
        stats: Dict[str, Any] = {
            "sample_values": list(dict.fromkeys(values[:8])),
            "numeric_ratio": round(len(numeric_values) / max(1, len(values)), 3),
        }
        if numeric_values:
            stats.update({
                "min": min(numeric_values),
                "max": max(numeric_values),
                "mean": sum(numeric_values) / len(numeric_values),
            })
        schema.append({
            "name": f"col_{index:02d}",
            "inferred_name": inferred,
            "confidence": confidence,
            "evidence": evidence,
            "stats": stats,
        })

    role = "numeric_text_data" if ncols >= 3 and any(col["stats"]["numeric_ratio"] > 0.5 for col in schema) else "text_or_notes"
    if known_maduo_mechanism:
        role = "focal_mechanism_catalog"
    supported = []
    inferred_names = {col["inferred_name"] for col in schema}
    if {"latitude", "longitude"} & inferred_names or {"latitude_candidate", "longitude_candidate"} <= inferred_names:
        supported.append("spatial_mapping")
    if "depth_km" in inferred_names or "positive_quantity_or_depth_candidate" in inferred_names:
        supported.append("depth_distribution")
    if "quality_class" in inferred_names:
        supported.append("quality_filtering")
    if {"strike1_deg", "dip1_deg", "rake1_deg"} <= inferred_names:
        supported.extend(["focal_mechanism_classification", "mechanism_orientation_statistics"])
    if {"year", "month", "day"} & inferred_names:
        supported.append("time_series")

    return {
        "path": _safe_rel(path, root),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "role": role,
        "delimiter": delimiter,
        "sample_line_count": len(rows),
        "dominant_column_count": ncols,
        "column_count_distribution": dict(ncols_counter.most_common(8)),
        "schema": schema,
        "supported_analyses": sorted(set(supported)),
        "sample_rows": [" ".join(row[: min(len(row), 16)]) for row in consistent_rows[:5]],
        "uncertainties": [] if supported else ["File may need domain-specific parsing before use."],
    }


def _classify_non_data_file(path: Path, root: Path, preview: str) -> Dict[str, Any]:
    suffix = path.suffix.lower()
    lowered = f"{path.name}\n{preview[:1000]}".lower()
    if suffix in {".pdf"} or any(k in lowered for k in ["abstract", "doi", "journal", "references", "关键词", "摘要"]):
        role = "reference_paper_or_report"
    elif suffix in {".doc", ".docx", ".md", ".rst", ".txt"} and any(k in lowered for k in ["数据说明", "field", "字段", "format", "文件", "notes"]):
        role = "data_description_or_project_notes"
    elif suffix in {".py", ".sh"}:
        role = "script_or_code"
    elif suffix in {".tex", ".bib"}:
        role = "paper_template_or_bibliography"
    else:
        role = "project_file"
    return {
        "path": _safe_rel(path, root),
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "role": role,
        "preview": preview[:1500],
    }


def _iter_recon_files(project_root: Path, max_files: int = 2000) -> List[Path]:
    files: List[Path] = []
    for path in sorted(project_root.rglob("*")):
        if len(files) >= max_files:
            break
        if path.is_dir():
            continue
        rel_parts = set(path.relative_to(project_root).parts[:-1])
        if rel_parts & _RECON_SKIP_DIRS:
            continue
        if any(part.startswith(".") and part not in {".gmt"} for part in path.relative_to(project_root).parts[:-1]):
            continue
        suffix = path.suffix.lower()
        if suffix in _RECON_TEXT_EXTS or suffix in _RECON_DOC_EXTS or suffix in _RECON_DATA_EXTS:
            files.append(path)
    return files


def _write_builtin_reconnaissance(project_root: str, output_dir: str, goal: str) -> Dict[str, Any]:
    root = Path(project_root).resolve()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    files = _iter_recon_files(root)
    entries: List[Dict[str, Any]] = []
    candidate_data = 0
    data_notes = 0
    literature = 0

    for path in files:
        preview = _read_text_preview(path)
        suffix = path.suffix.lower()
        if suffix in _RECON_DATA_EXTS and suffix in {".txt", ".dat", ".csv", ".tsv"}:
            entry = _profile_text_data(path, root, preview)
        else:
            entry = _classify_non_data_file(path, root, preview)
        if entry.get("role") in {"numeric_text_data", "focal_mechanism_catalog"}:
            candidate_data += 1
        if entry.get("role") == "data_description_or_project_notes":
            data_notes += 1
        if entry.get("role") == "reference_paper_or_report":
            literature += 1
        entries.append(entry)

    manifest = {
        "project_root": str(root),
        "goal": goal,
        "scanned_files": len(files),
        "candidate_data_files": candidate_data,
        "data_note_files": data_notes,
        "reference_files": literature,
        "entries": entries,
    }

    manifest_path = out / "data_schema_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Data Reconnaissance Report",
        "",
        f"- Project root: `{root}`",
        f"- Research goal: {goal}",
        f"- Scanned files: {len(files)}",
        f"- Candidate data files: {candidate_data}",
        f"- Data-note files: {data_notes}",
        f"- Reference papers/reports: {literature}",
        "",
        "## Recommended Analysis Order",
        "",
        "1. Read data-note/project-note files first and use them to verify inferred schemas.",
        "2. Parse candidate data files with explicit column checks and non-empty smoke tests.",
        "3. Generate a broad exploratory figure/table pool, then keep only mechanism-oriented evidence in the main paper.",
        "4. Treat generic field lists, quality counts, and parser diagnostics as supplementary/QC material.",
        "",
        "## Candidate Data Files",
        "",
    ]
    for entry in entries:
        if entry.get("role") not in {"numeric_text_data", "focal_mechanism_catalog"}:
            continue
        md_lines.extend([
            f"### `{entry['path']}`",
            "",
            f"- Role: `{entry.get('role')}`",
            f"- Delimiter: `{entry.get('delimiter')}`",
            f"- Dominant columns: `{entry.get('dominant_column_count')}`",
            f"- Supported analyses: {', '.join(entry.get('supported_analyses') or ['to be determined'])}",
            "",
            "| Column | Inferred meaning | Confidence | Evidence |",
            "|---|---:|---:|---|",
        ])
        for col in entry.get("schema", [])[:40]:
            md_lines.append(
                f"| `{col['name']}` | `{col['inferred_name']}` | {col['confidence']:.2f} | {col['evidence']} |"
            )
        md_lines.extend(["", "**Sample rows**", ""])
        for row in entry.get("sample_rows", [])[:5]:
            md_lines.append(f"- `{row}`")
        md_lines.append("")

    md_lines.extend([
        "## Data Notes And Literature Files",
        "",
        "| File | Role | Preview evidence |",
        "|---|---|---|",
    ])
    for entry in entries:
        if entry.get("role") in {"numeric_text_data", "focal_mechanism_catalog"}:
            continue
        preview = (entry.get("preview") or "").replace("\n", " ")[:180]
        md_lines.append(f"| `{entry['path']}` | `{entry.get('role')}` | {preview} |")

    recon_path = out / "data_reconnaissance.md"
    recon_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    contract_lines = [
        "# IO Contracts For Scientific Analysis",
        "",
        "These contracts are generated before planning so every later coding step has explicit inputs, outputs, and checks.",
        "",
        "## 1. Data Parsing",
        "",
        "- Inputs: candidate data files listed in `data_schema_manifest.json`.",
        "- Required checks: row count > 0, dominant column count stable, required inferred fields present or marked uncertain.",
        "- Outputs: normalized CSV/JSON tables in the run output directory; parser summary Markdown.",
        "",
        "## 2. Exploratory Figure/Table Pool",
        "",
        "- Inputs: normalized event tables plus literature/data-note context.",
        "- Required checks: every figure/table prints source rows, selected fields, and output path.",
        "- Outputs: candidate figures/tables testing spatial, depth, time, segmentation, mechanism, and robustness hypotheses.",
        "",
        "## 3. Scientific Convergence",
        "",
        "- Inputs: candidate artifacts and claim-evidence-warrant matrix.",
        "- Required checks: each main conclusion cites at least one data/statistical artifact and one literature/RAG/web evidence item when available.",
        "- Outputs: Markdown paper draft with embedded figures, supplement/QC notes, and missing-information list.",
        "",
    ]
    contracts_path = out / "io_contracts.md"
    contracts_path.write_text("\n".join(contract_lines), encoding="utf-8")

    stdout = (
        f"[SAGE_TEST] scanned_files={len(files)} candidate_data_files={candidate_data} "
        f"data_notes={data_notes} references={literature} manifest={manifest_path}\n"
        f"Created {recon_path}\nCreated {contracts_path}"
    )
    return {
        "manifest": manifest,
        "stdout": stdout,
        "output_files": [str(manifest_path), str(recon_path), str(contracts_path)],
        "candidate_data_files": candidate_data,
        "scanned_files": len(files),
        "data_notes": data_notes,
        "references": literature,
    }


def _generate_step_code(
    step: PlanStep,
    paper_context: str,
    memory_context: str,
    llm_config: Dict,
    goal: str,
    error_context: str = "",
) -> str:
    """Call LLM to generate code for a single step."""
    user_content = (
        f"Overall goal: {goal}\n\n"
        f"Current step [{step.index}]: {step.description}\n"
        f"Expected output: {step.expected_output}\n\n"
    )
    if paper_context:
        user_content += f"Paper method summary:\n{paper_context[:3000]}\n\n"
    if memory_context:
        user_content += f"Prior step results (variables/files available for reuse):\n{memory_context}\n\n"
    if error_context:
        user_content += (
            "Previous attempt failed. Diagnose and repair it automatically.\n"
            "Use the traceback/stdout below to change the plan, inspect files, or choose a simpler robust method.\n"
            f"{error_context[:4000]}\n\n"
        )
    user_content += "Generate complete Python code for this step:"

    # Inject relevant skill documentation based on step description + goal
    skill_query = f"{goal} {step.description}"
    skill_ctx = _build_skill_context(skill_query, max_chars=4000, top_k=2)
    step_system = _STEP_SYSTEM
    if skill_ctx:
        step_system = _STEP_SYSTEM + "\n\n" + skill_ctx

    messages = [
        {"role": "system", "content": step_system},
        {"role": "user", "content": user_content},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.2, "num_predict": 3000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 3000}

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"})

    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode())

    import re
    if provider == "ollama":
        raw = body.get("message", {}).get("content", "")
    else:
        raw = body.get("choices", [{}])[0].get("message", {}).get("content", "")

    return _extract_python_code(raw)


def _extract_python_code(raw: str) -> str:
    """Extract usable Python from an LLM response, including unclosed fences."""
    text = (raw or "").strip()
    if not text:
        return ""

    # Prefer python/py fenced blocks. Some local models forget the closing fence,
    # so the pattern accepts end-of-string as a valid terminator.
    blocks = re.findall(r"```(?:python|py)\s*\n?(.*?)(?:```|\Z)", text, flags=re.I | re.S)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)(?:```|\Z)", text, flags=re.S)
    if blocks:
        text = max(blocks, key=len).strip()

    # Last-ditch cleanup for responses that still include Markdown wrappers or
    # labels. Keep this intentionally small; the debug loop handles real errors.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        if stripped.lower() in {"python", "py"}:
            continue
        lines.append(line)
    text = "\n".join(lines).strip()

    try:
        from seismo_code.ce_utils import _pre_sanitize  # type: ignore
        text = _pre_sanitize(text)
    except Exception:
        pass
    return text


def _python_syntax_error(code: str) -> str:
    """Return a concise syntax error for generated code, or an empty string."""
    try:
        compile(code, "<llm_generated_step>", "exec")
        return ""
    except SyntaxError as exc:
        line = ""
        if exc.text:
            line = exc.text.rstrip()
        return f"SyntaxError before execution: {exc.msg} at line {exc.lineno}: {line}"


def _explain_paper_methods(paper_context: str, goal: str, llm_config: Dict) -> str:
    """Ask LLM to summarize the key method from the paper for the given goal."""
    if not paper_context.strip():
        return ""

    messages = [
        {"role": "system", "content": (
            "你是一位地震学专家。请从以下文献内容中，提取与用户目标最相关的"
            "核心方法、公式和算法步骤，用简洁的中文总结（500字以内）。"
        )},
        {"role": "user", "content": f"用户目标：{goal}\n\n文献内容：\n{paper_context[:5000]}"},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")

    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": 1000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1000}

    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"})

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        if provider == "ollama":
            return body.get("message", {}).get("content", "")
        return body.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        return ""


def _write_markdown_paper(
    goal: str,
    method_summary: str,
    memory: AgentMemory,
    figures: List[str],
    output_files: List[str],
    output_dir: str,
    llm_config: Dict,
    artifact_plan: str = "",
    evidence_synthesis: str = "",
    rag_context: str = "",
    web_context: str = "",
    followup_questions: Optional[List[str]] = None,
) -> str:
    """Use the LLM to synthesize a Markdown paper draft with figure links."""
    output_path = Path(output_dir) / "science_paper_draft.md"
    fig_lines = "\n".join(f"- {Path(f).name}: {f}" for f in figures) or "(no generated figures)"
    file_lines = "\n".join(f"- {Path(f).name}: {f}" for f in output_files) or "(no extra output files)"
    table_summary = _summarize_table_artifacts(output_files)
    step_log = memory.steps_summary()
    followup_text = "\n\n---\n\n".join(followup_questions or [])
    prompt = f"""
Research goal:
{goal}

Literature/method summary:
{method_summary or '(not available)'}

LLM-planned paper figures and tables:
{artifact_plan or '(no explicit artifact plan was generated)'}

Final claim-evidence synthesis brief:
{evidence_synthesis or '(not available; infer cautiously only from the evidence below)'}

Scientific follow-up questions and tested hypotheses:
{followup_text or '(none)'}

RAG/local knowledge evidence excerpts:
{rag_context or '(not available)'}

Online literature evidence excerpts:
{web_context or '(not available)'}

Executed analysis log:
{step_log}

Generated figures:
{fig_lines}

Generated data/files:
{file_lines}

Generated tables/statistical artifacts:
{table_summary or '(no generated table/statistical artifact previews)'}

Write a grounded Markdown research paper draft in Chinese. Use this structure:
# Title
## Abstract
## Introduction
## Data
## Methods
## Results
## Discussion
## Conclusions
## Limitations and Missing Information
## References and Evidence

Rules:
- The Results section is the scientific core. It must not read like a data-quality report or a list of figure captions.
- Organize Results by mechanism-oriented claims, then use the generated figures/tables as evidence for those claims.
- Embed all scientifically relevant candidate figures with Markdown image syntax. Mark them as Main Figure or Supplementary Figure according to the artifact refinement plan.
- Each Results subsection must follow: claim -> data/statistical evidence -> literature/web evidence -> mechanism interpretation -> uncertainty/counter-evidence.
- Every main claim must cite at least one generated data/table/figure artifact; if local literature or web evidence is available, connect it explicitly by title/DOI/URL/source name.
- Prefer strong scientific headings such as "Shallow weak zones modulate reverse faulting" over descriptive headings such as "data quality and feature statistics".
- Use only the figures that directly support the core scientific question as main figures; do not center the paper on QC/diagnostic plots. Extra diagnostic plots may be embedded in a Supplementary evidence section when they explain uncertainty, filtering, or robustness.
- Include Markdown tables for the most important statistical artifacts. If a table file is listed, summarize its key columns/values and cite the file path.
- If no figure or table was generated, explicitly state that the analysis is incomplete and do not pretend a result is supported.
- Distinguish observations, interpretations, hypotheses, and missing evidence. Use labels such as 已验证 / 部分支持 / 待验证.
- Do not invent citations, numbers, or conclusions. If evidence is missing, say so.
- Mention which output files support each main claim.
- Do not merely write "Figure X shows"; explain what scientific question the figure tests and what alternative explanation remains possible.
- The paper should be substantial. Unless evidence is truly insufficient, write detailed Introduction, Results, Discussion, and Limitations sections rather than a brief status report.
{_SCIENCE_FIGURE_POLICY}
"""

    messages = [
        {"role": "system", "content": "你是一位严谨的科研论文写作助手，只根据给定日志、图件、文件和文献摘要写作。"},
        {"role": "user", "content": prompt},
    ]

    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")
    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.3, "num_predict": 8000}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 8000}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode())
    if provider == "ollama":
        text = body.get("message", {}).get("content", "").strip()
    else:
        text = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    if not text:
        text = "# Science Paper Draft\n\nNo draft was generated."
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def _plan_paper_artifacts(goal: str, method_summary: str, memory_context: str, llm_config: Dict) -> str:
    """
    Let the LLM decide which figures/tables are scientifically warranted before coding.

    The output is deliberately raw text rather than JSON so small/local models can
    participate. Downstream CodeEngine treats it as a research design brief.
    """
    prompt = f"""
你是严谨的科学论文图表编辑和数据分析设计者。

请根据研究目标、文献/方法背景和已有上下文，规划这篇论文最应该生成哪些图和表。
不要机械固定模板；只规划当前数据和研究问题真正支持的图表。

{_SCIENCE_FIGURE_POLICY}

要求：
1. 先识别研究区、数据类型、空间/时间范围、可能的构造背景和可检验假设。
2. 设计“数据区域研究 -> 统计绘图 -> 图后提出新科学问题 -> 再验证”的迭代闭环。
3. 再列出探索型候选图件方案：Candidate Figure 1, Candidate Figure 2... 每张图说明目的、输入数据、统计/绘图方法、预期检验什么，以及看图后可能触发的下一轮追问。
4. 再列出候选表格方案：Candidate Table 1, Candidate Table 2... 每张表说明字段、统计量、来源文件、用途。
5. 每个图表必须说明证据来源；如果数据不足，标记“待验证/缺失信息”。
6. 不要编造不存在的数据列或结论；把不确定性写清楚。
7. 探索阶段应尽可能充分：如果数据支持，先规划 6-10 张候选图件和 3-5 个候选表格/统计笔记，覆盖研究区背景、空间、深度、时间、分段、机制分类、文献假设对比和稳健性。
8. 收敛阶段再从候选图表中选择 3-5 张主文图和 1-3 张主表；其余保留为 supplementary/QC 产物，不要删除。
9. 输出中文 raw text，便于后续 CodeEngine 编程执行。

研究目标：
{goal}

文献/方法背景：
{method_summary or '(not available)'}

已有执行上下文/文件线索：
{memory_context or '(none)'}
"""
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")
    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": {"temperature": 0.25, "num_predict": 1800}}
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.25, "max_tokens": 1800}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = json.loads(resp.read().decode())
        if provider == "ollama":
            return body.get("message", {}).get("content", "").strip()
        return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as exc:
        return f"图表规划失败：{exc}"


def _llm_text(prompt: str, llm_config: Dict, *, timeout: int = 90, max_tokens: int = 1800, temperature: float = 0.25) -> str:
    """Small raw-text LLM helper. Avoids JSON so local/smaller models can participate."""
    provider = llm_config.get("provider", "ollama")
    model = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key = llm_config.get("api_key", "")
    if provider == "ollama":
        url = api_base.rstrip("/") + "/api/chat"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
    else:
        url = api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else "Bearer none",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
    if provider == "ollama":
        return body.get("message", {}).get("content", "").strip()
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def _build_agent_rag_context(query: str, *, top_k: int = 6, max_chars: int = 5000) -> str:
    """Retrieve project/global KB evidence for scientific follow-up planning."""
    try:
        import sys
        web_app_dir = Path(__file__).parent.parent / "web_app"
        if str(web_app_dir) not in sys.path:
            sys.path.insert(0, str(web_app_dir))
        from rag_engine import get_knowledge_base  # type: ignore

        kb = get_knowledge_base()
        return kb.build_rag_context(query, top_k=top_k, max_chars=max_chars)
    except Exception as exc:
        return f"(RAG context unavailable: {exc})"


def _build_agent_web_context(query: str, *, max_results: int = 5, max_chars: int = 5000) -> str:
    """Fetch lightweight online literature clues for follow-up planning."""
    q = " ".join(str(query or "").split())[:500]
    if not q:
        return ""
    try:
        url = (
            "https://api.openalex.org/works?search="
            + urllib.parse.quote(q)
            + f"&per-page={max(1, min(max_results, 10))}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "SeismicX/0.1 (mailto:research@example.com)"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="ignore"))
        lines = ["Online literature clues from OpenAlex (verify before citing):"]
        for item in body.get("results", [])[:max_results]:
            title = (item.get("title") or "").strip()
            year = item.get("publication_year") or ""
            doi = item.get("doi") or ""
            venue = ((item.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
            authors = []
            for auth in (item.get("authorships") or [])[:4]:
                name = ((auth.get("author") or {}).get("display_name") or "").strip()
                if name:
                    authors.append(name)
            abstract = ""
            inv = item.get("abstract_inverted_index") or {}
            if isinstance(inv, dict):
                pairs = []
                for word, positions in inv.items():
                    for pos in positions:
                        pairs.append((pos, word))
                abstract = " ".join(w for _, w in sorted(pairs))[:450]
            lines.append(
                f"- {title} ({year}). {', '.join(authors)}. {venue}. DOI/URL: {doi or item.get('id', '')}. "
                + (f"Abstract clue: {abstract}" if abstract else "")
            )
        return "\n".join(lines)[:max_chars]
    except Exception as exc:
        return f"(Web literature context unavailable: {exc})"


def _plan_followup_question(
    goal: str,
    method_summary: str,
    artifact_plan: str,
    memory_context: str,
    rag_context: str,
    web_context: str,
    llm_config: Dict,
    round_no: int,
) -> str:
    """Ask the LLM for the next mechanism-oriented, testable scientific follow-up."""
    prompt = f"""
你是 Scientific Analysis Agent 的“科学追问规划器”。
你的任务不是总结数据质量，而是基于已有证据提出下一轮值得用编程验证的科学问题。

输出 raw text，不要输出 JSON。若没有足够依据继续追问，请只输出：
NO_FOLLOWUP_NEEDED: 原因

追问必须满足：
1. 面向机制问题，例如断层几何、弱层/低速体、应力转移、分段破裂、震源机制随空间/深度变化、构造控制等。
2. 能被当前数据、已有图表/统计、RAG/文献证据和一次 CodeEngine 计算部分验证或证伪。
3. 必须列出：QUESTION、HYPOTHESIS、WHY_NOW、DATA_REGION_OR_SUBSET、DATA_TO_USE、COMPUTATION、FIGURES_TABLES、EVIDENCE_REQUIRED、STOP_CRITERIA。
4. 不要把猜想写成事实；证据不足就写“待验证/缺失信息”。
5. 必须基于上一轮图形/表格/统计中显露出的模式提出问题，例如空间分段、深度层化、沿走向变化、异常区、机制类型转变或与文献假设不一致处。
6. 避免重复上一轮已经做过的统计；优先提出更高层科学问题。

这是第 {round_no} 轮追问。

研究总目标：
{goal}

文献/方法摘要：
{method_summary or '(not available)'}

论文图表初始规划：
{artifact_plan or '(not available)'}

已完成步骤、图件、表格和错误上下文：
{memory_context or '(none)'}

RAG/知识库证据（仅作为线索，必须以完整证据为准）：
{rag_context or '(not available)'}

在线文献线索（必须二次核验后才能作为引用，不要直接编造成事实）：
{web_context or '(not available)'}
"""
    try:
        return _llm_text(prompt, llm_config, timeout=90, max_tokens=1800, temperature=0.25)
    except Exception as exc:
        return f"NO_FOLLOWUP_NEEDED: follow-up planning failed: {exc}"


def _no_followup_needed(text: str) -> bool:
    return not text.strip() or text.strip().upper().startswith("NO_FOLLOWUP_NEEDED")


def _synthesize_scientific_claims(
    goal: str,
    method_summary: str,
    artifact_plan: str,
    memory_context: str,
    figures: List[str],
    output_files: List[str],
    rag_context: str,
    web_context: str,
    followup_questions: List[str],
    llm_config: Dict,
) -> str:
    """Build a claim-evidence-warrant brief before final paper writing."""
    fig_lines = "\n".join(f"- {Path(f).name}: {f}" for f in figures) or "(no generated figures)"
    file_lines = "\n".join(f"- {Path(f).name}: {f}" for f in output_files) or "(no generated files)"
    table_summary = _summarize_table_artifacts(output_files)
    followup_text = "\n\n---\n\n".join(followup_questions) or "(none)"
    prompt = f"""
你是地震学论文的科学主编。你的任务不是润色，而是在写论文前把数据、图表、
本地文献/RAG、在线文献线索和追问验证结果综合成“科学结论-证据-反证”简报。

请输出中文 Markdown raw text，不要输出 JSON。

研究目标：
{goal}

文献/方法摘要：
{method_summary or '(not available)'}

LLM 原始图表规划：
{artifact_plan or '(not available)'}

已完成计算与日志：
{memory_context or '(none)'}

生成图件：
{fig_lines}

生成文件：
{file_lines}

统计表格预览：
{table_summary or '(none)'}

科学追问与验证：
{followup_text}

RAG/本地知识库证据：
{rag_context or '(not available)'}

在线文献线索（需要谨慎引用，只能作为线索或已核验证据）：
{web_context or '(not available)'}

请严格按下面结构输出：

## Core Scientific Question
用一句话写出这篇文章真正想回答的机制问题。

## Main Claim-Evidence Matrix
| Claim | Data/statistical evidence | Local literature/RAG evidence | Web evidence | Counter-evidence or missing information | Status |
|---|---|---|---|---|---|

## Results Blueprint
为 Results 设计 3-5 个小节。每个小节必须包含：
- 机制型小节标题
- 要证明/反证的具体 claim
- 需要引用的图、表、统计文件
- 需要连接的本地论文或在线文献证据
- 仍然不能证明的部分

## Discussion Logic
说明这些结果怎样共同支持一个新的科学解释，而不是只描述数据质量。

约束：
- 不要把“Figure X shows...”当作结论；图件只是证据。
- 每个 claim 必须至少有一个数据/统计/图件来源；没有就标记“待验证”。
- 本地文献证据请尽量写出论文标题、DOI、文件名或 paper 编号。
- 在线证据请写出标题/DOI/URL；如果只是线索，明确写“线索，待核验”。
- 不要编造数值、引用或结论；证据不足要写缺失信息。
- 优先挖掘类似“浅部弱层、断层几何、分段破裂、局部应力释放模式、流体/低速层控制”等机制问题。
"""
    try:
        return _llm_text(prompt, llm_config, timeout=120, max_tokens=3200, temperature=0.2)
    except Exception as exc:
        return f"证据综合失败：{exc}"


def _plan_artifact_refinement(
    goal: str,
    artifact_plan: str,
    evidence_synthesis: str,
    figures: List[str],
    output_files: List[str],
    llm_config: Dict,
) -> str:
    """Review figures/tables after conclusions emerge and plan add/drop/refine actions."""
    fig_lines = "\n".join(f"- {Path(f).name}: {f}" for f in figures) or "(no generated figures)"
    file_lines = "\n".join(f"- {Path(f).name}: {f}" for f in output_files) or "(no generated files)"
    table_summary = _summarize_table_artifacts(output_files)
    prompt = f"""
你是科学论文的图表审稿人。现在已经形成了初步科学结论，请反过来审查图件和表格：
哪些应该保留为主文图表，哪些应该降为补充/QC，哪些还缺少、需要补充生成，哪些会削弱论证。

输出 raw text，不要 JSON。若不需要任何调整，请只输出：
NO_ARTIFACT_CHANGE: 原因

研究目标：
{goal}

原始图表规划：
{artifact_plan or '(not available)'}

科学结论与证据综合：
{evidence_synthesis or '(not available)'}

已有图件：
{fig_lines}

已有文件：
{file_lines}

表格/统计预览：
{table_summary or '(none)'}

请按以下结构输出：

## Main-paper artifact decision
- KEEP_MAIN: 列出应作为主文 Figure/Table 的文件名和原因。
- DEMOTE_SUPPLEMENT: 列出应降为补充或 QC 的文件名和原因。
- REMOVE_FROM_ARGUMENT: 列出不应在论文论证中使用的文件名和原因。

## Missing artifact plan
只列出真正能加强核心科学结论的缺失图件/表格。每项必须写：
- artifact name
- scientific claim it tests
- input files/statistics
- expected computation
- why existing artifacts are insufficient

## Caption/argument repair
指出哪些图表 caption 或正文引用需要从“描述图”改成“证明 claim”。

约束：
- 不要为了好看而加图；只为证明或反证核心科学结论加图。
- 不要建议删除磁盘文件；只决定主文/补充/不引用。
- 缺失图表最多 2 个，缺失主表最多 1 个。
- 如果证据不足，建议生成 missing_information.md，而不是编造图件。
"""
    try:
        return _llm_text(prompt, llm_config, timeout=90, max_tokens=2200, temperature=0.2)
    except Exception as exc:
        return f"NO_ARTIFACT_CHANGE: artifact refinement planning failed: {exc}"


def _no_artifact_change_needed(text: str) -> bool:
    return not text.strip() or text.strip().upper().startswith("NO_ARTIFACT_CHANGE")


def _three_reviewer_review(
    goal: str,
    draft_text: str,
    evidence_synthesis: str,
    artifact_refinement_plan: str,
    figures: List[str],
    output_files: List[str],
    rag_context: str,
    web_context: str,
    llm_config: Dict,
    round_no: int,
) -> str:
    """Simulate three strict reviewers and return raw review text."""
    fig_lines = "\n".join(f"- {Path(f).name}: {f}" for f in figures) or "(no generated figures)"
    file_lines = "\n".join(f"- {Path(f).name}: {f}" for f in output_files) or "(no generated files)"
    table_summary = _summarize_table_artifacts(output_files, max_chars=5000)
    prompt = f"""
你是一个严格的期刊内部审稿委员会，请模拟 3 个审稿人审稿。
目标是让 Scientific Analysis Agent 的论文从“简单描述数据”提升到“可投稿的科学论证”。

输出 raw text，不要 JSON。必须包含下面四行之一的判定格式：
REVIEWER_1_DECISION: ACCEPT|MINOR|MAJOR|REJECT
REVIEWER_2_DECISION: ACCEPT|MINOR|MAJOR|REJECT
REVIEWER_3_DECISION: ACCEPT|MINOR|MAJOR|REJECT
OVERALL_DECISION: ACCEPT|MINOR|MAJOR|REJECT

审稿人角色：
1. Reviewer 1 - Science/Mechanism: 是否提出了新的机制性科学问题？结论是否超越数据质量描述？
2. Reviewer 2 - Data/Statistics/Reproducibility: 数据解析、统计、图表和代码证据是否足够支撑结论？
3. Reviewer 3 - Literature/Evidence/Writing: 是否充分结合本地论文、RAG、web search，是否避免幻觉，引用是否清楚？

第 {round_no} 轮审稿。

研究目标：
{goal}

当前论文草稿：
{draft_text[:18000]}

科学证据综合：
{evidence_synthesis or '(not available)'}

结论驱动图表复审：
{artifact_refinement_plan or '(not available)'}

已有图件：
{fig_lines}

已有文件：
{file_lines}

表格/统计预览：
{table_summary or '(none)'}

RAG/本地证据：
{rag_context or '(not available)'}

Web 证据：
{web_context or '(not available)'}

每个审稿人请给：
- Major concern(s)
- Required revision(s)
- Suggested figure/table changes
- Evidence/citation gaps
- 是否需要新增计算或只是改写

判定标准：
- ACCEPT/MINOR：科学主张清楚，主要证据链完整，只需措辞、引用、图注或小补充。
- MAJOR：核心科学结论仍太弱，缺少关键图/表/统计/文献证据，或 Results 仍像数据描述。
- REJECT：结论无法由现有数据支持，或明显有编造/错误证据。
"""
    try:
        return _llm_text(prompt, llm_config, timeout=120, max_tokens=3500, temperature=0.15)
    except Exception as exc:
        return f"OVERALL_DECISION: MINOR\n审稿模拟失败：{exc}"


def _review_is_minor_or_better(review_text: str) -> bool:
    text = (review_text or "").upper()
    if "REJECT" in text or "MAJOR" in text:
        return False
    if "OVERALL_DECISION: ACCEPT" in text or "OVERALL_DECISION: MINOR" in text:
        return True
    decisions = re.findall(r"REVIEWER_\d+_DECISION:\s*(ACCEPT|MINOR|MAJOR|REJECT)", text)
    return bool(decisions) and all(d in {"ACCEPT", "MINOR"} for d in decisions)


def _review_requests_more_analysis(review_text: str) -> bool:
    text = (review_text or "").lower()
    triggers = (
        "new computation", "additional computation", "新增计算", "补充计算",
        "新增图", "补图", "additional figure", "new figure",
        "新增表", "补表", "additional table", "new table",
    )
    return any(t in text for t in triggers)


def _revise_markdown_paper_with_reviews(
    goal: str,
    draft_text: str,
    review_text: str,
    evidence_synthesis: str,
    artifact_refinement_plan: str,
    figures: List[str],
    output_files: List[str],
    llm_config: Dict,
) -> str:
    """Revise the Markdown paper according to the three-reviewer critique."""
    fig_lines = "\n".join(f"- {Path(f).name}: {f}" for f in figures) or "(no generated figures)"
    file_lines = "\n".join(f"- {Path(f).name}: {f}" for f in output_files) or "(no generated files)"
    prompt = f"""
你是论文第一作者。请根据三位严格审稿人的意见修订 Markdown 论文草稿。

研究目标：
{goal}

审稿意见：
{review_text}

科学证据综合：
{evidence_synthesis or '(not available)'}

结论驱动图表复审：
{artifact_refinement_plan or '(not available)'}

可用图件：
{fig_lines}

可用文件：
{file_lines}

当前草稿：
{draft_text[:20000]}

修订要求：
- 保留 Markdown 结构，并输出完整修订稿。
- Results 必须以科学 claim 为小节核心，而不是图件说明。
- 对每个核心 claim 明确写出数据/表格/图件证据、文献或 web 证据、反证与缺失信息。
- 根据审稿意见把不支撑主线的图件降为补充，不要强行嵌入。
- 不要编造数字、引用或不存在的图表。证据不足就写待验证。
- 使用 Markdown 图片语法嵌入主文图件，例如 ![caption](figure.png)。
"""
    try:
        text = _llm_text(prompt, llm_config, timeout=120, max_tokens=4200, temperature=0.2)
        return text.strip() or draft_text
    except Exception:
        return draft_text


def _is_table_artifact(path: str | Path) -> bool:
    p = Path(path)
    if p.suffix.lower() not in _TABLE_EXTS:
        return False
    name = p.name.lower()
    if p.suffix.lower() in {".csv", ".tsv", ".xlsx", ".xls", ".json"}:
        return True
    return any(k in name for k in ("table", "stat", "summary", "result", "metrics", "evidence"))


def _table_artifacts(paths: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in paths:
        if not item or item in seen:
            continue
        seen.add(item)
        if _is_table_artifact(item) and Path(item).exists():
            out.append(item)
    return out


def _summarize_table_artifacts(paths: List[str], max_tables: int = 8, max_chars: int = 7000) -> str:
    """Return compact previews of table/statistical files for paper writing."""
    lines: List[str] = []
    for item in _table_artifacts(paths)[:max_tables]:
        p = Path(item)
        suffix = p.suffix.lower()
        lines.append(f"### {p.name}\nPath: {p}")
        try:
            if suffix in {".csv", ".tsv"}:
                dialect = "\t" if suffix == ".tsv" else ","
                with p.open(newline="", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f, delimiter=dialect)
                    rows = [row for _, row in zip(range(8), reader)]
                if rows:
                    header = rows[0]
                    lines.append(f"Columns: {header}")
                    lines.append(f"Preview rows: {rows[1:5]}")
            elif suffix in {".xlsx", ".xls"}:
                try:
                    import pandas as pd  # type: ignore
                    df = pd.read_excel(p, nrows=5)
                    lines.append(f"Shape preview: {df.shape}; columns: {list(df.columns)}")
                    lines.append(df.head(5).to_markdown(index=False))
                except Exception as exc:
                    lines.append(f"Excel preview unavailable: {exc}")
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")[:1200]
                lines.append(text)
        except Exception as exc:
            lines.append(f"Preview unavailable: {exc}")
        lines.append("")
        if len("\n".join(lines)) > max_chars:
            break
    return "\n".join(lines)[:max_chars].strip()


# ---------------------------------------------------------------------------
# SeismoAgent
# ---------------------------------------------------------------------------

@dataclass
class AgentRunResult:
    success: bool
    summary: str
    figures: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    steps_completed: int = 0
    steps_total: int = 0
    output_dir: str = ""


class SeismoAgent:
    """
    地震学自主 Agent

    Usage
    -----
    agent = SeismoAgent(llm_config)
    result = agent.run(
        goal="实现文献中的 HVSR 分析方法并绘图",
        paper_source="/path/to/paper.pdf",
        output_dir="results/agent_run/",
        progress_cb=print,
    )
    """

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        project_root: Optional[str] = None,
        mode: str = "autonomous",
    ):
        if llm_config is None:
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config_manager import LLMConfigManager
                llm_config = LLMConfigManager().get_llm_config()
            except Exception:
                llm_config = {"provider": "ollama", "model": "",
                              "api_base": "http://localhost:11434"}

        self.llm_config = llm_config
        self.project_root = project_root or str(Path(__file__).parent.parent)
        self.mode = mode
        self.memory = AgentMemory()
        self.planner = TaskPlanner(llm_config)

    def is_llm_available(self) -> bool:
        try:
            provider = self.llm_config.get("provider", "ollama")
            api_base = self.llm_config.get("api_base", "http://localhost:11434")
            model = self.llm_config.get("model", "")
            if provider != "ollama":
                # Many OpenAI-compatible online providers either do not expose
                # /models or require different permissions. Chat itself uses
                # /chat/completions, so do not reject a configured online model
                # before the actual request path has a chance to run.
                return bool(api_base and model)
            url = api_base.rstrip("/") + "/api/tags"
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    def load_paper(self, source: str, cb: Optional[Callable] = None) -> bool:
        """Load a paper from source (PDF path, arXiv ID, DOI, or text)."""
        if cb:
            cb(f"📖 加载文献：{source[:60]}")
        try:
            paper = load_paper(source)
            key = self.memory.add_paper(paper)
            if cb:
                cb(f"   ✓ 已加载：{paper.title[:60]} [{key}]")
            return True
        except Exception as e:
            if cb:
                cb(f"   ⚠  文献加载失败：{e}")
            return False

    def run(
        self,
        goal: str,
        paper_source: Optional[Any] = None,
        output_dir: Optional[str] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
        guidance_provider: Optional[Callable[[], str]] = None,
        max_steps: int = 8,
        max_retries: int = 2,
        max_followup_rounds: int = 2,
        max_review_rounds: int = 3,
        produce_latex: bool = True,
        use_web_search: bool = True,
    ) -> Dict:
        """
        Run the full agentic loop.

        Parameters
        ----------
        goal : str
            Research/programming goal in natural language.
        paper_source : str or list[str], optional
            Path/URL/ID of one or more papers to read.
        output_dir : str, optional
            Directory for output figures and files.
        progress_cb : callable, optional
            Called with progress messages.
        max_steps : int
            Safety cap on number of steps.
        max_retries : int
            Retries per step on failure.
        max_followup_rounds : int
            Number of mechanism-oriented scientific follow-up rounds to run.
        max_review_rounds : int
            Number of three-reviewer manuscript critique/revision rounds.
        produce_latex : bool
            If an article/ LaTeX template exists under the project, ask CodeEngine
            to convert the Markdown paper and artifacts into a LaTeX draft.
        use_web_search : bool
            Add lightweight online literature clues to follow-up planning.

        Returns
        -------
        dict  with keys: success, summary, figures, output_files, ...
        """
        cb = progress_cb or (lambda x: None)
        self.memory.goal = goal

        # Output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.memory.output_dir = output_dir
        else:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix="sage_agent_")
            self.memory.output_dir = output_dir

        cb(f"\n🤖 SeismoAgent 启动")
        cb(f"   目标：{goal}")
        cb(f"   输出目录：{output_dir}\n")

        # Check LLM
        if not self.is_llm_available():
            return {
                "success": False,
                "summary": (
                    "⚠️  LLM 服务不可用。\n"
                    "请启动 Ollama（`ollama serve`）或配置 API（`python seismic_cli.py llm setup`）后重试。"
                ),
                "figures": [],
            }

        # Step 1: Load paper(s)
        if paper_source:
            if isinstance(paper_source, (list, tuple, set)):
                paper_sources = [str(s) for s in paper_source if str(s).strip()]
            else:
                paper_sources = [str(paper_source)]
            cb(f"   检测到 {len(paper_sources)} 个文献输入")
            loaded = 0
            for src in paper_sources:
                if self.load_paper(src, cb):
                    loaded += 1
            if loaded == 0:
                cb("   ⚠ 文献均未能加载，将基于文件画像和目标继续规划")
        else:
            cb("   （无文献输入，根据目标直接规划）")

        # Step 2: Extract key methods from paper
        paper_context = self.memory.get_paper_context()
        method_summary = ""
        if paper_context:
            cb("\n🔍 理解文献方法...")
            try:
                method_summary = _explain_paper_methods(paper_context, goal, self.llm_config)
                if method_summary:
                    cb(f"   核心方法摘要：\n{method_summary[:400]}{'...' if len(method_summary) > 400 else ''}")
            except Exception as e:
                cb(f"   ⚠  方法提取失败（{e}），继续规划...")

        all_figures: List[str] = []
        all_output_files: List[str] = []
        followup_questions: List[str] = []
        latex_path = ""
        latex_bib_path = ""
        latex_pdf_path = ""

        # Step 3: Data reconnaissance before scientific planning.
        # This restores the "inspect first, then plan/code" behavior: the agent
        # must read data notes and infer schemas before it asks for figures.
        effective_context = method_summary or paper_context[:3000]
        cb("\n🔎 数据结构侦察...")
        try:
            recon = _write_builtin_reconnaissance(self.project_root, output_dir, goal)
            recon_files = recon.get("output_files", [])
            all_output_files.extend([f for f in recon_files if f not in all_output_files])
            self.memory.record_step(StepResult(
                step_index=len(self.memory.step_results) + 1,
                description="数据结构侦察：内置扫描器读取数据说明、推断数据结构并写入步骤输入输出契约",
                code="# Built-in deterministic data reconnaissance; no LLM-generated code.",
                stdout=recon.get("stdout", ""),
                figures=[],
                output_files=recon_files,
                success=True,
                error="",
            ))
            cb(
                "   ✓ 数据结构侦察完成："
                f"扫描 {recon.get('scanned_files', 0)} 个文件，"
                f"候选数据 {recon.get('candidate_data_files', 0)} 个，"
                f"说明文档 {recon.get('data_notes', 0)} 个，"
                f"文献 {recon.get('references', 0)} 个"
            )
        except Exception as exc:
            cb(f"   ⚠ 数据结构侦察失败：{exc}。将退回文件画像和文献上下文继续规划。")

        effective_context = "\n\n".join([
            effective_context,
            "===== Data reconnaissance / IO contracts =====",
            self.memory.accumulated_context(max_chars=7000),
        ]).strip()

        # Step 4: Plan
        cb("\n📋 规划执行步骤...")
        steps = self.planner.plan(goal=goal, paper_context=effective_context)
        steps = steps[:max_steps]
        self.memory.plan = [s.description for s in steps]

        cb(f"   共 {len(steps)} 个步骤：")
        for s in steps:
            cb(f"   {s.index}. [{s.step_type}] {s.description}")

        cb("\n🧭 规划论文图件与表格...")
        artifact_plan = _plan_paper_artifacts(
            goal=goal,
            method_summary=effective_context,
            memory_context=self.memory.accumulated_context(max_chars=9000),
            llm_config=self.llm_config,
        )
        if artifact_plan:
            self.memory.notes.append("论文图表规划：\n" + artifact_plan)
            cb(f"   图表规划：\n{artifact_plan[:800]}{'...' if len(artifact_plan) > 800 else ''}")

        # Step 5: Execute steps
        cb("\n⚙️  开始执行...\n")
        completed_steps: List[PlanStep] = []

        if artifact_plan:
            cb("── 图表规划执行: 先按 LLM 图表/统计计划编程生成候选证据池")
            try:
                from seismo_code.code_engine import CodeEngine

                engine = CodeEngine(
                    llm_config=self.llm_config,
                    project_root=self.project_root,
                    python_executable=self.llm_config.get("python_executable"),
                )
                pre_artifact_request = "\n".join([
                    "You are the first coding stage of Scientific Analysis Agent.",
                    "Before ordinary analysis steps, execute the LLM-planned figure/table/statistics plan.",
                    "The purpose is to create a broad candidate evidence pool first; later stages will inspect the figures and converge on new scientific questions.",
                    "",
                    "Research goal:",
                    goal,
                    "",
                    "LLM figure/table/statistics plan to execute:",
                    artifact_plan,
                    "",
                    "Paper/literature method context:",
                    effective_context[:7000] or "(not available)",
                    "",
                    "Data reconnaissance and IO-contract context:",
                    self.memory.accumulated_context(max_chars=9000) or "(none)",
                    "",
                    f"Project root: {self.project_root}",
                    f"Output directory: {output_dir}",
                    "",
                    "Task:",
                    _SCIENCE_FIGURE_POLICY,
                    "- Traverse the project directory and inspect real data, notes, schema manifests, and prior artifacts before assuming schemas.",
                    "- If data_reconnaissance.md, data_schema_manifest.json, or io_contracts.md exist, read them first and use them as the source of truth unless raw data contradicts them.",
                    "- For this coding stage, write or update step_io_contracts.md with: input files, inferred fields used, output files, smoke-test checks, and evidence status.",
                    "- Implement the planned candidate figures/tables as far as current data support them.",
                    "- Generate a broad candidate pool first: target 6-10 figures and 3-5 tables/Markdown notes when the data support it.",
                    "- Each artifact must be tied to a research-region/data pattern and a testable scientific question.",
                    "- Create figure_table_plan_execution.md mapping planned artifact -> code/data used -> output path -> evidence status.",
                    "- If an artifact cannot be generated, record why in missing_information.md instead of inventing data.",
                    "- Print all created artifact paths and at least one `[SAGE_TEST]` line confirming non-empty parsed data plus figure/table counts.",
                ])

                def _pre_artifact_progress(d):
                    msg = d.get("message", "") if isinstance(d, dict) else str(d)
                    phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                    attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                    if msg:
                        cb(f"   [Artifact Plan CodeEngine:{phase}/{attempt}] {msg}")

                pre_artifact_run = engine.run(
                    pre_artifact_request,
                    max_debug_rounds=max_retries + 3,
                    timeout=300,
                    run_verify=False,
                    on_progress=_pre_artifact_progress,
                    output_dir=output_dir,
                )
                if pre_artifact_run.exec_result:
                    pre_figs = pre_artifact_run.exec_result.figures or []
                    pre_files = pre_artifact_run.exec_result.output_files or []
                    all_figures.extend([f for f in pre_figs if f not in all_figures])
                    all_output_files.extend([f for f in pre_files if f not in all_output_files])
                    self.memory.record_step(StepResult(
                        step_index=len(self.memory.step_results) + 1,
                        description="图表规划执行：按 LLM 规划先生成候选图表/统计证据池",
                        code=pre_artifact_run.code,
                        stdout=pre_artifact_run.stdout,
                        figures=pre_figs,
                        output_files=pre_files,
                        success=pre_artifact_run.success,
                        error=(pre_artifact_run.exec_result.error if not pre_artifact_run.success else ""),
                    ))
                    cb(f"   ✓ 图表规划执行完成：候选图件 {len(pre_figs)}，文件 {len(pre_files)}")
                else:
                    cb(f"   ⚠ 图表规划执行未产生执行结果：{pre_artifact_run.response[:240]}")
            except Exception as exc:
                cb(f"   ⚠ 图表规划执行失败：{exc}")
            cb("")

        for step in steps:
            cb(f"── 步骤 {step.index}/{len(steps)}: {step.description}")
            runtime_guidance = guidance_provider() if callable(guidance_provider) else ""
            step_goal = goal
            if runtime_guidance:
                cb("   ↪ 已接收运行中补充引导，将用于本步骤。")
                step_goal = (
                    goal
                    + "\n\n===== Runtime user guidance =====\n"
                    + runtime_guidance[-4000:]
                )

            if step.step_type == "qa":
                # QA step — just log, no execution
                result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    stdout=f"（说明步骤，无需执行代码）",
                    success=True,
                )
                self.memory.record_step(result)
                completed_steps.append(step)
                cb(f"   ✓ 说明步骤，跳过执行\n")
                continue

            # Generate, execute, and debug code through the shared CodeEngine.
            # This keeps scientific analysis on the same stronger coding path
            # used by chat/code tasks: RAG + SKILL context, pre-sanitization,
            # mini tests, syntax/runtime debugging, and artifact collection.
            try:
                from seismo_code.code_engine import CodeEngine

                engine = CodeEngine(
                    llm_config=self.llm_config,
                    project_root=self.project_root,
                    python_executable=self.llm_config.get("python_executable"),
                )

                code_request = "\n".join([
                    f"Overall research goal:\n{step_goal}",
                    "",
                    f"Current autonomous science step {step.index}/{len(steps)}:",
                    step.description,
                    "",
                    f"Expected output:\n{step.expected_output}",
                    "",
                    "Paper/literature method context:",
                    effective_context[:5000] or "(not available)",
                    "",
                    "LLM-planned paper figures and tables:",
                    artifact_plan[:5000] or "(not available)",
                    "",
                    "Prior step memory and available artifacts:",
                    self.memory.accumulated_context(max_chars=9000) or "(none)",
                    "",
                    f"Project root: {self.project_root}",
                    f"All outputs for this step must be written under: {output_dir}",
                    "",
                    "Coding requirements:",
                    _SCIENCE_FIGURE_POLICY,
                    "- Read data_reconnaissance.md, data_schema_manifest.json, and io_contracts.md from the output directory if they exist.",
                    "- Inspect real files, delimiters, columns, and row examples before assuming schemas; if the manifest disagrees with raw samples, update the manifest or write a correction note.",
                    "- At the start of the script, print a concise input contract: input files, required columns/fields, and expected outputs.",
                    "- At the end of the script, print a concise output contract: created files, non-empty checks, figure/table counts, and any missing evidence.",
                    "- Write or update step_io_contracts.md for this step with the verified input/output contract.",
                    "- Include a small smoke test: print shapes, columns, non-empty checks, and output paths.",
                    "- Print at least one `[SAGE_TEST]` line that confirms non-empty parsed data and lists created artifacts.",
                    "- Reuse validated CSV/JSON/artifacts from prior steps when available.",
                    "- If data format is unclear, write an inspection artifact and continue with the safest verified interpretation.",
                    "- In early/exploratory steps, prefer a broad candidate artifact set over a single minimal plot; later steps may converge and select main figures.",
                    "- Do not invent data, citations, or results; label missing evidence explicitly.",
                ])

                def _code_progress(d):
                    msg = d.get("message", "") if isinstance(d, dict) else str(d)
                    phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                    attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                    if msg:
                        cb(f"   [CodeEngine:{phase}/{attempt}] {msg}")

                code_run = engine.run(
                    code_request,
                    max_debug_rounds=max_retries + 2,
                    timeout=180,
                    run_verify=False,
                    on_progress=_code_progress,
                    output_dir=output_dir,
                )
                exec_result = code_run.exec_result
                code = code_run.code
                if not code_run.success:
                    cb(f"   ⚠  CodeEngine 未完全通过：{code_run.response[:240]}")
            except Exception as e:
                exec_result = None
                code = ""
                cb(f"   ✗ CodeEngine 执行失败：{e}")

            if exec_result is None:
                step_result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    success=False,
                    error="CodeEngine 执行失败",
                )
            else:
                # Copy figures to output_dir
                step_figs = []
                for fig in exec_result.figures:
                    dst = os.path.join(output_dir, os.path.basename(fig))
                    try:
                        if fig != dst:
                            shutil.copy2(fig, dst)
                        step_figs.append(dst)
                    except Exception:
                        step_figs.append(fig)

                step_result = StepResult(
                    step_index=step.index,
                    description=step.description,
                    code=code,
                    stdout=exec_result.stdout,
                    figures=step_figs,
                    output_files=exec_result.output_files,
                    success=exec_result.success,
                    error=exec_result.error,
                )

            self.memory.record_step(step_result)
            all_figures.extend(step_result.figures)
            all_output_files.extend(step_result.output_files)

            # Status
            if step_result.success:
                out_preview = step_result.stdout.strip()[:120] if step_result.stdout.strip() else ""
                cb(f"   ✓ 完成" + (f"\n   输出: {out_preview}" if out_preview else ""))
                if step_result.figures:
                    cb(f"   图像: {[os.path.basename(f) for f in step_result.figures]}")
                completed_steps.append(step)
            else:
                cb(f"   ✗ 步骤失败: {step_result.error[:100]}")
            cb("")

        # Broad exploration safety net. The planner may under-specify code steps or
        # an early CodeEngine failure may leave the paper with too little evidence.
        # Before synthesis, force one publication-oriented exploration pass that
        # inspects real files and creates a candidate figure/table pool. Later
        # refinement/reviewer stages decide which artifacts belong in the main text.
        table_paths = _table_artifacts(all_output_files)
        if len(all_figures) < 5 or len(table_paths) < 2:
            cb("── 科学探索补全: 候选图表不足，进入广泛数据探索与论文证据生成")
            try:
                from seismo_code.code_engine import CodeEngine

                engine = CodeEngine(
                    llm_config=self.llm_config,
                    project_root=self.project_root,
                    python_executable=self.llm_config.get("python_executable"),
                )
                exploration_request = "\n".join([
                    "You are the exploratory coding scientist for Scientific Analysis Agent.",
                    "The current run has too few figures/tables to support a real scientific paper.",
                    "Do a broad but grounded exploration first; later stages will converge and select main-text artifacts.",
                    "",
                    "Research goal:",
                    goal,
                    "",
                    "Paper/literature method context:",
                    effective_context[:7000] or "(not available)",
                    "",
                    "LLM-planned paper figures and tables:",
                    artifact_plan or "(not available)",
                    "",
                    "Prior memory and artifacts:",
                    self.memory.accumulated_context(max_chars=14000) or "(none)",
                    "",
                    "Existing table/statistical previews:",
                    _summarize_table_artifacts(all_output_files) or "(none)",
                    "",
                    f"Project root: {self.project_root}",
                    f"Output directory: {output_dir}",
                    "",
                    "Task:",
                    _SCIENCE_FIGURE_POLICY,
                    "- Read data_reconnaissance.md, data_schema_manifest.json, io_contracts.md, and step_io_contracts.md from the output directory if they exist.",
                    "- Traverse the project directory and inspect data, notes, literature-derived clues, prior CSV/JSON artifacts, and parameter-optimization records if present.",
                    "- Create or update data_inventory.md describing real input files, inferred roles, schemas, row counts, and limitations.",
                    "- For every generated figure/table, record input files, fields used, output path, and validation checks in step_io_contracts.md.",
                    "- Generate a broad candidate artifact pool when data support it: target at least 6 figures and 3 tables/Markdown notes. If fewer are possible, explain why in missing_information.md.",
                    "- Candidate figures should test different mechanism hypotheses, not repeat the same QC distribution: spatial segmentation, depth dependence, along-strike variation, mechanism-class maps, literature-hypothesis comparisons, robustness/sensitivity, and composite summary figures are preferred when supported.",
                    "- Candidate tables should summarize statistics used by the paper, not just file schemas.",
                    "- Create exploration_journal.md mapping each artifact to the scientific question, input files, computation, evidence status, and uncertainty.",
                    "- Print all created artifact paths and at least one `[SAGE_TEST]` line confirming non-empty parsed data plus figure/table counts.",
                    "- Do not invent data, citations, or unsupported claims.",
                ])

                def _explore_progress(d):
                    msg = d.get("message", "") if isinstance(d, dict) else str(d)
                    phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                    attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                    if msg:
                        cb(f"   [Exploration CodeEngine:{phase}/{attempt}] {msg}")

                exploration_run = engine.run(
                    exploration_request,
                    max_debug_rounds=max_retries + 3,
                    timeout=300,
                    run_verify=False,
                    on_progress=_explore_progress,
                    output_dir=output_dir,
                )
                if exploration_run.exec_result:
                    extra_figs = exploration_run.exec_result.figures or []
                    extra_files = exploration_run.exec_result.output_files or []
                    all_figures.extend([f for f in extra_figs if f not in all_figures])
                    all_output_files.extend([f for f in extra_files if f not in all_output_files])
                    self.memory.record_step(StepResult(
                        step_index=len(self.memory.step_results) + 1,
                        description="科学探索补全：广泛遍历数据并生成候选论文图表池",
                        code=exploration_run.code,
                        stdout=exploration_run.stdout,
                        figures=extra_figs,
                        output_files=extra_files,
                        success=exploration_run.success,
                        error=(exploration_run.exec_result.error if not exploration_run.success else ""),
                    ))
                    cb(f"   ✓ 科学探索补全完成：候选图件 {len(extra_figs)}，文件 {len(extra_files)}")
                else:
                    cb(f"   ⚠ 科学探索补全未产生执行结果：{exploration_run.response[:240]}")
            except Exception as exc:
                cb(f"   ⚠ 科学探索补全失败：{exc}")

        # Ensure the final paper has LLM-planned figures/tables as primary evidence.
        table_paths = _table_artifacts(all_output_files)
        if artifact_plan and (len(all_figures) < 3 or len(table_paths) < 1):
            cb("── 论文图表补全: 根据 LLM 图表规划生成缺失图件/表格")
            try:
                from seismo_code.code_engine import CodeEngine

                engine = CodeEngine(
                    llm_config=self.llm_config,
                    project_root=self.project_root,
                    python_executable=self.llm_config.get("python_executable"),
                )
                artifact_request = "\n".join([
                    "You are completing the publication artifact stage for Scientific Analysis Agent.",
                    "Use the LLM-planned figures/tables below to decide what to generate; do not use a fixed template.",
                    "",
                    "Research goal:",
                    goal,
                    "",
                    "LLM-planned paper figures and tables:",
                    artifact_plan,
                    "",
                    "Existing artifacts and step results:",
                    self.memory.accumulated_context(max_chars=12000),
                    "",
                    f"Project root: {self.project_root}",
                    f"Output directory: {output_dir}",
                    "",
                    "Task:",
                    _SCIENCE_FIGURE_POLICY,
                    "- Read data_reconnaissance.md, data_schema_manifest.json, io_contracts.md, and step_io_contracts.md before choosing inputs.",
                    "- Inspect available data and existing artifacts.",
                    "- For each missing planned artifact, state its input contract before coding and append the verified output contract after execution.",
                    "- Generate only missing planned publication figures that directly test the core hypothesis.",
                    "- Generate missing planned statistical tables as CSV and, when useful, Markdown table files.",
                    "- Print a concise mapping: Figure/Table -> source file(s) -> output artifact.",
                    "- Print at least one `[SAGE_TEST]` line confirming generated figure/table counts and non-empty source data.",
                    "- If a planned artifact is unsupported by the data, create a missing_information.md note explaining why.",
                ])

                def _artifact_progress(d):
                    msg = d.get("message", "") if isinstance(d, dict) else str(d)
                    phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                    attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                    if msg:
                        cb(f"   [CodeEngine:{phase}/{attempt}] {msg}")

                artifact_run = engine.run(
                    artifact_request,
                    max_debug_rounds=max_retries + 2,
                    timeout=180,
                    run_verify=False,
                    on_progress=_artifact_progress,
                    output_dir=output_dir,
                )
                if artifact_run.exec_result:
                    extra_figs = artifact_run.exec_result.figures or []
                    extra_files = artifact_run.exec_result.output_files or []
                    all_figures.extend([f for f in extra_figs if f not in all_figures])
                    all_output_files.extend([f for f in extra_files if f not in all_output_files])
                    self.memory.record_step(StepResult(
                        step_index=len(self.memory.step_results) + 1,
                        description="论文图表补全：按 LLM 图表规划生成缺失图件和表格",
                        code=artifact_run.code,
                        stdout=artifact_run.stdout,
                        figures=extra_figs,
                        output_files=extra_files,
                        success=artifact_run.success,
                        error=(artifact_run.exec_result.error if not artifact_run.success else ""),
                    ))
                    cb(f"   ✓ 图表补全完成：图件 {len(extra_figs)}，文件 {len(extra_files)}")
                else:
                    cb(f"   ⚠ 图表补全未产生执行结果：{artifact_run.response[:240]}")
            except Exception as exc:
                cb(f"   ⚠ 图表补全失败：{exc}")

        # Step 5: Scientific follow-up loop. Each round asks the LLM for the next
        # mechanism-oriented question, then lets CodeEngine verify it with code.
        if max_followup_rounds > 0:
            cb("\n🔁 科学追问：基于已有证据进行多轮 CodeEngine 验证...")
        for round_no in range(1, max_followup_rounds + 1):
            memory_context = "\n\n".join([
                self.memory.accumulated_context(max_chars=12000),
                "表格/统计产物预览：\n" + (_summarize_table_artifacts(all_output_files) or "(none)"),
            ])
            rag_query = "\n".join([
                goal,
                artifact_plan[:2500],
                memory_context[:2500],
            ])
            rag_context = _build_agent_rag_context(rag_query, top_k=6, max_chars=5000)
            web_context = _build_agent_web_context(rag_query, max_results=5, max_chars=5000) if use_web_search else ""
            followup = _plan_followup_question(
                goal=goal,
                method_summary=effective_context,
                artifact_plan=artifact_plan,
                memory_context=memory_context,
                rag_context=rag_context,
                web_context=web_context,
                llm_config=self.llm_config,
                round_no=round_no,
            )
            if _no_followup_needed(followup):
                cb(f"   第 {round_no} 轮：无需继续追问（{followup[:180]}）")
                break

            followup_questions.append(followup)
            self.memory.notes.append(f"科学追问第 {round_no} 轮：\n{followup}")
            cb(f"   第 {round_no} 轮追问：\n{followup[:900]}{'...' if len(followup) > 900 else ''}")

            try:
                from seismo_code.code_engine import CodeEngine

                engine = CodeEngine(
                    llm_config=self.llm_config,
                    project_root=self.project_root,
                    python_executable=self.llm_config.get("python_executable"),
                )
                follow_request = "\n".join([
                    "You are the coding verifier for Scientific Analysis Agent follow-up research.",
                    "The LLM has proposed a mechanism-oriented scientific follow-up. Test it with real data and evidence.",
                    "",
                    "Research goal:",
                    goal,
                    "",
                    f"Follow-up round {round_no} plan:",
                    followup,
                    "",
                    "Prior evidence, artifacts, and table previews:",
                    memory_context,
                    "",
                    "RAG/knowledge-base evidence clues:",
                    rag_context,
                    "",
                    "Online literature clues:",
                    web_context or "(web search disabled or unavailable)",
                    "",
                    f"Project root: {self.project_root}",
                    f"Output directory: {output_dir}",
                    "",
                    "Task:",
                    f"- Generate at least one follow-up note: followup_round_{round_no}.md.",
                    "- If possible, generate at most one focused figure and/or one CSV/Markdown table that directly tests the hypothesis.",
                    "- Do not create generic QC/distribution plot collections in a follow-up round.",
                    "- The note must include: question, tested hypothesis, data used, computation, result, evidence status, remaining uncertainty.",
                    "- If the test is impossible with current files, create a missing_information note instead of inventing results.",
                    "- Reuse existing parsed CSV/JSON artifacts when valid; otherwise inspect raw files and create robust parsers.",
                    "- Print at least one `[SAGE_TEST]` line confirming what was tested and which artifacts were created.",
                ])

                def _follow_progress(d):
                    msg = d.get("message", "") if isinstance(d, dict) else str(d)
                    phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                    attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                    if msg:
                        cb(f"   [Follow-up CodeEngine:{phase}/{attempt}] {msg}")

                follow_run = engine.run(
                    follow_request,
                    max_debug_rounds=max_retries + 2,
                    timeout=180,
                    run_verify=False,
                    on_progress=_follow_progress,
                    output_dir=output_dir,
                )
                if follow_run.exec_result:
                    extra_figs = follow_run.exec_result.figures or []
                    extra_files = follow_run.exec_result.output_files or []
                    all_figures.extend([f for f in extra_figs if f not in all_figures])
                    all_output_files.extend([f for f in extra_files if f not in all_output_files])
                    self.memory.record_step(StepResult(
                        step_index=len(self.memory.step_results) + 1,
                        description=f"科学追问第 {round_no} 轮：CodeEngine 计算验证",
                        code=follow_run.code,
                        stdout=follow_run.stdout,
                        figures=extra_figs,
                        output_files=extra_files,
                        success=follow_run.success,
                        error=(follow_run.exec_result.error if not follow_run.success else ""),
                    ))
                    cb(f"   ✓ 第 {round_no} 轮追问验证完成：图件 {len(extra_figs)}，文件 {len(extra_files)}")
                else:
                    cb(f"   ⚠ 第 {round_no} 轮追问没有执行结果：{follow_run.response[:240]}")
            except Exception as exc:
                cb(f"   ⚠ 第 {round_no} 轮追问验证失败：{exc}")

        final_rag_context = ""
        final_web_context = ""
        evidence_synthesis = ""
        artifact_refinement_plan = ""
        try:
            cb("\n🧠 综合数据、图表、本地论文和在线证据，提炼科学结论...")
            final_query = "\n".join([
                goal,
                artifact_plan[:3000],
                self.memory.accumulated_context(max_chars=6000),
                _summarize_table_artifacts(all_output_files, max_chars=3000),
                "\n\n".join(followup_questions)[-3000:],
            ])
            final_rag_context = _build_agent_rag_context(final_query, top_k=8, max_chars=8000)
            final_web_context = _build_agent_web_context(final_query, max_results=6, max_chars=8000) if use_web_search else ""
            evidence_synthesis = _synthesize_scientific_claims(
                goal=goal,
                method_summary=method_summary or effective_context,
                artifact_plan=artifact_plan,
                memory_context=self.memory.accumulated_context(max_chars=12000),
                figures=all_figures,
                output_files=all_output_files,
                rag_context=final_rag_context,
                web_context=final_web_context,
                followup_questions=followup_questions,
                llm_config=self.llm_config,
            )
            if evidence_synthesis:
                synthesis_path = Path(output_dir) / "scientific_evidence_synthesis.md"
                synthesis_path.write_text(evidence_synthesis, encoding="utf-8")
                if str(synthesis_path) not in all_output_files:
                    all_output_files.append(str(synthesis_path))
                self.memory.notes.append("最终科学证据综合：\n" + evidence_synthesis)
                cb(f"   ✓ 科学证据综合完成: {synthesis_path}")
        except Exception as exc:
            cb(f"   ⚠ 科学证据综合失败：{exc}")

        if evidence_synthesis:
            try:
                cb("\n🧪 图表复审：根据已形成结论反向检查需要增加、降级或移除的图表...")
                artifact_refinement_plan = _plan_artifact_refinement(
                    goal=goal,
                    artifact_plan=artifact_plan,
                    evidence_synthesis=evidence_synthesis,
                    figures=all_figures,
                    output_files=all_output_files,
                    llm_config=self.llm_config,
                )
                if artifact_refinement_plan:
                    refine_path = Path(output_dir) / "artifact_refinement_plan.md"
                    refine_path.write_text(artifact_refinement_plan, encoding="utf-8")
                    if str(refine_path) not in all_output_files:
                        all_output_files.append(str(refine_path))
                    self.memory.notes.append("结论驱动图表复审：\n" + artifact_refinement_plan)
                    cb(f"   ✓ 图表复审完成: {refine_path}")

                if artifact_refinement_plan and not _no_artifact_change_needed(artifact_refinement_plan):
                    cb("   ↪ 图表复审建议调整，调用 CodeEngine 补充关键图表/主文选择清单...")
                    try:
                        from seismo_code.code_engine import CodeEngine

                        engine = CodeEngine(
                            llm_config=self.llm_config,
                            project_root=self.project_root,
                            python_executable=self.llm_config.get("python_executable"),
                        )
                        refine_request = "\n".join([
                            "You are refining publication artifacts after the scientific conclusions are known.",
                            "Do not delete existing files. Demoted figures should be listed as supplementary, not removed from disk.",
                            "",
                            "Research goal:",
                            goal,
                            "",
                            "Scientific evidence synthesis:",
                            evidence_synthesis,
                            "",
                            "Artifact refinement plan:",
                            artifact_refinement_plan,
                            "",
                            "Existing analysis memory and artifacts:",
                            self.memory.accumulated_context(max_chars=14000),
                            "",
                            "Table/statistical previews:",
                            _summarize_table_artifacts(all_output_files) or "(none)",
                            "",
                            "Existing figures:",
                            "\n".join(all_figures) or "(none)",
                            "",
                            f"Project root: {self.project_root}",
                            f"Output directory: {output_dir}",
                            "",
                            "Task:",
                            "- Generate only the ADD_NEEDED artifacts from the refinement plan that can be supported by current data.",
                            "- Create main_artifact_selection.md listing KEEP_MAIN, DEMOTE_SUPPLEMENT, REMOVE_FROM_ARGUMENT, and ADD_NEEDED outcomes.",
                            "- If a requested artifact is unsupported, write why in main_artifact_selection.md or missing_information.md.",
                            "- At most two new figures and one new table. Prefer composite evidence figures over many diagnostic plots.",
                            "- Print created file paths and the final main-figure/table set.",
                            "- Print at least one `[SAGE_TEST]` line confirming the refinement artifact and any new figure/table counts.",
                        ])

                        def _refine_progress(d):
                            msg = d.get("message", "") if isinstance(d, dict) else str(d)
                            phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                            attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                            if msg:
                                cb(f"   [Artifact Review CodeEngine:{phase}/{attempt}] {msg}")

                        refine_run = engine.run(
                            refine_request,
                            max_debug_rounds=max_retries + 2,
                            timeout=180,
                            run_verify=False,
                            on_progress=_refine_progress,
                            output_dir=output_dir,
                        )
                        if refine_run.exec_result:
                            extra_figs = refine_run.exec_result.figures or []
                            extra_files = refine_run.exec_result.output_files or []
                            all_figures.extend([f for f in extra_figs if f not in all_figures])
                            all_output_files.extend([f for f in extra_files if f not in all_output_files])
                            self.memory.record_step(StepResult(
                                step_index=len(self.memory.step_results) + 1,
                                description="结论驱动图表复审：补充缺失图表并生成主文/补充图表选择清单",
                                code=refine_run.code,
                                stdout=refine_run.stdout,
                                figures=extra_figs,
                                output_files=extra_files,
                                success=refine_run.success,
                                error=(refine_run.exec_result.error if not refine_run.success else ""),
                            ))
                            cb(f"   ✓ 图表复审执行完成：新增图件 {len(extra_figs)}，文件 {len(extra_files)}")
                        else:
                            cb(f"   ⚠ 图表复审没有执行结果：{refine_run.response[:240]}")
                    except Exception as exc:
                        cb(f"   ⚠ 图表复审执行失败：{exc}")

                    cb("   ↪ 根据复审后的图表集合重新综合科学证据...")
                    evidence_synthesis = _synthesize_scientific_claims(
                        goal=goal,
                        method_summary=method_summary or effective_context,
                        artifact_plan=(artifact_plan + "\n\n===== Artifact refinement plan =====\n" + artifact_refinement_plan),
                        memory_context=self.memory.accumulated_context(max_chars=14000),
                        figures=all_figures,
                        output_files=all_output_files,
                        rag_context=final_rag_context,
                        web_context=final_web_context,
                        followup_questions=followup_questions,
                        llm_config=self.llm_config,
                    )
                    if evidence_synthesis:
                        synthesis_path = Path(output_dir) / "scientific_evidence_synthesis.md"
                        synthesis_path.write_text(evidence_synthesis, encoding="utf-8")
                        self.memory.notes.append("图表复审后的最终科学证据综合：\n" + evidence_synthesis)
                        cb("   ✓ 已完成复审后的科学证据再综合")
            except Exception as exc:
                cb(f"   ⚠ 图表复审失败：{exc}")

        markdown_paper_path = ""
        try:
            cb("\n📝 基于图件、统计结果和文献摘要撰写 Markdown 论文草稿...")
            markdown_paper_path = _write_markdown_paper(
                goal=goal,
                method_summary=method_summary,
                memory=self.memory,
                figures=all_figures,
                output_files=all_output_files,
                output_dir=output_dir,
                llm_config=self.llm_config,
                artifact_plan=(artifact_plan + "\n\n===== Conclusion-driven artifact refinement =====\n" + artifact_refinement_plan),
                evidence_synthesis=evidence_synthesis,
                rag_context=final_rag_context,
                web_context=final_web_context,
                followup_questions=followup_questions,
            )
            all_output_files.append(markdown_paper_path)
            cb(f"   ✓ Markdown 论文草稿: {markdown_paper_path}")
        except Exception as e:
            cb(f"   ⚠ Markdown 论文草稿生成失败：{e}")

        review_reports: List[str] = []
        if markdown_paper_path and max_review_rounds > 0:
            cb("\n🧐 三审稿人内部评审：循环修改直到小修或达到轮次上限...")
            for review_round in range(1, max_review_rounds + 1):
                try:
                    draft_text = Path(markdown_paper_path).read_text(encoding="utf-8", errors="ignore")
                    review_text = _three_reviewer_review(
                        goal=goal,
                        draft_text=draft_text,
                        evidence_synthesis=evidence_synthesis,
                        artifact_refinement_plan=artifact_refinement_plan,
                        figures=all_figures,
                        output_files=all_output_files,
                        rag_context=final_rag_context,
                        web_context=final_web_context,
                        llm_config=self.llm_config,
                        round_no=review_round,
                    )
                    review_path = Path(output_dir) / f"peer_review_round_{review_round}.md"
                    review_path.write_text(review_text, encoding="utf-8")
                    review_reports.append(str(review_path))
                    if str(review_path) not in all_output_files:
                        all_output_files.append(str(review_path))
                    self.memory.notes.append(f"三审稿人评审第 {review_round} 轮：\n{review_text}")
                    cb(f"   ✓ 第 {review_round} 轮审稿完成: {review_path}")

                    if _review_is_minor_or_better(review_text):
                        cb(f"   ✓ 三位审稿人已达到小修/接收状态，停止评审循环。")
                        break

                    if _review_requests_more_analysis(review_text):
                        cb("   ↪ 审稿意见要求补充证据，调用 CodeEngine 做最多一轮针对性补充分析...")
                        try:
                            from seismo_code.code_engine import CodeEngine

                            engine = CodeEngine(
                                llm_config=self.llm_config,
                                project_root=self.project_root,
                                python_executable=self.llm_config.get("python_executable"),
                            )
                            review_code_request = "\n".join([
                                "You are addressing peer-review comments for Scientific Analysis Agent.",
                                "Run only targeted additional analyses needed by the review; do not make generic QC plots.",
                                "",
                                "Research goal:",
                                goal,
                                "",
                                "Reviewer comments:",
                                review_text,
                                "",
                                "Current evidence synthesis:",
                                evidence_synthesis,
                                "",
                                "Existing artifacts and memory:",
                                self.memory.accumulated_context(max_chars=14000),
                                "",
                                "Table/statistical previews:",
                                _summarize_table_artifacts(all_output_files) or "(none)",
                                "",
                                "Existing figures:",
                                "\n".join(all_figures) or "(none)",
                                "",
                                f"Project root: {self.project_root}",
                                f"Output directory: {output_dir}",
                                "",
                                "Task:",
                                "- Generate at most one targeted figure and one targeted table/note that directly answers the reviewers.",
                                "- Create reviewer_response_analysis.md summarizing what was computed, what changed, and what remains unsupported.",
                                "- If the requested evidence cannot be produced from current data, explain that clearly instead of inventing results.",
                                "- Print all created file paths.",
                                "- Print at least one `[SAGE_TEST]` line confirming reviewer-response artifacts.",
                            ])

                            def _review_code_progress(d):
                                msg = d.get("message", "") if isinstance(d, dict) else str(d)
                                phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                                attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                                if msg:
                                    cb(f"   [Peer Review CodeEngine:{phase}/{attempt}] {msg}")

                            review_run = engine.run(
                                review_code_request,
                                max_debug_rounds=max_retries + 2,
                                timeout=180,
                                run_verify=False,
                                on_progress=_review_code_progress,
                                output_dir=output_dir,
                            )
                            if review_run.exec_result:
                                extra_figs = review_run.exec_result.figures or []
                                extra_files = review_run.exec_result.output_files or []
                                all_figures.extend([f for f in extra_figs if f not in all_figures])
                                all_output_files.extend([f for f in extra_files if f not in all_output_files])
                                self.memory.record_step(StepResult(
                                    step_index=len(self.memory.step_results) + 1,
                                    description=f"三审稿人第 {review_round} 轮：按审稿意见补充针对性分析",
                                    code=review_run.code,
                                    stdout=review_run.stdout,
                                    figures=extra_figs,
                                    output_files=extra_files,
                                    success=review_run.success,
                                    error=(review_run.exec_result.error if not review_run.success else ""),
                                ))
                                cb(f"   ✓ 审稿补充分析完成：新增图件 {len(extra_figs)}，文件 {len(extra_files)}")
                        except Exception as exc:
                            cb(f"   ⚠ 审稿补充分析失败：{exc}")

                        evidence_synthesis = _synthesize_scientific_claims(
                            goal=goal,
                            method_summary=method_summary or effective_context,
                            artifact_plan=(artifact_plan + "\n\n===== Artifact refinement plan =====\n" + artifact_refinement_plan),
                            memory_context=self.memory.accumulated_context(max_chars=14000),
                            figures=all_figures,
                            output_files=all_output_files,
                            rag_context=final_rag_context,
                            web_context=final_web_context,
                            followup_questions=followup_questions,
                            llm_config=self.llm_config,
                        )
                        synthesis_path = Path(output_dir) / "scientific_evidence_synthesis.md"
                        synthesis_path.write_text(evidence_synthesis, encoding="utf-8")

                    revised_text = _revise_markdown_paper_with_reviews(
                        goal=goal,
                        draft_text=Path(markdown_paper_path).read_text(encoding="utf-8", errors="ignore"),
                        review_text=review_text,
                        evidence_synthesis=evidence_synthesis,
                        artifact_refinement_plan=artifact_refinement_plan,
                        figures=all_figures,
                        output_files=all_output_files,
                        llm_config=self.llm_config,
                    )
                    revision_path = Path(output_dir) / f"science_paper_revision_round_{review_round}.md"
                    revision_path.write_text(revised_text, encoding="utf-8")
                    Path(markdown_paper_path).write_text(revised_text, encoding="utf-8")
                    if str(revision_path) not in all_output_files:
                        all_output_files.append(str(revision_path))
                    cb(f"   ✓ 已按第 {review_round} 轮审稿意见修订论文: {revision_path}")
                except Exception as exc:
                    cb(f"   ⚠ 第 {review_round} 轮审稿/修订失败：{exc}")
                    break

        if produce_latex:
            article_dir = Path(self.project_root) / "article"
            template_candidates = []
            if article_dir.exists():
                preferred = article_dir / "maduo_mechanism_agujournal.tex"
                if preferred.exists():
                    template_candidates.append(preferred)
                template_candidates.extend(sorted(p for p in article_dir.glob("*.tex") if p not in template_candidates))
            if template_candidates:
                template_path = template_candidates[0]
                cb("\n📄 使用 article 模板生成 LaTeX 论文草稿...")
                try:
                    from seismo_code.code_engine import CodeEngine

                    engine = CodeEngine(
                        llm_config=self.llm_config,
                        project_root=self.project_root,
                        python_executable=self.llm_config.get("python_executable"),
                    )
                    latex_request = "\n".join([
                        "You are preparing a submission-style LaTeX manuscript directly from the project evidence.",
                        "The local article/ folder contains the manuscript template, AGU class/style files, and bib files.",
                        "Use that template as the primary writing target, not as a loose reference.",
                        "Write all new outputs inside SAGE_OUTDIR only.",
                        "",
                        f"Markdown paper path, if available: {markdown_paper_path or '(not generated; write directly from evidence)'}",
                        f"Article template path: {template_path}",
                        f"Article directory with cls/sty/bib files: {article_dir}",
                        f"Output directory: {output_dir}",
                        "",
                        "Research goal:",
                        goal,
                        "",
                        "Method/literature summary:",
                        method_summary or effective_context or "(not available)",
                        "",
                        "LLM-planned paper figures and tables:",
                        artifact_plan or "(not available)",
                        "",
                        "Conclusion-driven artifact refinement plan:",
                        artifact_refinement_plan or "(not available)",
                        "",
                        "Scientific follow-up questions and tested hypotheses:",
                        "\n\n---\n\n".join(followup_questions) or "(none)",
                        "",
                        "Final claim-evidence synthesis brief:",
                        evidence_synthesis or "(not available)",
                        "",
                        "RAG/local evidence excerpts:",
                        final_rag_context or "(not available)",
                        "",
                        "Online literature evidence excerpts:",
                        final_web_context or "(not available)",
                        "",
                        "Internal peer-review reports:",
                        "\n".join(review_reports) or "(none)",
                        "",
                        "Step memory and evidence summary:",
                        self.memory.accumulated_context(max_chars=14000),
                        "",
                        "Table/statistical artifact previews:",
                        _summarize_table_artifacts(all_output_files) or "(none)",
                        "",
                        "Available figures:",
                        "\n".join(all_figures) or "(none)",
                        "",
                        "Available statistical/table artifacts:",
                        "\n".join(_table_artifacts(all_output_files)) or "(none)",
                        "",
                        "Task:",
                        "- Create a complete science_article.tex in SAGE_OUTDIR by adapting the local template structure.",
                        "- Copy the required .cls/.sty files from article/ into SAGE_OUTDIR so the manuscript can compile there.",
                        "- Create science_article_refs.bib in SAGE_OUTDIR. Reuse entries from article/*.bib when relevant.",
                        "- Cite only keys that exist in science_article_refs.bib. Never invent fake citations.",
                        "- Include generated figures with valid relative paths and captions tied to evidence.",
                        "- Include key statistical tables directly in LaTeX when they support the scientific argument.",
                        "- Write a mechanism-oriented title and paper, not a data-quality report.",
                        "- Explicitly mark unsupported claims as limitations or future tests.",
                        "- If latexmk/pdflatex is available, compile to science_article.pdf in SAGE_OUTDIR; otherwise create compile_instructions.md.",
                        "- Print all created file paths.",
                    ])

                    def _latex_progress(d):
                        msg = d.get("message", "") if isinstance(d, dict) else str(d)
                        phase = d.get("phase", "code") if isinstance(d, dict) else "code"
                        attempt = d.get("attempt", 0) if isinstance(d, dict) else 0
                        if msg:
                            cb(f"   [LaTeX CodeEngine:{phase}/{attempt}] {msg}")

                    latex_run = engine.run(
                        latex_request,
                        max_debug_rounds=max_retries + 2,
                        timeout=180,
                        run_verify=False,
                        on_progress=_latex_progress,
                        output_dir=output_dir,
                    )
                    if latex_run.exec_result:
                        latex_files = latex_run.exec_result.output_files or []
                        all_output_files.extend([f for f in latex_files if f not in all_output_files])
                        search_root = Path(output_dir)
                        preferred_tex = search_root / "science_article.tex"
                        preferred_bib = search_root / "science_article_refs.bib"
                        preferred_pdf = search_root / "science_article.pdf"
                        tex_files = sorted(search_root.rglob("*.tex"), key=lambda p: p.stat().st_mtime, reverse=True)
                        bib_files = sorted(search_root.rglob("*.bib"), key=lambda p: p.stat().st_mtime, reverse=True)
                        pdf_files = sorted(search_root.rglob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
                        latex_path = str(preferred_tex if preferred_tex.exists() else (tex_files[0] if tex_files else ""))
                        latex_bib_path = str(preferred_bib if preferred_bib.exists() else (bib_files[0] if bib_files else ""))
                        latex_pdf_path = str(preferred_pdf if preferred_pdf.exists() else (pdf_files[0] if pdf_files else ""))
                        self.memory.record_step(StepResult(
                            step_index=len(self.memory.step_results) + 1,
                            description="LaTeX 论文草稿生成：基于 article 模板和 Markdown/图表产物",
                            code=latex_run.code,
                            stdout=latex_run.stdout,
                            output_files=latex_files,
                            success=latex_run.success,
                            error=(latex_run.exec_result.error if not latex_run.success else ""),
                        ))
                        cb(f"   ✓ LaTeX 产物: {latex_path or '(no tex)'}")
                except Exception as exc:
                    cb(f"   ⚠ LaTeX 论文生成失败：{exc}")

        n_ok = sum(1 for r in self.memory.step_results if r.success)
        n_total = len(self.memory.step_results)
        overall_success = n_ok == n_total

        summary_lines = [
            f"\n{'✅' if overall_success else '⚠️ '} Agent 执行完成",
            f"   步骤完成: {n_ok}/{n_total}",
        ]
        if all_figures:
            summary_lines.append(f"   生成图像: {len(all_figures)} 张")
            for f in all_figures:
                summary_lines.append(f"     • {f}")
        if all_output_files:
            summary_lines.append(f"   生成文件: {len(all_output_files)} 个")
        summary_lines.append(f"   输出目录: {output_dir}")

        # Detailed step log
        summary_lines.append("\n执行日志:")
        for r in self.memory.step_results:
            summary_lines.append(r.brief())

        summary = "\n".join(summary_lines)
        cb(summary)

        return {
            "success": overall_success,
            "summary": summary,
            "figures": all_figures,
            "output_files": all_output_files,
            "steps_completed": n_ok,
            "steps_total": n_total,
            "output_dir": output_dir,
            "method_summary": method_summary,
            "paper_artifact_plan": artifact_plan,
            "scientific_evidence_synthesis": evidence_synthesis,
            "artifact_refinement_plan": artifact_refinement_plan,
            "peer_review_reports": review_reports,
            "scientific_questions": followup_questions,
            "table_artifacts": _table_artifacts(all_output_files),
            "statistical_results": [
                {"title": "论文图表规划", "content": artifact_plan},
                {"title": "科学证据综合", "content": evidence_synthesis},
                {"title": "结论驱动图表复审", "content": artifact_refinement_plan},
                {"title": "三审稿人内部评审", "content": "\n".join(review_reports)},
                {"title": "科学追问", "content": "\n\n---\n\n".join(followup_questions)},
                {"title": "表格/统计产物", "content": _summarize_table_artifacts(all_output_files)},
            ],
            "markdown_paper_path": markdown_paper_path,
            "markdown_paper": Path(markdown_paper_path).read_text(encoding="utf-8") if markdown_paper_path else "",
            "latex_path": latex_path,
            "latex_bib_path": latex_bib_path,
            "latex_pdf": latex_pdf_path,
            "latex_paper": Path(latex_path).read_text(encoding="utf-8", errors="ignore") if latex_path else "",
        }
