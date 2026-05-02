"""
ce_utils.py — Pure utility functions and data classes for the code engine.

All functions here are side-effect-free (except _profile_file which runs a
subprocess) and can be imported / tested independently of CodeEngine.

Public API
----------
Data classes : DebugAttempt, CodeRunResult, StepResult, WorkflowRunResult
Functions    : _call_llm, _extract_code, _is_bash_code, _extract_diagnosis,
               _extract_plan, _find_file_paths, _profile_file,
               _format_file_context, _pre_sanitize
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DebugAttempt:
    attempt:   int
    diagnosis: str
    code:      str
    error:     str
    stdout:    str
    success:   bool


@dataclass
class CodeRunResult:
    success:     bool
    response:    str
    code:        str
    exec_result: Optional[Any]
    attempts:    int = 1
    debug_trace: List[DebugAttempt] = field(default_factory=list)
    verify_pass: Optional[bool] = None
    verify_note: str = ""
    plan:        List[str] = field(default_factory=list)
    script_path: str = ""

    @property
    def figures(self) -> List[str]:
        return self.exec_result.figures if self.exec_result else []

    @property
    def output_files(self) -> List[str]:
        return self.exec_result.output_files if self.exec_result else []

    @property
    def stdout(self) -> str:
        return self.exec_result.stdout if self.exec_result else ""


@dataclass
class StepResult:
    """Result of executing a single workflow step."""
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


@dataclass
class WorkflowRunResult:
    """Aggregate result of a multi-step workflow run."""
    workflow_name:    str
    workflow_title:   str
    success:          bool
    steps_total:      int
    steps_done:       int
    step_results:     List[StepResult]
    all_figures:      List[str]
    all_output_files: List[str]
    response:         str
    exec_dir:         str = ""

    @property
    def failed_steps(self) -> List[StepResult]:
        return [s for s in self.step_results if not s.success and not s.skipped]

    @property
    def skipped_steps(self) -> List[StepResult]:
        return [s for s in self.step_results if s.skipped]


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def _call_llm(messages: List[Dict], llm_config: Dict, max_tokens: int = 4096) -> str:
    """Call the configured LLM (Ollama or OpenAI-compatible) and return reply text."""
    provider    = llm_config.get("provider", "ollama")
    model       = llm_config.get("model", "qwen2.5:7b")
    api_base    = llm_config.get("api_base", "http://localhost:11434")
    api_key     = llm_config.get("api_key", "")
    temperature = llm_config.get("temperature", 0.2)

    if provider == "ollama":
        url     = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
    else:
        url     = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"LLM connection failed ({url}): {e}")

    if provider == "ollama":
        return body.get("message", {}).get("content", "")
    return body.get("choices", [{}])[0].get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# Code / response parsers
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> str:
    """
    Extract Python or bash source from LLM response.
    Preference: ```python > ```bash/sh > bare ``` > raw text.
    """
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return "# lang:python\n" + m.group(1).strip()
    m = re.search(r"```(?:bash|sh)\s*(.*?)```", text, re.DOTALL)
    if m:
        return "# lang:bash\n" + m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _is_bash_code(code: str) -> bool:
    """Return True when code should be executed as a bash script."""
    stripped = code.strip()
    if stripped.startswith("# lang:bash"):
        return True
    first = stripped.splitlines()[0] if stripped else ""
    return (
        first.startswith("#!/bin/bash")
        or first.startswith("#!/usr/bin/env bash")
        or first.startswith("#!/bin/sh")
        or bool(re.match(r"^gmt\s+begin\b", stripped, re.I))
    )


def _extract_diagnosis(text: str) -> str:
    """Extract the [DIAGNOSIS] line from a debugger response."""
    m = re.search(r"\[DIAGNOSIS\]\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("```"):
            return line[:200]
    return "Unknown error"


def _extract_plan(text: str) -> List[str]:
    """Extract numbered steps from a [PLAN] block."""
    m = re.search(r"\[PLAN\](.*?)(?:\Z|\[)", text, re.DOTALL)
    block = m.group(1).strip() if m else text
    steps = []
    for line in block.splitlines():
        mm = re.match(r"^\d+[\.\)]\s+(.+)", line.strip())
        if mm:
            steps.append(mm.group(1).strip())
    return steps


# ---------------------------------------------------------------------------
# File path detector & data profiler
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(
    r'(?:^|[\s"\'])(/(?:[^\s"\']+/)*)([^\s"\']+\.'
    r'(csv|tsv|txt|json|xlsx|xls|npy|npz|h5|hdf5|sac|mseed|seed))',
    re.IGNORECASE | re.MULTILINE,
)


def _find_file_paths(text: str) -> List[str]:
    """Return absolute file paths mentioned in text that actually exist on disk."""
    found = []
    for m in _FILE_PATH_RE.finditer(text):
        p = (m.group(1) + m.group(2)).strip()
        if Path(p).is_file() and p not in found:
            found.append(p)
    return found


_PROFILE_SCRIPT = """
import sys, os, json, traceback
path = {path!r}
result = {{"path": path, "exists": os.path.isfile(path)}}
if not result["exists"]:
    print(json.dumps(result)); sys.exit(0)

ext = os.path.splitext(path)[1].lower()
result["size_mb"] = round(os.path.getsize(path) / 1e6, 2)
try:
    import pandas as pd
    if ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(path, sep="\\t" if ext == ".tsv" else ",", nrows=5000)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=5000)
    else:
        df = None
    if df is not None:
        result["type"] = "tabular"
        result["shape"]   = list(df.shape)
        result["columns"] = list(df.columns)
        result["dtypes"]  = {{c: str(t) for c, t in df.dtypes.items()}}
        result["sample"]  = df.head(3).to_dict(orient="records")
        stats = {{}}
        for col in df.select_dtypes("number").columns:
            s = df[col]
            stats[col] = {{"min": round(float(s.min()),4), "max": round(float(s.max()),4),
                           "mean": round(float(s.mean()),4), "nunique": int(s.nunique())}}
        result["stats"] = stats
except Exception:
    result["profile_error"] = traceback.format_exc(limit=2)
print(json.dumps(result, default=str))
"""


def _profile_file(path: str, project_root: str,
                  python_executable: Optional[str] = None) -> dict:
    """Run a quick pandas profile of path in the sandbox."""
    from .safe_executor import execute_code  # local import avoids circular deps
    script = _PROFILE_SCRIPT.format(path=path)
    res = execute_code(script, project_root=project_root, timeout=30, keep_dir=False,
                       python_executable=python_executable)
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"path": path, "profile_error": res.stderr or "no JSON output"}


def _format_file_context(profile: dict) -> str:
    """Turn a profile dict into a [FILE CONTEXT] text block for the LLM."""
    if not profile.get("exists"):
        return f"[FILE NOT FOUND] {profile.get('path','?')}"

    lines = [f"[FILE CONTEXT: {profile['path']}]",
             f"  Size: {profile.get('size_mb','?')} MB"]

    if profile.get("type") == "tabular":
        shape = profile.get("shape", [])
        cols  = profile.get("columns", [])
        lines += [f"  Shape: {shape[0]} rows × {shape[1]} columns",
                  f"  Columns: {', '.join(cols)}"]
        stats = profile.get("stats", {})
        if stats:
            lines.append("  Numeric column ranges:")
            for col, s in list(stats.items())[:12]:
                lines.append(f"    {col}: min={s['min']}, max={s['max']}, "
                              f"mean={s['mean']}, nunique={s['nunique']}")
        sample = profile.get("sample", [])
        if sample:
            lines.append(f"  Sample row: {json.dumps(sample[0], default=str)}")

        if cols:
            def _find_col(keywords, excl=None):
                excl = excl or []
                cands = []
                for kw in keywords:
                    for c in cols:
                        cl = c.lower()
                        if any(e in cl for e in excl):
                            continue
                        if cl == kw:
                            return c
                        if kw in cl:
                            cands.append(c)
                if not cands:
                    return None
                return next((c for c in cands if c.endswith("1")), cands[0])

            has_lat  = any("lat" in c.lower() for c in cols)
            lat_kw   = ["lat", "latitude"] + ([] if has_lat else ["y"])
            excl_id  = ["id", "index", "ray", "no", "num"]
            lon_col  = _find_col(["lon", "longitude", "long", "lng", "x"], excl_id)
            lat_col  = _find_col(lat_kw, excl_id)
            dep_col  = _find_col(["dep", "depth", "z"], excl_id)
            mag_col  = _find_col(["mag", "magnitude", "ml", "mw", "ms"], excl_id)

            if lon_col or lat_col:
                lines.append("  ⚠ USE EXACTLY these column names:")
                if lon_col:
                    lines.append(f"    longitude → df['{lon_col}']")
                if lat_col:
                    lines.append(f"    latitude  → df['{lat_col}']")
                if dep_col:
                    lines.append(f"    depth     → df['{dep_col}']")
                if mag_col:
                    lines.append(f"    magnitude → df['{mag_col}']")
                lines.append(f"  df = pd.read_csv(r'{profile['path']}')")
                if lon_col:
                    lines.append(f"  lon = df['{lon_col}'].values")
                if lat_col:
                    lines.append(f"  lat = df['{lat_col}'].values")

    if profile.get("profile_error"):
        lines.append(f"  [profile error: {profile['profile_error'][:200]}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Code pre-sanitiser
# ---------------------------------------------------------------------------

def _pre_sanitize(code: str) -> str:
    """Fix obvious LLM mistakes before execution. Bash scripts pass through unchanged."""
    if _is_bash_code(code):
        return code

    code = re.sub(r"\bplt\.show\(\s*\)", "pass  # display suppressed", code)
    code = re.sub(r"\bfig\.show\(\s*\)", "pass  # display suppressed", code)

    if "cartopy" in code and "matplotlib.use" not in code:
        code = "import matplotlib; matplotlib.use('Agg')\n" + code

    if "pd." in code and "import pandas as pd" not in code:
        code = "import pandas as pd\n" + code

    code = re.sub(r"^\s*from\s+seismo_code\.toolkit\s+import\s+.*$",
                  "pass  # toolkit pre-injected", code, flags=re.MULTILINE)

    if re.search(r"subprocess\.\w+\(\s*\[.{0,20}['\"]gmt['\"]", code):
        code = re.sub(
            r"subprocess\.\w+\(\s*\[.{0,20}['\"]gmt['\"][^\n]*",
            "raise RuntimeError('SAGE_HINT: use run_gmt(script) not subprocess for GMT')",
            code,
        )

    if re.search(r"gmt\s+grdimage", code) and re.search(r"gmt\s+coast.*-G\w", code):
        code = re.sub(r"(gmt\s+coast\b[^\n]*?)\s+-G[a-zA-Z/0-9@]+", r"\1", code)

    if re.search(r"\$\{[A-Za-z_]+\[", code) or re.search(r"<<\s*EOF", code):
        hint = ("raise RuntimeError('SAGE_HINT: bash array/EOF syntax inside Python "
                "string — write data with np.savetxt() then use ${{bash_var}} in f-string')")
        code = hint + "\n" + code

    if "gmt" in code and re.search(r"f(['\"]{{3}})", code):
        code = re.sub(r"awk\s+(['\"])\{print\s+([^}]+)\}(['\"])",
                      r"awk \1{{print \2}}\3", code)
        code = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", r"${{\1}}", code)

    if re.search(r"from\s+sage\s+import|import\s+sage\b", code):
        code = re.sub(r"^\s*(from\s+sage\s+import.*|import\s+sage\b.*)$",
                      "pass  # SAGE_HINT: toolkit pre-injected",
                      code, flags=re.MULTILINE)
    return code
