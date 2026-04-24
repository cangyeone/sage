"""
code_engine.py — Full-Cycle Python Programming Agent

Loop
----
  Plan → Code → Run → [Debug × N] → Verify → Return

Each phase:
  1. Plan      (optional): For complex requests, decompose into subtasks.
  2. Code      : LLM generates a self-contained Python script.
  3. Run       : Execute in an isolated subprocess (safe_executor).
  4. Debug     : If execution failed, an LLM debugger analyzes the error,
                 identifies the root cause, and emits a corrected script.
                 Repeats up to `max_debug_rounds` (default 4).
  5. Verify    : After success, the LLM critic checks whether the output
                 actually answers the user's request (optional, fast).

Progress callbacks
------------------
Pass `on_progress=callback` to `engine.run()`.
The callback receives a dict:
  { "phase": "generating"|"executing"|"debugging"|"verifying"|"done",
    "attempt": int, "message": str }
"""

from __future__ import annotations

import json
import re
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .safe_executor import ExecutionResult, execute_code

# seismo_skill context (optional)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from seismo_skill import build_skill_context as _build_skill_context
except Exception:
    def _build_skill_context(query: str, **_kw) -> str:  # type: ignore
        return ""


# ---------------------------------------------------------------------------
# ── Seismology toolkit context (injected into system prompts) ───────────────
# ---------------------------------------------------------------------------

_TOOLKIT_SUMMARY = """
## Built-in Seismology Toolkit (call directly — no import needed)

> These functions are pre-injected via `from seismo_code.toolkit import *`.
> ❌ Wrong: `from obspy import read_stream_from_dir`  (NOT an obspy function!)
> ✅ Right:  `st = read_stream_from_dir("/path/")`     (call directly)
> ✅ Native obspy: `from obspy import read; st = read("file.sac")`

### Data I/O
- `read_stream(path)` → obspy.Stream
- `read_stream_from_dir(directory)` → Stream

### Waveform Processing
- `detrend_stream(st, type='demean')` → Stream
- `taper_stream(st, max_percentage=0.05)` → Stream
- `filter_stream(st, filter_type, freqmin, freqmax, corners=4, zerophase=True)` → Stream
- `resample_stream(st, sampling_rate)` → Stream
- `trim_stream(st, starttime, endtime)` → Stream
- `merge_stream(st)` → Stream
- `remove_response(st, inventory_or_paz, output='VEL')` → Stream

### Visualization
- `plot_stream(st, title, outfile, picks, normalize=True)` → str (image path)
- `plot_spectrogram(tr, title, outfile, wlen=1.0)` → str
- `plot_psd(tr, title, outfile)` → (freqs, psd, str)
- `plot_particle_motion(st, outfile)` → str
- `plot_travel_time_curve(dist_range, depth_km, model, phases)` → str

### Travel Time
- `taup_arrivals(dist_deg, depth_km, model='iasp91', phases)` → list of dict
- `p_travel_time(dist_km, depth_km, model)` → float
- `s_travel_time(dist_km, depth_km, model)` → float

### Spectral Analysis
- `compute_spectrum(tr, method='fft')` → (freqs, amplitudes)
- `compute_hvsr(st, f_min, f_max, ...)` → (freqs, hvsr_mean, hvsr_std)

### Source Parameters
- `estimate_magnitude_ml(tr, dist_km)` → float (ML)
- `estimate_corner_freq(tr, dist_km, ...)` → (fc Hz, omega0)
- `estimate_seismic_moment(tr, dist_km)` → float (M₀)
- `moment_to_mw(M0)` → float (Mw)
- `estimate_stress_drop(M0, fc, vs=3500)` → float (MPa)

### Utilities
- `stream_info(st)` → str
- `picks_to_dict(picks_file)` → list of dict

### GMT Mapping
- `run_gmt(script, outname='gmt_map', title='GMT Map')` → str (PNG path)

### Image Saving
- All `plot_*` functions auto-save; manual: `savefig('filename.png')`
"""

# ── Code Generation System Prompt ──────────────────────────────────────────
_CODEGEN_SYSTEM = """You are an expert seismologist and Python programmer.
Users describe seismological data processing, analysis, and visualization tasks.
Generate directly executable Python code.

## ⚠️ CRITICAL: Toolkit usage
The execution environment pre-injects these functions — call directly, do NOT import:
  read_stream, read_stream_from_dir, detrend_stream, taper_stream, filter_stream,
  plot_stream, plot_spectrogram, plot_psd, plot_particle_motion, stream_info, picks_to_dict,
  taup_arrivals, p_travel_time, s_travel_time, compute_spectrum, compute_hvsr,
  estimate_magnitude_ml, estimate_corner_freq, estimate_seismic_moment, savefig, run_gmt

❌ Forbidden:  from seismo_code.toolkit import ...   # already injected
❌ Forbidden:  from obspy import read_stream_from_dir  # not an obspy function
✅ Correct:    st = read_stream_from_dir("/path/")
✅ Obspy OK:   from obspy import read; st = read("file.sac")

## Rules
1. Output ONLY a ```python ... ``` code block. No explanations.
2. Code must be self-contained. Reuse paths/variables from conversation history.
3. NEVER call plt.show() — server has no display. Use savefig() or plot_*() instead.
4. Use try/except for file I/O and network calls; print clear error messages.
5. Print all numerical results with print().
6. For directory listings: print full path of each file with os.path.join().
7. For plot requests: read data → process → call plot_stream() / savefig().
8. Combine related steps in ONE code block (read + filter + plot + stats).

## Available libraries
obspy, numpy, scipy, matplotlib (Agg backend), pandas, sklearn (if installed)

{toolkit}
""".format(toolkit=_TOOLKIT_SUMMARY)

# ── Debugger System Prompt ──────────────────────────────────────────────────
_DEBUG_SYSTEM = """You are an expert Python debugger specializing in scientific computing.

You will receive:
- A failing Python script
- The full traceback / error message
- Any partial stdout before the crash

Your job:
1. Identify the root cause in ONE sentence.
2. Output the COMPLETE corrected Python script (not a patch — the full file).

Response format (strict):
[DIAGNOSIS]
<one-sentence root cause>

```python
<complete corrected code>
```

Rules:
- Fix ONLY what is broken; preserve the user's intent.
- If the error is a missing library, add a try/except fallback or use an alternative.
- If a file path is wrong, add code to search for the correct path.
- If the error is a logic bug, fix the logic.
- NEVER use plt.show(). NEVER re-import toolkit functions.
- The output code block must be complete and self-contained.
"""

# ── Verifier System Prompt ──────────────────────────────────────────────────
_VERIFY_SYSTEM = """You are a code output verifier for seismological Python scripts.

Given the user's original request and the program's stdout + list of generated files,
decide whether the output actually fulfills the request.

Respond with ONE of:
  PASS
  FAIL: <brief reason (≤ 20 words)>

Be lenient — if the key result was produced (figure, numerical answer, file),
output PASS even if minor details differ.
"""

# ── Planner System Prompt ───────────────────────────────────────────────────
_PLAN_SYSTEM = """You are a scientific Python programming assistant.

Given a user's data analysis request and (optionally) a summary of the data file,
produce a concise execution plan — what the code will do step by step.

Output format (strict):
[PLAN]
1. <step>
2. <step>
...

Rules:
- 3–7 steps maximum.
- Each step ≤ 12 words.
- Cover: data loading, structure inspection, computation, visualization.
- Do NOT output any code.
"""


# ---------------------------------------------------------------------------
# ── LLM client ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _call_llm(messages: List[Dict], llm_config: Dict, max_tokens: int = 4096) -> str:
    provider = llm_config.get("provider", "ollama")
    model    = llm_config.get("model", "qwen2.5:7b")
    api_base = llm_config.get("api_base", "http://localhost:11434")
    api_key  = llm_config.get("api_key", "")
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


def _extract_code(text: str) -> str:
    """Extract Python source from LLM response."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m: return m.group(1).strip()
    return text.strip()


def _extract_diagnosis(text: str) -> str:
    """Extract [DIAGNOSIS] line from debugger response."""
    m = re.search(r"\[DIAGNOSIS\]\s*(.+?)(?:\n|$)", text)
    if m: return m.group(1).strip()
    # fallback: first non-empty line before the code block
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("```"):
            return line[:200]
    return "Unknown error"


# ---------------------------------------------------------------------------
# ── File-path detector & data profiler ──────────────────────────────────────
# ---------------------------------------------------------------------------

_FILE_PATH_RE = re.compile(
    r'(?:^|[\s"\'])(/(?:[^\s"\']+/)*)([^\s"\']+\.(csv|tsv|txt|json|xlsx|xls|npy|npz|h5|hdf5|sac|mseed|seed))',
    re.IGNORECASE | re.MULTILINE,
)


def _find_file_paths(text: str) -> List[str]:
    """Return absolute file paths mentioned in *text* that actually exist."""
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
    print(json.dumps(result))
    sys.exit(0)

ext = os.path.splitext(path)[1].lower()
result["size_mb"] = round(os.path.getsize(path) / 1e6, 2)

try:
    import pandas as pd
    if ext in (".csv", ".tsv", ".txt"):
        sep = "\\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=5000)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, nrows=5000)
    else:
        df = None

    if df is not None:
        result["type"] = "tabular"
        result["shape"] = list(df.shape)
        result["columns"] = list(df.columns)
        result["dtypes"] = {{c: str(t) for c, t in df.dtypes.items()}}
        result["sample"] = df.head(3).to_dict(orient="records")
        stats = {{}}
        for col in df.select_dtypes("number").columns:
            s = df[col]
            stats[col] = {{"min": round(float(s.min()),4),
                           "max": round(float(s.max()),4),
                           "mean": round(float(s.mean()),4),
                           "nunique": int(s.nunique())}}
        result["stats"] = stats
except Exception:
    result["profile_error"] = traceback.format_exc(limit=2)

print(json.dumps(result, default=str))
"""


def _profile_file(path: str, project_root: str) -> dict:
    """
    Run a quick pandas profile of *path* in the sandbox.
    Returns a dict with shape, columns, stats, sample rows.
    """
    script = _PROFILE_SCRIPT.format(path=path)
    res = execute_code(script, project_root=project_root, timeout=30, keep_dir=False)
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"path": path, "profile_error": res.stderr or "no JSON output"}


def _format_file_context(profile: dict) -> str:
    """Turn a profile dict into a concise text block for the LLM prompt."""
    if not profile.get("exists"):
        return f"[FILE NOT FOUND] {profile.get('path','?')}"

    lines = [f"[FILE CONTEXT: {profile['path']}]"]
    lines.append(f"  Size: {profile.get('size_mb','?')} MB")

    if profile.get("type") == "tabular":
        shape = profile.get("shape", [])
        lines.append(f"  Shape: {shape[0]} rows × {shape[1]} columns")
        lines.append(f"  Columns: {', '.join(profile.get('columns', []))}")

        stats = profile.get("stats", {})
        if stats:
            lines.append("  Numeric column ranges:")
            for col, s in list(stats.items())[:12]:
                lines.append(
                    f"    {col}: min={s['min']}, max={s['max']}, "
                    f"mean={s['mean']}, nunique={s['nunique']}"
                )

        sample = profile.get("sample", [])
        if sample:
            lines.append(f"  Sample row: {json.dumps(sample[0], default=str)}")

    if profile.get("profile_error"):
        lines.append(f"  [profile error: {profile['profile_error'][:200]}]")

    return "\n".join(lines)


def _extract_plan(text: str) -> List[str]:
    """Extract numbered steps from a [PLAN] block."""
    m = re.search(r"\[PLAN\](.*?)(?:\Z|\[)", text, re.DOTALL)
    block = m.group(1).strip() if m else text
    steps = []
    for line in block.splitlines():
        line = line.strip()
        mm = re.match(r"^\d+[\.\)]\s+(.+)", line)
        if mm:
            steps.append(mm.group(1).strip())
    return steps


def _pre_sanitize(code: str) -> str:
    """Fix obvious LLM mistakes before execution (fast, no LLM call)."""
    # Neutralise plt.show() calls (server has no display)
    code = re.sub(r"\bplt\.show\(\s*\)", "pass  # display suppressed", code)
    # Remove re-imports of the pre-injected toolkit (causes ImportError)
    code = re.sub(
        r"^\s*from\s+seismo_code\.toolkit\s+import\s+.*$",
        "pass  # toolkit pre-injected",
        code, flags=re.MULTILINE,
    )
    return code


# ---------------------------------------------------------------------------
# ── Data classes ─────────────────────────────────────────────────────────────
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
    success:      bool
    response:     str            # Human-readable summary shown to user
    code:         str            # Final code (successful or last attempt)
    exec_result:  Optional[ExecutionResult]
    attempts:     int = 1        # Total execution attempts (1 = no retries needed)
    debug_trace:  List[DebugAttempt] = field(default_factory=list)
    verify_pass:  Optional[bool]  = None   # None = not run
    verify_note:  str = ""
    plan:         List[str] = field(default_factory=list)   # planned steps
    script_path:  str = ""        # path to saved .py script (for download)

    @property
    def figures(self) -> List[str]:
        return self.exec_result.figures if self.exec_result else []

    @property
    def output_files(self) -> List[str]:
        return self.exec_result.output_files if self.exec_result else []

    @property
    def stdout(self) -> str:
        return self.exec_result.stdout if self.exec_result else ""


# ---------------------------------------------------------------------------
# ── Code Engine ──────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class CodeEngine:
    """
    Full-cycle Python programming agent.

    Loop: Code → Run → [Debug × N] → Verify

    Usage
    -----
    engine = CodeEngine(llm_config)
    result = engine.run(
        "Filter /data/event/ waveforms 1-10 Hz and plot",
        max_debug_rounds=4,
        run_verify=True,
        on_progress=lambda p: print(p["message"]),
    )
    """

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        project_root: Optional[str] = None,
    ):
        if llm_config is None:
            llm_config = self._load_llm_config()
        self.llm_config   = llm_config
        self.project_root = project_root or str(Path(__file__).parent.parent)
        # Multi-turn conversation history for the code generator
        self._history: List[Dict] = [{"role": "system", "content": _CODEGEN_SYSTEM}]
        self._last_exec_dir: Optional[str] = None

    # ── Config helpers ────────────────────────────────────────────────────────
    @staticmethod
    def _load_llm_config() -> Dict:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config_manager import LLMConfigManager
            return LLMConfigManager().get_llm_config()
        except Exception:
            return {"provider": "ollama", "model": "",
                    "api_base": "http://localhost:11434"}

    def is_llm_available(self) -> bool:
        try:
            provider = self.llm_config.get("provider", "ollama")
            api_base = self.llm_config.get("api_base", "http://localhost:11434")
            url = api_base.rstrip("/") + ("/api/tags" if provider == "ollama" else "/models")
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _emit(
        self,
        on_progress: Optional[Callable],
        phase: str,
        attempt: int,
        message: str,
    ):
        if on_progress:
            try:
                on_progress({"phase": phase, "attempt": attempt, "message": message})
            except Exception:
                pass

    def _run_code(self, code: str, timeout: int) -> ExecutionResult:
        """Execute code and return result."""
        clean = _pre_sanitize(code)
        return execute_code(
            clean,
            project_root=self.project_root,
            timeout=timeout,
            keep_dir=True,
        )

    def _build_error_context(self, code: str, exec_res: ExecutionResult) -> str:
        """Build a compact error context string for the debugger."""
        parts = []
        if exec_res.stdout.strip():
            # Only last 1500 chars of stdout (partial output is useful)
            parts.append("=== Partial stdout (last 1500 chars) ===\n" +
                         exec_res.stdout.strip()[-1500:])
        stderr = (exec_res.stderr or "").strip()
        if stderr:
            parts.append("=== Traceback ===\n" + stderr[-3000:])
        if exec_res.error:
            parts.append("=== Error summary ===\n" + exec_res.error)
        return "\n\n".join(parts) if parts else "No error details captured."

    # ── Debug + Fix ───────────────────────────────────────────────────────────
    def _debug_and_fix(
        self,
        original_request: str,
        failed_code: str,
        exec_res: ExecutionResult,
        attempt: int,
        timeout: int,
        on_progress: Optional[Callable],
    ) -> tuple[str, ExecutionResult, str]:
        """
        Ask the LLM debugger to fix the failing code.

        Returns (fixed_code, new_exec_result, diagnosis)
        """
        error_ctx = self._build_error_context(failed_code, exec_res)

        debug_messages = [
            {"role": "system", "content": _DEBUG_SYSTEM},
            {"role": "user", "content": (
                f"## Original user request\n{original_request}\n\n"
                f"## Failing code\n```python\n{failed_code}\n```\n\n"
                f"## Error output\n{error_ctx}\n\n"
                "Fix the code. Output [DIAGNOSIS] then the corrected ```python``` block."
            )},
        ]

        self._emit(on_progress, "debugging", attempt,
                   f"Analyzing error (attempt {attempt})…")

        try:
            raw = _call_llm(debug_messages, self.llm_config, max_tokens=4096)
        except ConnectionError as e:
            return failed_code, exec_res, str(e)

        diagnosis  = _extract_diagnosis(raw)
        fixed_code = _extract_code(raw)

        self._emit(on_progress, "executing", attempt,
                   f"Running fixed code (attempt {attempt})…")
        new_exec = self._run_code(fixed_code, timeout)
        return fixed_code, new_exec, diagnosis

    # ── Verify output ─────────────────────────────────────────────────────────
    def _verify_output(
        self,
        original_request: str,
        exec_res: ExecutionResult,
    ) -> tuple[bool, str]:
        """
        Quick LLM sanity-check: did the output fulfil the request?
        Returns (passed, note).
        """
        files_list = "\n".join(
            [f"  [figure] {p}" for p in exec_res.figures] +
            [f"  [file]   {p}" for p in exec_res.output_files]
        ) or "  (none)"

        verify_messages = [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": (
                f"## User request\n{original_request}\n\n"
                f"## Stdout\n{exec_res.stdout.strip()[-2000:] or '(empty)'}\n\n"
                f"## Generated files\n{files_list}\n\n"
                "Does the output fulfil the request? Reply PASS or FAIL: <reason>."
            )},
        ]
        try:
            verdict = _call_llm(verify_messages, self.llm_config, max_tokens=80).strip()
        except Exception:
            return True, ""   # don't block on verify failure

        if verdict.upper().startswith("PASS"):
            return True, ""
        m = re.match(r"FAIL[:\s]+(.*)", verdict, re.IGNORECASE)
        note = m.group(1).strip() if m else verdict[:120]
        return False, note

    # ── Main entry point ──────────────────────────────────────────────────────
    def run(
        self,
        user_request: str,
        data_hint: Optional[str] = None,
        max_debug_rounds: int = 4,
        timeout: int = 120,
        run_verify: bool = False,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> CodeRunResult:
        """
        Generate, execute, debug and (optionally) verify Python code.

        Parameters
        ----------
        user_request : str
            Natural-language task description.
        data_hint : str, optional
            File or directory path to prepend to the prompt.
        max_debug_rounds : int
            Maximum automatic fix attempts after first failure (default 4).
        timeout : int
            Execution timeout per attempt in seconds (default 120).
        run_verify : bool
            Run LLM self-check after a successful execution (default False).
        on_progress : callable, optional
            Called with progress dicts during the run.

        Returns
        -------
        CodeRunResult
        """
        # ── 1. Detect & profile files mentioned in the request ───────────────
        file_contexts: List[str] = []
        all_text = user_request + (f"\n{data_hint}" if data_hint else "")
        found_paths = _find_file_paths(all_text)
        if found_paths:
            self._emit(on_progress, "analyzing", 0,
                       f"Analyzing {len(found_paths)} file(s)…")
            for fp in found_paths[:3]:   # profile up to 3 files
                profile = _profile_file(fp, self.project_root)
                file_contexts.append(_format_file_context(profile))

        # ── 2. Build user message ─────────────────────────────────────────────
        msg_content = user_request
        if data_hint:
            msg_content += f"\n\nData path: {data_hint}"
        if file_contexts:
            msg_content += "\n\n" + "\n\n".join(file_contexts)

        self._history.append({"role": "user", "content": msg_content})

        # Inject relevant skill docs into the system prompt for this turn
        skill_ctx = _build_skill_context(user_request, max_chars=5000, top_k=2)
        system_content = _CODEGEN_SYSTEM
        if skill_ctx:
            system_content += "\n\n## Relevant skill docs\n" + skill_ctx
        messages = [{"role": "system", "content": system_content}] + \
                   [m for m in self._history if m["role"] != "system"]

        # ── 3. Generate plan (fast, optional) ────────────────────────────────
        plan: List[str] = []
        self._emit(on_progress, "planning", 0, "Planning…")
        try:
            plan_msgs = [
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content":
                    f"Request: {user_request}\n\n" +
                    ("\n".join(file_contexts) if file_contexts else "") +
                    "\n\nList the execution steps."},
            ]
            raw_plan = _call_llm(plan_msgs, self.llm_config, max_tokens=400)
            plan = _extract_plan(raw_plan)
        except Exception:
            pass   # planning failure is non-fatal

        if plan:
            self._emit(on_progress, "planning", 0,
                       "Plan: " + " → ".join(plan))

        # ── 4. Generate initial code ──────────────────────────────────────────
        self._emit(on_progress, "generating", 0, "Generating code…")

        try:
            raw_response = _call_llm(messages, self.llm_config)
        except ConnectionError as e:
            return CodeRunResult(success=False, response=str(e),
                                 code="", exec_result=None)

        code = _extract_code(raw_response)

        # ── 3. First execution ────────────────────────────────────────────────
        self._emit(on_progress, "executing", 0, "Executing code…")
        exec_res = self._run_code(code, timeout)

        debug_trace: List[DebugAttempt] = []
        attempt = 0

        # ── 4. Debug loop ─────────────────────────────────────────────────────
        while not exec_res.success and attempt < max_debug_rounds:
            attempt += 1
            error_summary = f"{exec_res.stderr}\n{exec_res.error}".strip()

            debug_trace.append(DebugAttempt(
                attempt=attempt,
                diagnosis="",         # filled below
                code=code,
                error=error_summary,
                stdout=exec_res.stdout,
                success=False,
            ))

            fixed_code, new_exec, diagnosis = self._debug_and_fix(
                original_request=user_request,
                failed_code=code,
                exec_res=exec_res,
                attempt=attempt,
                timeout=timeout,
                on_progress=on_progress,
            )

            # Record diagnosis in the trace
            debug_trace[-1].diagnosis = diagnosis

            code     = fixed_code
            exec_res = new_exec

            if exec_res.success:
                debug_trace.append(DebugAttempt(
                    attempt=attempt,
                    diagnosis=f"Fixed: {diagnosis}",
                    code=code,
                    error="",
                    stdout=exec_res.stdout,
                    success=True,
                ))
                self._emit(on_progress, "executing", attempt,
                           f"✓ Fixed after {attempt} debug round(s)")
                break
            else:
                self._emit(on_progress, "debugging", attempt,
                           f"Attempt {attempt} still failing, retrying…")

        # ── 5. Update conversation history (add final code) ───────────────────
        self._history.append({
            "role": "assistant",
            "content": f"```python\n{code}\n```"
        })
        if exec_res:
            self._last_exec_dir = exec_res.exec_dir

        # ── 6. Verify (optional) ──────────────────────────────────────────────
        verify_pass, verify_note = None, ""
        if run_verify and exec_res and exec_res.success:
            self._emit(on_progress, "verifying", attempt, "Verifying output…")
            verify_pass, verify_note = self._verify_output(user_request, exec_res)

        # ── 7. Save final script to a .py file (always — for download) ───────
        script_path = ""
        if code:
            try:
                import tempfile, os as _os
                script_dir = exec_res.exec_dir if (exec_res and exec_res.exec_dir) \
                             else tempfile.mkdtemp(prefix="sage_script_")
                script_path = _os.path.join(script_dir, "analysis.py")
                header = (
                    f"# Generated by SeismicX — {user_request[:80]}\n"
                    f"# Attempts: {attempt + 1}\n\n"
                )
                with open(script_path, "w", encoding="utf-8") as _f:
                    _f.write(header + code)
            except Exception:
                pass

        # ── 8. Build human-readable response ─────────────────────────────────
        total_attempts = attempt + 1
        response = self._build_response(exec_res, total_attempts, verify_pass, verify_note)

        self._emit(on_progress, "done", attempt, response)

        return CodeRunResult(
            success=exec_res.success if exec_res else False,
            response=response,
            code=code,
            exec_result=exec_res,
            attempts=total_attempts,
            debug_trace=debug_trace,
            verify_pass=verify_pass,
            verify_note=verify_note,
            plan=plan,
            script_path=script_path,
        )

    def _build_response(
        self,
        exec_res: Optional[ExecutionResult],
        attempts: int,
        verify_pass: Optional[bool],
        verify_note: str,
    ) -> str:
        if not exec_res:
            return "Execution failed — no result."

        lines = []
        if exec_res.success:
            if attempts == 1:
                lines.append("✓ Code ran successfully")
            else:
                lines.append(f"✓ Code succeeded after {attempts} attempts (auto-debugged)")
        else:
            lines.append(f"✗ Execution failed after {attempts} attempt(s)")

        if exec_res.stdout.strip():
            lines.append("Output:\n" + textwrap.indent(exec_res.stdout.strip(), "  "))

        if exec_res.figures:
            lines.append(f"Generated {len(exec_res.figures)} figure(s)")

        if exec_res.output_files:
            lines.append(f"Generated {len(exec_res.output_files)} file(s)")

        if not exec_res.success:
            err = (exec_res.stderr or exec_res.error or "").strip()
            if err:
                lines.append("Last error:\n" + textwrap.indent(err[-800:], "  "))

        if verify_pass is False:
            lines.append(f"⚠ Output check: {verify_note}")

        return "\n".join(lines)

    # ── Session management ────────────────────────────────────────────────────
    def reset(self):
        """Reset conversation history."""
        self._history     = [{"role": "system", "content": _CODEGEN_SYSTEM}]
        self._last_exec_dir = None


# ---------------------------------------------------------------------------
# ── Singleton / factory ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_engine_instance: Optional[CodeEngine] = None


def get_code_engine(llm_config: Optional[Dict] = None) -> CodeEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CodeEngine(llm_config)
    return _engine_instance


def reset_code_engine():
    global _engine_instance
    if _engine_instance:
        _engine_instance.reset()
