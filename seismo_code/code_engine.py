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

from .safe_executor import ExecutionResult, execute_code, execute_bash

# seismo_skill context (optional)
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent))
    from seismo_skill import build_skill_context as _build_skill_context
    from seismo_skill import build_skill_context_with_rag as _build_skill_context_with_rag
except Exception:
    def _build_skill_context(query: str, **_kw) -> str:  # type: ignore
        return ""
    def _build_skill_context_with_rag(query: str, **_kw):  # type: ignore
        return "", ""


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
- For **pure GMT tasks** → output a ```bash script (see GMT section below).
- For **mixed Python+GMT** (data prep in Python then GMT map) → call
  `run_gmt(bash_script_string, outname='map')` as a last resort only.

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

## CSV/TXT data files
- Use `pandas.read_csv(path, sep=None, engine='python')` for unknown delimiters.
- For tab-delimited `.txt`, use `pd.read_table(path)` or `pd.read_csv(path, sep='\t')`.
- If the file has no header, use `header=None` and assign `names=[...]`.
- If parsing fails, inspect the first few lines with `open(path).read().splitlines()[:20]`.
- Always print `df.columns.tolist()` and `df.head(3)` when you first read a table.
- When using pandas, always include `import pandas as pd` at the top of the script.
- `read_stream_from_dir(path)` is only for waveform directories or collections, not a `.csv` tabular file.
- If the requested path ends with `.csv`, prefer pandas and do NOT pass the file directly into `read_stream_from_dir`.

## ⚠️ CRITICAL — CSV column names
When a [FILE CONTEXT] block is provided, it shows the EXACT column names.
USE those exact names — do NOT write detection loops that scan columns.

REQUIRED pattern (copy exactly, substitute real column names from FILE CONTEXT):
```python
df = pd.read_csv(path)
print("DataFrame columns:", df.columns.tolist())
print("First 3 rows:\n", df.head(3))

# Use EXACT names from FILE CONTEXT — do NOT auto-detect
lon_col = 'lon1'   # ← set to exact name from FILE CONTEXT
lat_col = 'lat1'   # ← set to exact name from FILE CONTEXT
lon = df[lon_col].values
lat = df[lat_col].values

# ⚠️ MANDATORY validation — catches wrong column selection immediately
assert lon.min() >= -180 and lon.max() <= 180, \
    f"Longitude out of range [{lon.min():.2f}, {lon.max():.2f}] — wrong column '{lon_col}'?"
assert lat.min() >= -90  and lat.max() <= 90, \
    f"Latitude out of range [{lat.min():.2f}, {lat.max():.2f}] — wrong column '{lat_col}'? Try 'lat1' or 'lat2'."
print(f"Using lon={lon_col} ({lon.min():.4f}~{lon.max():.4f}), lat={lat_col} ({lat.min():.4f}~{lat.max():.4f})")
```

Column selection rules:
- If FILE CONTEXT says "longitude → df['lon1']" → use lon_col = 'lon1'
- If there are lon1/lon2 and lat1/lat2 pairs, prefer lon1/lat1 (source station)
- NEVER use id columns (ray_id, id, index) as coordinates — they are not geographic
- 'y' alone is NOT a reliable latitude indicator when 'lat1'/'lat2' exist

## ⚠️ Map / Geographic Plotting

### DEFAULT: matplotlib + cartopy  (always use this unless user says "GMT")

```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# lon, lat, depth — numpy arrays already available
pad  = max((lon.max()-lon.min())*0.12, (lat.max()-lat.min())*0.12, 0.5)
extent = [lon.min()-pad, lon.max()+pad, lat.min()-pad, lat.max()+pad]

fig, ax = plt.subplots(figsize=(10, 8),
                        subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent(extent, crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,      facecolor='#f0ede5', zorder=0)
ax.add_feature(cfeature.OCEAN,     facecolor='#d6eaf8', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='#444', zorder=2)
ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle=':', color='#777', zorder=2)

gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray',
                  alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

sc = ax.scatter(lon, lat, c=depth, cmap='plasma_r', s=20,
                transform=ccrs.PlateCarree(), zorder=5,
                alpha=0.85, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Depth (km)', shrink=0.65, pad=0.02)
ax.set_title('Seismicity Map', fontsize=13, pad=10)

savefig('seismicity_map.png')   # pre-injected: saves PNG and registers [FIGURE]
plt.close()
```

cartopy rules:
- ✅ Always pass `transform=ccrs.PlateCarree()` to scatter/plot on a GeoAxes
- ✅ `ax.set_extent([w,e,s,n])` — NOT ax.set_xlim/set_ylim
- ✅ `savefig('name.png')` — pre-injected, auto-registers figure
- ❌ NEVER call `plt.show()` — server has no display

Magnitude-scaled markers: `s = (2**mag) * 3` (exponential)
Multi-panel map+cross-section: use `subplot_kw` only on the map axes

---

### GMT: ONLY when user explicitly says "GMT" or "gmt绘图"

| Situation | Output format |
|-----------|--------------|
| Pure GMT mapping (no user data) | **```bash** script — always preferred |
| Needs Python data prep first (load CSV / compute arrays) | **```python** calling `run_gmt(script_str, outname)` |

- In bash: normal `${VAR}`, `$(cmd)`, `awk '{print $6}'` — no escaping needed
- In Python f-strings: bash vars → `${{Z_MIN}}`, awk → `{{print $6}}`, Python vars → `{var}`
- Script must `cd "${SAGE_OUTDIR}"` at the top; use `gmt begin <name> PNG` ... `gmt end`
- Always include terrain: `gmt grdimage` → `gmt coast` (no `-G` fill) → data → `gmt colorbar`
- ❌ `@earth_relief_01m` — timeout; chain 02m→05m fallback instead
- See the **`gmt_plotting` skill** for complete templates, topography snippet, and all GMT rules

## Available libraries
obspy, numpy, scipy, matplotlib (Agg backend), cartopy, pandas, sklearn (if installed)

""" + _TOOLKIT_SUMMARY

# ── Bash / GMT error pattern helpers ─────────────────────────────────────────
_BASH_ERROR_HINTS = """
## Bash / Shell Script Debugging Rules

### Exit-code errors (non-zero return from subprocess / run_gmt)
- Check the last line of stderr for the actual error message.
- `exit 1` usually means a preceding command failed — trace up to find it.
- `Permission denied` → file/dir permissions; add `chmod +x` or change path.
- `command not found` → package not installed or PATH issue; check with `which <cmd>`.

### GMT-specific errors
- `Option -B: Unrecognized modifier` → wrong annotation syntax; check GMT 6 docs.
- `grdimage: Cannot find file` → DEM path wrong or download failed; check grid exists.
- `makecpt: No color table` → wrong CPT name; use `geo`, `topo`, `hot`, `jet`, etc.
- `psxy: Ambiguous option` → mixed classic/modern flags; stay in modern mode.
- Silent blank output → wrong layer order; see `gmt_plotting` skill for correct sequence.
- For mixed Python+GMT: f-string bash vars → `${{VAR}}`, awk → `{{print $1}}`, Python vars → `{var}`
- See **`gmt_plotting` skill** for complete GMT scripting guidance and templates.

### Python + Bash mixed debugging
- If a subprocess call exits with code 1 and stderr is empty, add `check=False` and
  print stderr to diagnose.
- For `CalledProcessError`: capture output with `capture_output=True, text=True` and
  print `result.stderr` for details.
- For timeout errors: increase timeout or split into smaller sub-calls.
"""

# ── Debugger System Prompt ──────────────────────────────────────────────────
_DEBUG_SYSTEM = """You are an expert Python and Bash debugger specializing in scientific computing.

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
- If the failure is due to CSV/TXT parsing, inspect the file header and delimiter and use pandas with fallback parsing.
- If the failure is due to unknown data structure, print the first rows and inferred column mapping, then retry with guessed columns.
- If the error is `NameError: name 'lon' is not defined` (or 'lat', 'depth', etc.),
  check the [Data file context] block for EXACT column names and use df['col_name'].values.
  NEVER reference bare variable names like `lon`, `lat` unless they were explicitly assigned.
- If the script uses `subprocess.run(['gmt', ...])` or has a GMT SyntaxError/IndentationError:
  For pure GMT tasks, rewrite as a **```bash** code block (the engine runs it directly).
  For mixed Python+GMT, call `run_gmt(script_str, outname)` with an f-string script where
  bash vars use `${{Z_MIN}}`, awk uses `{{print $6}}`, and Python vars use `{var}`.
  Use @earth_relief_02m → @earth_relief_05m chain (never 01m); check `[ -f topo.grd ]` exists.
- If the error is `ModuleNotFoundError: No module named 'sage'` or similar:
  The toolkit functions (run_gmt, savefig, read_stream_from_dir, etc.) are PRE-INJECTED.
  NEVER write `from sage import ...` or `import sage`. Call them directly: `run_gmt(script)`.
- If the error is `ModuleNotFoundError: No module named 'cartopy'`:
  Fall back to a plain matplotlib scatter without a geo projection:
  ```python
  fig, ax = plt.subplots(figsize=(10, 8))
  sc = ax.scatter(lon, lat, c=depth, cmap='plasma_r', s=20, alpha=0.8)
  plt.colorbar(sc, ax=ax, label='Depth (km)')
  ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
  ax.set_title('Seismicity Map'); ax.grid(True, alpha=0.3)
  savefig('seismicity_map.png'); plt.close()
  ```
- If the error is `AttributeError: 'GeoAxes' object has no attribute 'set_xlim'`:
  Replace `ax.set_xlim/set_ylim` with `ax.set_extent([west, east, south, north])`.
- If the error is `ValueError: ... transform ... PlateCarree`:
  Add `transform=ccrs.PlateCarree()` to every scatter/plot call on a GeoAxes.
- NEVER use plt.show(). NEVER re-import toolkit functions. NEVER import from sage.
- The output code block must be complete and self-contained.
""" + _BASH_ERROR_HINTS

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
    """Extract Python or bash source from LLM response.

    Preference order: ```python  >  ```bash/sh  >  bare ``` block  >  raw text.
    The language tag is preserved as the first line (``# lang:python`` /
    ``# lang:bash``) so the engine can detect which executor to use.
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
    """Return True when *code* should be executed as a bash script."""
    stripped = code.strip()
    if stripped.startswith("# lang:bash"):
        return True
    # Heuristic fallback: shebang or bare `gmt begin` at the top
    first_line = stripped.splitlines()[0] if stripped else ""
    return (
        first_line.startswith("#!/bin/bash")
        or first_line.startswith("#!/usr/bin/env bash")
        or first_line.startswith("#!/bin/sh")
        or bool(re.match(r"^gmt\s+begin\b", stripped, re.I))
    )


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


def _profile_file(path: str, project_root: str, python_executable: Optional[str] = None) -> dict:
    """
    Run a quick pandas profile of *path* in the sandbox.
    Returns a dict with shape, columns, stats, sample rows.
    """
    script = _PROFILE_SCRIPT.format(path=path)
    res = execute_code(script, project_root=project_root, timeout=30, keep_dir=False, python_executable=python_executable)
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
        cols  = profile.get("columns", [])
        lines.append(f"  Shape: {shape[0]} rows × {shape[1]} columns")
        lines.append(f"  Columns: {', '.join(cols)}")

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

        # ── Emit explicit coordinate column hints ────────────────────────────
        if cols:
            def _find_col(keywords, exclude_patterns=None):
                """Find best matching column. Prefer exact match, then *1 suffix, then first substring."""
                exclude_patterns = exclude_patterns or []
                candidates = []
                for kw in keywords:
                    for c in cols:
                        cl = c.lower()
                        # skip if matches an exclusion pattern
                        if any(ep in cl for ep in exclude_patterns):
                            continue
                        if cl == kw:
                            return c   # exact match wins immediately
                        if kw in cl:
                            candidates.append(c)
                if not candidates:
                    return None
                # prefer columns ending in '1' (first of a pair like lon1/lon2)
                for c in candidates:
                    if c.endswith('1'):
                        return c
                return candidates[0]

            # 'y' only used for latitude if no 'lat*' column exists at all
            has_lat_col = any('lat' in c.lower() for c in cols)
            lat_keywords = ['lat', 'latitude'] + ([] if has_lat_col else ['y'])

            # exclude id/index columns from coordinate detection
            exclude_id = ['id', 'index', 'ray', 'no', 'num']

            lon_col = _find_col(['lon', 'longitude', 'long', 'lng', 'x'], exclude_id)
            lat_col = _find_col(lat_keywords, exclude_id)
            dep_col = _find_col(['dep', 'depth', 'z'], exclude_id)
            mag_col = _find_col(['mag', 'magnitude', 'ml', 'mw', 'ms'], exclude_id)

            if lon_col or lat_col:
                lines.append("  ⚠ USE EXACTLY these column names (do NOT use 'lon'/'lat'):")
                if lon_col:
                    lines.append(f"    longitude → df['{lon_col}']   # lon = df['{lon_col}'].values")
                if lat_col:
                    lines.append(f"    latitude  → df['{lat_col}']   # lat = df['{lat_col}'].values")
                if dep_col:
                    lines.append(f"    depth     → df['{dep_col}']")
                if mag_col:
                    lines.append(f"    magnitude → df['{mag_col}']")
                lines.append(f"  Minimal loading code:")
                lines.append(f"    df = pd.read_csv(r'{profile['path']}')")
                if lon_col:
                    lines.append(f"    lon = df['{lon_col}'].values")
                if lat_col:
                    lines.append(f"    lat = df['{lat_col}'].values")

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
    """Fix obvious LLM mistakes before execution (fast, no LLM call).

    Only applies to Python code — bash scripts are passed through unchanged.
    """
    if _is_bash_code(code):
        return code   # bash: nothing to sanitize here

    # Neutralise plt.show() / fig.show() calls (server has no display)
    code = re.sub(r"\bplt\.show\(\s*\)", "pass  # display suppressed", code)
    code = re.sub(r"\bfig\.show\(\s*\)", "pass  # display suppressed", code)

    # Ensure matplotlib uses Agg backend when cartopy is used
    # (MPLBACKEND env is already set, but belt-and-braces for any import order issue)
    if 'cartopy' in code and 'matplotlib.use' not in code:
        code = "import matplotlib; matplotlib.use('Agg')\n" + code

    # Auto-add pandas import if generated code uses pd but forgot the import.
    if 'pd.' in code and 'import pandas as pd' not in code:
        code = "import pandas as pd\n" + code

    # Remove re-imports of the pre-injected toolkit (causes ImportError)
    code = re.sub(
        r"^\s*from\s+seismo_code\.toolkit\s+import\s+.*$",
        "pass  # toolkit pre-injected",
        code, flags=re.MULTILINE,
    )
    # Detect direct GMT subprocess calls (wrong — no begin/end wrapper)
    # Replace with a RuntimeError so the debug loop rewrites it with run_gmt()
    if re.search(r"subprocess\.\w+\(\s*\[.{0,20}['\"]gmt['\"]", code):
        code = re.sub(
            r"subprocess\.\w+\(\s*\[.{0,20}['\"]gmt['\"][^\n]*",
            "raise RuntimeError('SAGE_HINT: do NOT call gmt via subprocess directly"
            " — use run_gmt(script) with a complete gmt begin/end bash script')",
            code,
        )

    # Remove -G land-fill from gmt coast when grdimage is also present.
    # -Gcolor fills land with solid color, completely hiding the terrain underneath.
    if re.search(r'gmt\s+grdimage', code) and re.search(r'gmt\s+coast.*-G\w', code):
        # Strip -Gcolor/-Gwhite/-Gtan etc. from coast lines inside string literals
        code = re.sub(r'(gmt\s+coast\b[^\n]*?)\s+-G[a-zA-Z/0-9@]+', r'\1', code)

    # Detect bash array/loop syntax inside Python strings that will cause SyntaxError
    # Pattern: ${arr[$i]} or ${var[@]} or <<EOF inside any string literal
    if re.search(r'\$\{[A-Za-z_]+\[', code) or re.search(r'<<\s*EOF', code):
        hint = (
            "raise RuntimeError('SAGE_HINT: GMT f-string conflict — "
            "bash syntax \\${arr[$i]} or <<EOF inside Python string. "
            "Write data with np.savetxt() in Python, then use f-string with "
            "{python_var} for Python values and ${{bash_var}} for bash variables.')"
        )
        # inject the hint at the top of the script so it fails immediately
        code = hint + "\n" + code

    # Fix common GMT f-string escaping mistakes inside generated scripts.
    if 'gmt' in code and re.search(r"f(['\"]{3})", code):
        code = re.sub(r"awk\s+(['\"])\{print\s+([^}]+)\}(['\"])",
                      r"awk \1{{print \2}}\3", code)
        code = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", r"${{\1}}", code)

    # Detect incorrect sage import (toolkit is pre-injected)
    if re.search(r'from\s+sage\s+import|import\s+sage\b', code):
        code = re.sub(
            r'^\s*(from\s+sage\s+import.*|import\s+sage\b.*)$',
            "pass  # SAGE_HINT: toolkit pre-injected — never import from sage",
            code, flags=re.MULTILINE,
        )

    return code


# ---------------------------------------------------------------------------
# ── Data classes ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

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
    diagnosis:    str = ""   # last debug diagnosis, or "" on first-try success
    skipped:      bool = False   # True when a required dependency failed


@dataclass
class WorkflowRunResult:
    """Aggregate result of a multi-step workflow run."""
    workflow_name:  str
    workflow_title: str
    success:        bool           # True only if ALL steps succeeded
    steps_total:    int
    steps_done:     int            # steps that completed successfully
    step_results:   List[StepResult]
    all_figures:    List[str]      # union of figures across steps
    all_output_files: List[str]    # union of non-figure outputs
    response:       str            # human-readable summary
    exec_dir:       str = ""       # shared working directory

    @property
    def failed_steps(self) -> List[StepResult]:
        return [s for s in self.step_results if not s.success and not s.skipped]

    @property
    def skipped_steps(self) -> List[StepResult]:
        return [s for s in self.step_results if s.skipped]


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
        python_executable: Optional[str] = None,
    ):
        if llm_config is None:
            llm_config = self._load_llm_config()
        self.llm_config   = llm_config
        self.project_root = project_root or str(Path(__file__).parent.parent)
        self.python_executable = python_executable or llm_config.get('python_executable')
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
            config = LLMConfigManager().get_llm_config()
            # Use current Python executable (same environment as Flask app)
            # This avoids segfaults from using a different Python with incompatible BLAS
            if 'python_executable' not in config:
                import sys as _sys
                config['python_executable'] = _sys.executable
            return config
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
        """Execute code (Python or bash) and return result."""
        if _is_bash_code(code):
            # Strip the lang tag before passing to the shell
            clean = re.sub(r"^#\s*lang:bash\s*\n", "", code, count=1)
            return execute_bash(
                clean,
                project_root=self.project_root,
                timeout=timeout,
                keep_dir=True,
            )
        # Python path — apply sanitizer first
        clean = _pre_sanitize(code)
        return execute_code(
            clean,
            project_root=self.project_root,
            timeout=timeout,
            keep_dir=True,
            python_executable=self.python_executable,
        )

    def _execution_success(self, exec_res: ExecutionResult) -> bool:
        """Detect whether the execution truly succeeded, including silent failures."""
        if not exec_res or not exec_res.success:
            return False

        combined = "\n".join([exec_res.stdout or "", exec_res.stderr or ""])
        combined = combined.strip()
        if not combined:
            return True

        # Detect Python traceback or explicit error messages in output
        if re.search(r"Traceback \(most recent call last\):", combined, re.I):
            return False
        if re.search(r"^\s*(Error|Exception|AssertionError|ValueError|TypeError|NameError|ImportError|ModuleNotFoundError|FileNotFoundError|OSError)[:\s]", combined, re.M):
            return False
        return True

    def _build_error_context(self, code: str, exec_res: ExecutionResult) -> str:
        """Build a compact error context string for the debugger.

        Detects whether the failure is Python-side or Bash/GMT-side and
        annotates accordingly so the debugger can apply the right fix strategy.
        """
        parts = []

        stderr = (exec_res.stderr or "").strip()
        stdout = exec_res.stdout.strip()

        # ── Classify error type ──────────────────────────────────────────────
        is_bash_error = bool(re.search(
            r"(gmt |command not found|exit status \d|CalledProcessError|"
            r"run_gmt|GMT warning|GMT error|bash:|/bin/sh:)",
            stderr + stdout, re.I
        ))
        is_python_error = bool(re.search(
            r"(Traceback \(most recent call last\)|Error:|Exception:|"
            r"SyntaxError|IndentationError|NameError|TypeError|ValueError)",
            stderr
        ))

        if is_bash_error and not is_python_error:
            parts.append("=== ERROR TYPE: Bash/GMT script failure ===")
        elif is_python_error:
            parts.append("=== ERROR TYPE: Python runtime error ===")

        if stdout:
            parts.append("=== Partial stdout (last 1500 chars) ===\n" +
                         stdout[-1500:])

        if stderr:
            parts.append("=== Traceback / stderr ===\n" + stderr[-3000:])

        if exec_res.error:
            parts.append("=== Error summary ===\n" + exec_res.error)

        # ── Extract GMT-specific last error line for quick diagnosis ─────────
        if is_bash_error:
            gmt_errors = re.findall(
                r"(?:GMT (?:Error|Warning)|error|Error).*", stderr, re.I
            )
            if gmt_errors:
                parts.append("=== GMT/Bash key error lines ===\n" +
                             "\n".join(gmt_errors[-5:]))

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
        file_contexts: Optional[List[str]] = None,
        extra_rag_ctx: str = "",
    ) -> tuple[str, ExecutionResult, str]:
        """
        Ask the LLM debugger to fix the failing code.

        Returns (fixed_code, new_exec_result, diagnosis)
        """
        error_ctx = self._build_error_context(failed_code, exec_res)

        # Include file context so debugger knows exact column names etc.
        file_ctx_str = ""
        if file_contexts:
            file_ctx_str = "\n\n## Data file context (use EXACT column names shown here)\n" + \
                           "\n\n".join(file_contexts)

        # Build debug system prompt — optionally enriched with error-targeted RAG docs
        debug_system = _DEBUG_SYSTEM
        if extra_rag_ctx:
            debug_system += (
                "\n\n## Error-targeted documentation (retrieved for this specific error)\n"
                + extra_rag_ctx
                + "\n\nConsult the documentation above to resolve API misuse, "
                "wrong parameter names, or version-specific syntax errors."
            )

        debug_messages = [
            {"role": "system", "content": debug_system},
            {"role": "user", "content": (
                f"## Original user request\n{original_request}"
                f"{file_ctx_str}\n\n"
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
                profile = _profile_file(fp, self.project_root, self.python_executable)
                file_contexts.append(_format_file_context(profile))

        # ── 2. Build user message ─────────────────────────────────────────────
        msg_content = user_request
        if data_hint:
            msg_content += f"\n\nData path: {data_hint}"
        if file_contexts:
            msg_content += "\n\n" + "\n\n".join(file_contexts)

        self._history.append({"role": "user", "content": msg_content})

        # ── Inject multi-skill + RAG context into system prompt ─────────────
        # Use the raw user request as-is — no cartopy bias. Let keyword scores
        # decide which skills are most relevant.
        try:
            skill_ctx, rag_ctx = _build_skill_context_with_rag(
                user_request, max_skill_chars=12000, max_rag_chars=4000, top_k=5
            )
        except Exception:
            skill_ctx, rag_ctx = "", ""   # graceful degradation — skill context unavailable
        system_content = _CODEGEN_SYSTEM
        if skill_ctx:
            # Count how many skill sections were injected
            _n_skills = skill_ctx.count("### 技能：")
            if _n_skills > 1:
                system_content += (
                    "\n\n## Relevant skill docs\n"
                    + skill_ctx
                    + "\n\n## How to combine these skills\n"
                    "The docs above may cover different aspects of the task. "
                    "Identify which functions/patterns from each skill apply, "
                    "then integrate them into a single coherent script. "
                    "Import only what is needed; resolve any API conflicts by "
                    "preferring the most specific skill for each sub-task."
                )
            else:
                system_content += "\n\n## Relevant skill docs\n" + skill_ctx
        if rag_ctx:
            system_content += (
                "\n\n## Knowledge Base (RAG) — relevant documentation excerpts\n"
                + rag_ctx
                + "\n\nUse the above documentation to verify correct API usage, "
                "parameter names, and version-specific syntax before writing code."
            )
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
        while not self._execution_success(exec_res) and attempt < max_debug_rounds:
            attempt += 1
            error_summary = f"{exec_res.stdout}\n{exec_res.stderr}\n{exec_res.error}".strip()

            debug_trace.append(DebugAttempt(
                attempt=attempt,
                diagnosis="",         # filled below
                code=code,
                error=error_summary,
                stdout=exec_res.stdout,
                success=False,
            ))

            # Re-query RAG with error context to retrieve docs relevant to the
            # specific failure (e.g. wrong GMT module name, bad API call, etc.)
            _err_query = f"{user_request} {error_summary[:400]}"
            try:
                _, _debug_rag_ctx = _build_skill_context_with_rag(
                    _err_query, max_skill_chars=1, max_rag_chars=3000, top_k=3
                )
            except Exception:
                _debug_rag_ctx = ""

            fixed_code, new_exec, diagnosis = self._debug_and_fix(
                original_request=user_request,
                failed_code=code,
                exec_res=exec_res,
                attempt=attempt,
                timeout=timeout,
                on_progress=on_progress,
                file_contexts=file_contexts,
                extra_rag_ctx=_debug_rag_ctx,
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

        # ── 5. Update conversation history (code + execution result) ────────────
        final_success = self._execution_success(exec_res)
        result_summary = "Execution " + ("succeeded." if final_success else "failed.")
        if exec_res and exec_res.figures:
            result_summary += "\nGenerated figures: " + str(
                [Path(f).name for f in exec_res.figures])
        if exec_res and exec_res.output_files:
            result_summary += "\nOutput files: " + str(
                [Path(f).name for f in exec_res.output_files])
        if exec_res and exec_res.stdout.strip():
            clean_out = "\n".join(
                l for l in exec_res.stdout.splitlines()
                if not l.startswith('[FIGURE]') and not l.startswith('[GMT_SCRIPT]')
            ).strip()
            if clean_out:
                result_summary += f"\nOutput (truncated):\n{clean_out[:400]}"
        if exec_res and not final_success:
            err = (exec_res.stderr or exec_res.error or "").strip()
            if err:
                result_summary += f"\nError:\n{err[:300]}"
        self._history.append({
            "role": "assistant",
            "content": f"```python\n{code}\n```\n\n[Result] {result_summary}"
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
        final_success = self._execution_success(exec_res)
        response = self._build_response(exec_res, total_attempts, verify_pass, verify_note, final_success)

        self._emit(on_progress, "done", attempt, response)

        return CodeRunResult(
            success=final_success,
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

    # ── Shared-directory execution helper ────────────────────────────────────
    def _run_code_in_dir(
        self,
        code: str,
        timeout: int,
        shared_dir: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code, optionally inside a pre-existing shared directory.

        When *shared_dir* is set every step's code starts with:
          os.chdir(shared_dir)
          os.environ['SAGE_OUTDIR'] = shared_dir
        so file I/O from all steps lands in the same place.
        """
        clean = _pre_sanitize(code)
        extra_env: Optional[Dict[str, str]] = None
        if shared_dir:
            chdir_preamble = (
                f"import os as _wf_os\n"
                f"_wf_os.chdir({shared_dir!r})\n"
                f"_wf_os.environ['SAGE_OUTDIR'] = {shared_dir!r}\n"
            )
            clean = chdir_preamble + clean
            extra_env = {"SAGE_OUTDIR": shared_dir}
        return execute_code(
            clean,
            project_root=self.project_root,
            timeout=timeout,
            keep_dir=True,
            extra_env=extra_env,
            python_executable=self.python_executable,
        )

    # ── Workflow helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _topo_sort(steps: List[Dict]) -> List[Dict]:
        """
        Return steps in a valid topological order based on `depends_on` lists.
        Preserves original order for steps at the same depth.
        Falls back to original order on cyclic / broken deps.
        """
        order:     List[Dict] = []
        remaining: Dict[str, Dict] = {s["id"]: s for s in steps}
        done:      set = set()

        while remaining:
            ready = [
                sid for sid, s in remaining.items()
                if all(d in done for d in s.get("depends_on", []))
            ]
            if not ready:
                # Cycle or broken ref — append the rest in file order
                order.extend(remaining[s["id"]] for s in steps if s["id"] in remaining)
                break
            # Among ready steps, preserve original list order
            for s in steps:
                if s["id"] in ready and s["id"] in remaining:
                    order.append(remaining.pop(s["id"]))
                    done.add(s["id"])

        return order

    def _build_step_prompt(
        self,
        step:          Dict,
        workflow:      Dict,
        step_index:    int,
        steps_total:   int,
        available_files: List[str],
        completed_steps: List[StepResult],
        user_request:  str,
    ) -> tuple[str, str]:
        """
        Build (system_content, user_message) for a single workflow step.

        Returns
        -------
        (system_content, user_message)
        """
        step_id  = step["id"]
        skill_nm = step.get("skill", "")
        desc     = step["description"]

        # ── Skill context for this specific step ─────────────────────────────
        step_query = f"{skill_nm} {desc} {user_request}"
        skill_ctx, rag_ctx = _build_skill_context_with_rag(
            step_query, max_skill_chars=8000, max_rag_chars=2000, top_k=3
        )

        system_content = _CODEGEN_SYSTEM
        if skill_ctx:
            system_content += f"\n\n## 当前步骤技能文档\n{skill_ctx}"
        if rag_ctx:
            system_content += (
                f"\n\n## Knowledge Base (RAG)\n{rag_ctx}\n\n"
                "Use the above documentation for correct API and parameter usage."
            )

        # ── Workflow context (guide trimmed to ~2000 chars) ───────────────────
        guide_excerpt = workflow.get("guide", "")[:2000]
        if len(workflow.get("guide", "")) > 2000:
            guide_excerpt += "\n...(截断)"

        # ── Previous step summary ─────────────────────────────────────────────
        prev_summary = ""
        if completed_steps:
            lines = []
            for sr in completed_steps:
                status = "✓" if sr.success else "✗"
                files  = ", ".join(Path(f).name for f in sr.figures + sr.output_files)
                files  = f" → 输出: {files}" if files else ""
                lines.append(f"  {status} {sr.step_id} [{sr.skill}]: {sr.description}{files}")
            prev_summary = "## 已完成步骤\n" + "\n".join(lines)

        # ── Available files ───────────────────────────────────────────────────
        files_str = ""
        if available_files:
            files_str = "## 工作目录中的文件（可直接读取）\n" + \
                        "\n".join(f"  {f}" for f in available_files[:20])

        # ── User message ──────────────────────────────────────────────────────
        user_msg = (
            f"# 工作流：{workflow['name']} — {workflow['title']}\n"
            f"# 当前步骤 {step_index+1}/{steps_total}: [{step_id}] {desc}\n"
            f"# 使用技能：{skill_nm or '(通用)'}\n\n"
        )
        if user_request:
            user_msg += f"## 用户原始需求\n{user_request}\n\n"
        if prev_summary:
            user_msg += prev_summary + "\n\n"
        if files_str:
            user_msg += files_str + "\n\n"
        if guide_excerpt:
            user_msg += f"## 工作流参考指南（节选）\n{guide_excerpt}\n\n"
        user_msg += (
            f"## 当前任务\n"
            f"请为步骤 `{step_id}` 生成并执行 Python/Bash 代码：{desc}\n\n"
            "输出代码块，不需要解释。"
        )

        return system_content, user_msg

    # ── Workflow runner ───────────────────────────────────────────────────────
    def run_workflow(
        self,
        workflow_name: str,
        user_request: str = "",
        data_hint: Optional[str] = None,
        max_debug_rounds: int = 3,
        timeout: int = 120,
        skip_on_failure: bool = False,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> WorkflowRunResult:
        """
        Execute a workflow step-by-step in topological order.

        Roles
        -----
        workflow   — declares steps, skills, order (the .md file)
        skill      — per-step operation manual (injected into system prompt)
        code engine — generates & debugs code for each step (this method)
        tool       — Python / GMT / Shell executed by safe_executor

        Each step shares a single working directory so files written in
        earlier steps (e.g. ``epicenter.txt``) are readable by later ones.

        Parameters
        ----------
        workflow_name : str
            Workflow name as declared in the .md frontmatter ``name:`` field.
        user_request : str, optional
            Original user task (provides extra context for code generation).
        data_hint : str, optional
            Data directory / file path forwarded to each step.
        max_debug_rounds : int
            Max debug attempts per step (default 3).
        timeout : int
            Execution timeout per step in seconds (default 120).
        skip_on_failure : bool
            If True, skip steps whose dependencies failed instead of aborting
            the entire workflow (default False).
        on_progress : callable, optional
            Called with dicts:
              { "phase": "workflow_step" | "step_done" | "workflow_done",
                "step_id": str, "step_n": int, "total": int, "message": str }

        Returns
        -------
        WorkflowRunResult
        """
        # ── 0. Load workflow ─────────────────────────────────────────────────
        try:
            _sys_path = str(Path(__file__).parent.parent)
            import sys as _sys
            if _sys_path not in _sys.path:
                _sys.path.insert(0, _sys_path)
            from seismo_skill.workflow_runner import load_workflow
            workflow = load_workflow(workflow_name)
        except Exception as e:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title="",
                success=False, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"无法加载工作流 '{workflow_name}': {e}",
            )

        if workflow is None:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title="",
                success=False, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"工作流 '{workflow_name}' 不存在",
            )

        steps_raw: List[Dict] = workflow.get("steps", [])
        if not steps_raw:
            return WorkflowRunResult(
                workflow_name=workflow_name,
                workflow_title=workflow["title"],
                success=True, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"工作流 '{workflow_name}' 没有定义步骤",
            )

        # ── 1. Topological sort ──────────────────────────────────────────────
        ordered_steps = self._topo_sort(steps_raw)
        steps_total   = len(ordered_steps)

        # ── 2. State tracking ────────────────────────────────────────────────
        step_results:  List[StepResult] = []
        all_figures:   List[str]        = []
        all_output_files: List[str]     = []
        failed_step_ids: set            = set()
        shared_dir:    Optional[str]    = None  # set after first step runs
        # Per-workflow LLM history — shared across all steps so the model
        # can reference earlier outputs and variable bindings.
        wf_history: List[Dict] = []

        def _emit_wf(phase: str, step_id: str, step_n: int, msg: str):
            if on_progress:
                try:
                    on_progress({
                        "phase": phase, "step_id": step_id,
                        "step_n": step_n, "total": steps_total,
                        "message": msg,
                    })
                except Exception:
                    pass

        # ── 3. Execute steps ─────────────────────────────────────────────────
        for step_n, step in enumerate(ordered_steps):
            step_id   = step["id"]
            skill_nm  = step.get("skill", "")
            desc      = step["description"]
            deps      = step.get("depends_on", [])

            # ── 3a. Skip if a required dependency failed ─────────────────────
            failed_deps = [d for d in deps if d in failed_step_ids]
            if failed_deps:
                msg = f"跳过步骤 {step_id}（依赖步骤失败: {', '.join(failed_deps)}）"
                _emit_wf("workflow_step", step_id, step_n, msg)
                step_results.append(StepResult(
                    step_id=step_id, skill=skill_nm, description=desc,
                    success=False, code="", skipped=True,
                    diagnosis=f"依赖失败: {', '.join(failed_deps)}",
                ))
                failed_step_ids.add(step_id)
                if not skip_on_failure:
                    break
                continue

            # ── 3b. Discover files already in the shared workspace ────────────
            available_files: List[str] = []
            if shared_dir and Path(shared_dir).exists():
                try:
                    available_files = sorted(
                        str(p) for p in Path(shared_dir).iterdir()
                        if p.is_file() and not p.name.startswith("run.")
                    )
                except Exception:
                    pass

            _emit_wf("workflow_step", step_id, step_n,
                     f"[{step_n+1}/{steps_total}] 生成步骤 {step_id}…")

            # ── 3c. Build step prompt ─────────────────────────────────────────
            completed = [r for r in step_results if r.success]
            sys_content, user_msg = self._build_step_prompt(
                step=step, workflow=workflow,
                step_index=step_n, steps_total=steps_total,
                available_files=available_files,
                completed_steps=completed,
                user_request=user_request + (f"\nData: {data_hint}" if data_hint else ""),
            )

            # ── 3d. Build messages (system + shared history) ──────────────────
            messages = [{"role": "system", "content": sys_content}] + wf_history + \
                       [{"role": "user", "content": user_msg}]

            # ── 3e. Generate code ─────────────────────────────────────────────
            try:
                raw_response = _call_llm(messages, self.llm_config)
            except ConnectionError as e:
                step_results.append(StepResult(
                    step_id=step_id, skill=skill_nm, description=desc,
                    success=False, code="", diagnosis=str(e),
                ))
                failed_step_ids.add(step_id)
                if not skip_on_failure:
                    break
                continue

            code = _extract_code(raw_response)

            # ── 3f. Execute ───────────────────────────────────────────────────
            _emit_wf("workflow_step", step_id, step_n,
                     f"[{step_n+1}/{steps_total}] 执行步骤 {step_id}…")
            exec_res   = self._run_code_in_dir(code, timeout, shared_dir)
            debug_trace: List[DebugAttempt] = []
            attempt    = 0
            diagnosis  = ""

            # Set shared dir from first step's exec dir
            if shared_dir is None and exec_res.exec_dir:
                shared_dir = exec_res.exec_dir

            # ── 3g. Debug loop ────────────────────────────────────────────────
            while not self._execution_success(exec_res) and attempt < max_debug_rounds:
                attempt += 1
                error_summary = f"{exec_res.stdout}\n{exec_res.stderr}\n{exec_res.error}".strip()

                _emit_wf("workflow_step", step_id, step_n,
                         f"[{step_n+1}/{steps_total}] 调试步骤 {step_id} (第 {attempt} 轮)…")

                # Error-targeted RAG re-query
                _err_query = f"{skill_nm} {desc} {error_summary[:300]}"
                try:
                    _, _dbg_rag = _build_skill_context_with_rag(
                        _err_query, max_skill_chars=1, max_rag_chars=2000, top_k=3
                    )
                except Exception:
                    _dbg_rag = ""

                file_contexts = []
                if available_files:
                    file_contexts = [f"Available: {f}" for f in available_files[:5]]

                fixed_code, new_exec, diagnosis = self._debug_and_fix(
                    original_request=f"{desc}\n{user_request}",
                    failed_code=code,
                    exec_res=exec_res,
                    attempt=attempt,
                    timeout=timeout,
                    on_progress=None,
                    file_contexts=file_contexts,
                    extra_rag_ctx=_dbg_rag,
                )
                # Keep running in shared dir
                if shared_dir:
                    new_exec = self._run_code_in_dir(fixed_code, timeout, shared_dir)

                debug_trace.append(DebugAttempt(
                    attempt=attempt, diagnosis=diagnosis,
                    code=fixed_code, error=error_summary,
                    stdout=exec_res.stdout, success=False,
                ))

                code     = fixed_code
                exec_res = new_exec

                if self._execution_success(exec_res):
                    break

            step_success = self._execution_success(exec_res)

            # ── 3h. Collect outputs ───────────────────────────────────────────
            step_figs  = exec_res.figures     if exec_res else []
            step_files = exec_res.output_files if exec_res else []
            all_figures.extend(step_figs)
            all_output_files.extend(step_files)

            sr = StepResult(
                step_id=step_id, skill=skill_nm, description=desc,
                success=step_success,
                code=code,
                stdout=(exec_res.stdout or "")[:2000] if exec_res else "",
                stderr=(exec_res.stderr or "")[:1000] if exec_res else "",
                figures=step_figs,
                output_files=step_files,
                attempts=attempt + 1,
                diagnosis=diagnosis,
            )
            step_results.append(sr)

            if not step_success:
                failed_step_ids.add(step_id)
                _emit_wf("step_done", step_id, step_n,
                         f"✗ 步骤 {step_id} 失败（共 {attempt+1} 次尝试）")
                if not skip_on_failure:
                    break
            else:
                _emit_wf("step_done", step_id, step_n,
                         f"✓ 步骤 {step_id} 完成"
                         + (f"，输出 {len(step_figs)} 图" if step_figs else ""))
                # Append to shared history so next step's LLM sees what was done
                wf_history.append({"role": "user", "content": user_msg})
                step_summary = (
                    f"步骤 {step_id} 完成。"
                    + (f" 生成文件: {', '.join(Path(f).name for f in step_figs+step_files)}" if step_figs+step_files else "")
                    + (f"\n关键输出:\n{exec_res.stdout.strip()[:500]}" if exec_res and exec_res.stdout.strip() else "")
                )
                wf_history.append({
                    "role": "assistant",
                    "content": f"```python\n{code}\n```\n\n[步骤结果] {step_summary}",
                })

        # ── 4. Build summary ─────────────────────────────────────────────────
        steps_done    = sum(1 for r in step_results if r.success)
        wf_success    = steps_done == steps_total and bool(step_results)
        skipped_count = sum(1 for r in step_results if r.skipped)

        lines = [f"工作流 **{workflow['name']}** — {workflow['title']}"]
        lines.append(f"进度：{steps_done}/{steps_total} 步完成"
                     + (f"，{skipped_count} 步跳过" if skipped_count else ""))
        for sr in step_results:
            icon = "✓" if sr.success else ("↷" if sr.skipped else "✗")
            lines.append(f"  {icon} [{sr.step_id}] {sr.description}"
                         + (f"（{sr.attempts} 次尝试）" if sr.attempts > 1 and not sr.skipped else ""))
        if all_figures:
            lines.append(f"生成图片：{len(all_figures)} 张")
        if all_output_files:
            lines.append(f"生成文件：{len(all_output_files)} 个")
        if shared_dir:
            lines.append(f"工作目录：{shared_dir}")

        response = "\n".join(lines)
        _emit_wf("workflow_done", "", steps_total, response)

        # Sync exec dir to engine (so follow-up single-step runs continue there)
        if shared_dir:
            self._last_exec_dir = shared_dir

        return WorkflowRunResult(
            workflow_name=workflow_name,
            workflow_title=workflow["title"],
            success=wf_success,
            steps_total=steps_total,
            steps_done=steps_done,
            step_results=step_results,
            all_figures=all_figures,
            all_output_files=all_output_files,
            response=response,
            exec_dir=shared_dir or "",
        )

    def _build_response(
        self,
        exec_res: Optional[ExecutionResult],
        attempts: int,
        verify_pass: Optional[bool],
        verify_note: str,
        success: Optional[bool] = None,
    ) -> str:
        if not exec_res:
            return "Execution failed — no result."

        if success is None:
            success = exec_res.success

        lines = []
        if success:
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
