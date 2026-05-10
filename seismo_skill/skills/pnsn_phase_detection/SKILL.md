---
name: pnsn_phase_detection
description: Use this skill for seismic phase picking, earthquake monitoring, Pg/Sg/Pn/Sn detection, P/S arrival picking, continuous waveform scanning, pnsn deep-learning pickers, PhaseNet/EQTransformer/RNN/LPPN pickers, and downstream phase association with FastLink, REAL, or GaMMA. Trigger when the user asks to detect phases, pick arrivals, monitor earthquakes, process continuous 3-component SAC/MSEED/SEED waveforms, associate picks into earthquake events, or autonomously write/debug custom Python picking code with component grouping, preprocessing, TorchScript inference, pick tables, figures, mini-tests, and self-check output.
---

# PNSN Phase Detection and Earthquake Monitoring

This skill wraps the skill-local `pnsn/` project stored at `seismo_skill/skills/pnsn_phase_detection/pnsn/` for automatic seismic phase picking and event monitoring. Use it when a task involves:

- detecting or picking Pg, Sg, Pn, Sn, P, or S arrivals;
- running deep-learning pickers on continuous 3-component waveforms;
- scanning SAC, MSEED, SEED, or miniseed waveform directories;
- converting picks into earthquake events with FastLink, REAL, or GaMMA;
- building an earthquake monitoring workflow from waveform data.

Project root assumptions:

- Run commands from the SAGE repo root.
- The pnsn code and model files are managed inside this skill folder, not at the repository root.
- If `seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py` is missing, instruct the user to run `git clone https://github.com/cangyeone/pnsn.git seismo_skill/skills/pnsn_phase_detection/pnsn` from the SAGE repo root.
- Default model files live at `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/`.
- Picker configuration lives at `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py`.
- Define `PNSN_ROOT = Path("seismo_skill/skills/pnsn_phase_detection/pnsn")` in generated code and build all model/script paths from it.

## Autonomous Programming Mode

This skill is not limited to calling `seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py`. When the user asks to "自己编程", "写代码实现拾取", "检测这几条波形", "画出拾取结果", "调试检测流程", or when the waveform organization does not match `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py`, write a custom Python program instead of only giving a CLI command.

Use custom code for:

- one station or a small set of files;
- uploaded SAC/MSEED files with known paths;
- non-standard filenames or component names;
- tasks requiring waveform plots, pick tables, SNR/AMP statistics, or intermediate diagnostics;
- debugging and mini-tests;
- integrating picks into a larger SAGE analysis pipeline.

Custom picking code should implement this workflow:

1. Discover waveform files.
2. Group files into 3-component station sets.
3. Read with ObsPy.
4. Align traces to a common time window.
5. Detrend, taper, bandpass, and resample to 100 Hz.
6. Stack data as `[n_samples, 3]`.
7. Load a TorchScript picker with `torch.jit.load`.
8. Run inference under `torch.no_grad()`.
9. Convert sample indices to relative and absolute times.
10. Save a CSV/text pick table.
11. Generate a waveform figure with vertical pick markers.
12. Print `[SAGE_TEST]` and file paths.

Prefer writing all outputs to `SAGE_OUTDIR` when available:

```python
import os
from pathlib import Path

OUTDIR = Path(os.environ.get("SAGE_OUTDIR", "outputs/pnsn_phase_detection"))
OUTDIR.mkdir(parents=True, exist_ok=True)
PNSN_ROOT = Path("seismo_skill/skills/pnsn_phase_detection/pnsn")
if not (PNSN_ROOT / "picker.py").exists():
    raise FileNotFoundError(
        "Missing skill-local pnsn/. Run: git clone https://github.com/cangyeone/pnsn.git "
        "seismo_skill/skills/pnsn_phase_detection/pnsn"
    )
```

## Component Discovery Pattern

When writing custom code, do not assume exact filenames unless the user gives them. Search common waveform suffixes and group by station/time key.

```python
from pathlib import Path
from collections import defaultdict
import re

def discover_waveforms(root):
    root = Path(root).expanduser()
    suffixes = {".sac", ".SAC", ".mseed", ".MSEED", ".seed", ".SEED", ".miniseed"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes]

def component_of(path):
    name = path.name.upper()
    # Handles BHE/BHN/BHZ, HHE/HHN/HHZ, E/N/Z endings.
    for comp in ["BHE", "BHN", "BHZ", "HHE", "HHN", "HHZ", "EHE", "EHN", "EHZ"]:
        if comp in name:
            return comp[-1]
    m = re.search(r"([ENZ])(?:\.[^.]+)?$", name)
    return m.group(1) if m else None

def station_key(path):
    parts = path.name.split(".")
    # Default SAGE/PNSN style: NET.STA.LOC.CHN...
    if len(parts) >= 4:
        return ".".join(parts[:3])
    return path.parent.name

def group_three_components(files):
    groups = defaultdict(dict)
    for p in files:
        comp = component_of(p)
        if comp:
            groups[station_key(p)][comp] = p
    return {k: v for k, v in groups.items() if {"E", "N", "Z"} <= set(v)}
```

If no 3-component group is found, print a clear diagnostic with example filenames and recommend updating grouping logic or `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py`.

## Robust Custom Picker Template

Use this as the default pattern when coding the picking workflow yourself. Adapt `DATA_ROOT` when the user provides a path.

```python
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import obspy
import matplotlib.pyplot as plt

OUTDIR = Path(os.environ.get("SAGE_OUTDIR", "outputs/pnsn_phase_detection"))
OUTDIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path("/path/to/waveforms")
PNSN_ROOT = Path("seismo_skill/skills/pnsn_phase_detection/pnsn")
if not (PNSN_ROOT / "picker.py").exists():
    raise FileNotFoundError(
        "Missing skill-local pnsn/. Run: git clone https://github.com/cangyeone/pnsn.git "
        "seismo_skill/skills/pnsn_phase_detection/pnsn"
    )
MODEL_PATH = PNSN_ROOT / "pickers" / "pnsn.v3.jit"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PHASE_NAMES = {0: "Pg", 1: "Sg", 2: "Pn", 3: "Sn"}

def prepare_stream(paths, target_rate=100.0):
    st = obspy.Stream()
    for p in paths:
        st += obspy.read(str(p))
    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    st.taper(0.02)
    st.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
    if abs(st[0].stats.sampling_rate - target_rate) > 1e-6:
        st.resample(target_rate)
    start = max(tr.stats.starttime for tr in st)
    end = min(tr.stats.endtime for tr in st)
    if end <= start:
        raise ValueError("No overlapping time window among three components")
    st.trim(start, end, pad=False)
    return st

def stream_to_array(st):
    comp_map = {}
    for tr in st:
        ch = tr.stats.channel.upper()
        if ch.endswith("E"): comp_map["E"] = tr
        elif ch.endswith("N"): comp_map["N"] = tr
        elif ch.endswith("Z"): comp_map["Z"] = tr
    missing = {"E", "N", "Z"} - set(comp_map)
    if missing:
        raise ValueError(f"Missing components: {sorted(missing)}")
    n = min(len(comp_map[c].data) for c in ["E", "N", "Z"])
    x = np.stack([comp_map[c].data[:n] for c in ["E", "N", "Z"]], axis=1)
    return x.astype(np.float32), comp_map["Z"].stats.starttime, comp_map["Z"].stats.sampling_rate

def run_picker(x, model):
    with torch.no_grad():
        picks = model(torch.tensor(x, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    rows = []
    for phase_type, sample, confidence in picks:
        rows.append({
            "phase": PHASE_NAMES.get(int(phase_type), str(int(phase_type))),
            "sample": int(sample),
            "relative_time_s": float(sample) / 100.0,
            "confidence": float(confidence),
        })
    return rows

model = torch.jit.load(str(MODEL_PATH), map_location=DEVICE).to(DEVICE).eval()
files = discover_waveforms(DATA_ROOT)
groups = group_three_components(files)
print(f"[INFO] found_files={len(files)} three_component_groups={len(groups)}")

all_rows = []
for station, comps in sorted(groups.items()):
    try:
        paths = [comps["E"], comps["N"], comps["Z"]]
        st = prepare_stream(paths)
        x, starttime, fs = stream_to_array(st)
        rows = run_picker(x, model)
        for r in rows:
            r["station"] = station
            r["absolute_time"] = str(starttime + r["relative_time_s"])
            r["model"] = str(MODEL_PATH)
        all_rows.extend(rows)

        if rows:
            fig_path = OUTDIR / f"{station.replace('.', '_')}_picks.png"
            t = np.arange(len(x)) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(t, x[:, 2], color="0.25", lw=0.8)
            for r in rows:
                color = {"Pg": "tab:red", "Sg": "tab:blue", "Pn": "tab:green", "Sn": "black"}.get(r["phase"], "black")
                plt.axvline(r["relative_time_s"], color=color, alpha=0.85)
                plt.text(r["relative_time_s"], np.nanmax(x[:, 2]), r["phase"], color=color, rotation=90, va="top")
            plt.title(f"PNSN picks: {station}")
            plt.xlabel("Time (s)")
            plt.ylabel("Z amplitude")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[FIGURE] {fig_path}")
    except Exception as exc:
        print(f"[WARN] station={station} skipped: {exc}")

pick_table = OUTDIR / "pnsn_picks.csv"
pd.DataFrame(all_rows).to_csv(pick_table, index=False)
print(f"[SAGE_TEST] pnsn custom picker finished: stations={len(groups)} picks={len(all_rows)}")
print(f"[SAGE_TEST] pick_table={pick_table}")
```

The template references `discover_waveforms` and `group_three_components`; include those helper functions in the actual generated script.

## Mini-Tests for Self-Debugging

When generating custom code, add small tests before full execution:

```python
def _mini_test_grouping():
    fake = [
        Path("X1.53085.01.BHE.D.2012.sac"),
        Path("X1.53085.01.BHN.D.2012.sac"),
        Path("X1.53085.01.BHZ.D.2012.sac"),
    ]
    groups = group_three_components(fake)
    assert len(groups) == 1, groups
    assert {"E", "N", "Z"} <= set(next(iter(groups.values())))

_mini_test_grouping()
print("[SAGE_TEST] grouping mini-test passed")
```

Also check:

- model path exists before loading;
- at least one waveform file is found;
- at least one 3-component group is formed;
- sampling rate after preprocessing is 100 Hz;
- output table exists even if zero picks are found.

## Default Tool Choice

Prefer the TorchScript pnsn v3 picker for most tasks:

```bash
python seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py \
  -i /path/to/waveforms \
  -o outputs/picks/pnsn_picks \
  -m seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.jit \
  -d cpu
```

Use `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.diff.jit` when the user wants the paper differential-input model or when high-frequency transients are important. Use `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v1.jit` only for legacy engineering compatibility.

For GPU, set `-d cuda:0`; otherwise use `-d cpu` for safer local execution.

## Required Input Format

The picker expects one station to have exactly three synchronized components in the same directory.

Typical assumptions:

- sampling rate: 100 Hz;
- channels: E/N/Z, commonly `BHE/BHN/BHZ`;
- waveform extensions: usually `.sac`, `.mseed`, `.seed`;
- one station group should not mix many time segments in the same directory unless `config/picker.py` has been adapted.

Before running, inspect a few filenames and update `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py` if needed:

```python
class Parameter:
    nchannel = 3
    samplerate = 100
    filenametag = ".sac"
    namekeyindex = [0, 1]
    channelindex = 3
    chnames = [["BHE", "BHN", "BHZ"]]
    prob = 0.3
    nmslen = 1000
    bandpass = [1, 10]
    ifplot = False
```

Critical checks:

- `filenametag` must match the actual file suffix.
- `channelindex` must point to the filename token containing the component name.
- `namekeyindex` should identify the station grouping tokens.
- `chnames` must match the component triplets in the data.
- If the data are not 100 Hz, resample before picking or adapt the workflow carefully.

## Output Format

`seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py` writes three files using the `-o` prefix:

- `<output>.txt`: phase picks;
- `<output>.log`: processed data log;
- `<output>.err`: problematic files or skipped data.

Pick file format:

```text
# path/to/file
phase_name,relative_time_s,confidence,absolute_time,SNR,AMP,station,extra
```

For TorchScript direct inference, model output is:

```text
[[phase_type, relative_sample, confidence], ...]
```

Phase mapping for pnsn Pg/Sg/Pn/Sn models:

- `0`: Pg
- `1`: Sg
- `2`: Pn
- `3`: Sn

Convert sample index to seconds with `relative_sample / sampling_rate`.

## Direct TorchScript Inference Pattern

Use this when the user gives one 3-component station waveform and wants a focused result or figure.

```python
from pathlib import Path
import numpy as np
import torch
import obspy
import matplotlib.pyplot as plt

PNSN_ROOT = Path("seismo_skill/skills/pnsn_phase_detection/pnsn")
if not (PNSN_ROOT / "picker.py").exists():
    raise FileNotFoundError(
        "Missing skill-local pnsn/. Run: git clone https://github.com/cangyeone/pnsn.git "
        "seismo_skill/skills/pnsn_phase_detection/pnsn"
    )
model_path = PNSN_ROOT / "pickers" / "pnsn.v3.jit"
device = torch.device("cpu")
model = torch.jit.load(str(model_path), map_location=device).eval()

tr_e = obspy.read("STA.BHE.sac")[0]
tr_n = obspy.read("STA.BHN.sac")[0]
tr_z = obspy.read("STA.BHZ.sac")[0]

st = obspy.Stream([tr_e, tr_n, tr_z])
st.detrend("demean")
st.detrend("linear")
st.taper(0.02)
st.filter("bandpass", freqmin=1.0, freqmax=10.0, corners=4, zerophase=True)
if abs(st[0].stats.sampling_rate - 100.0) > 1e-6:
    st.resample(100.0)

x = np.stack([st[0].data, st[1].data, st[2].data], axis=1).astype(np.float32)

with torch.no_grad():
    picks = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()

phase_names = {0: "Pg", 1: "Sg", 2: "Pn", 3: "Sn"}
for phase_type, sample, confidence in picks:
    print(phase_names.get(int(phase_type), str(int(phase_type))),
          float(sample) / 100.0,
          float(confidence))

plt.figure(figsize=(12, 4))
t = np.arange(len(x)) / 100.0
plt.plot(t, x[:, 2], lw=0.8, color="0.25")
for phase_type, sample, confidence in picks:
    color = {0: "tab:red", 1: "tab:blue", 2: "tab:green", 3: "black"}.get(int(phase_type), "black")
    plt.axvline(float(sample) / 100.0, color=color, alpha=0.8,
                label=f"{phase_names.get(int(phase_type), int(phase_type))} {confidence:.2f}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("PNSN phase picks")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("pnsn_phase_picks.png", dpi=200)
print("[FIGURE] pnsn_phase_picks.png")
print("[SAGE_TEST] pnsn phase picking finished")
```

Always include a small self-check in generated code:

- print number of waveforms loaded;
- print sampling rate and component names;
- print number of picks;
- write `[SAGE_TEST]` on success;
- emit `[FIGURE] path` for generated figures.

## Batch Picking Workflow

Use this when the user asks to monitor a directory or process many stations.

1. Inspect the waveform directory:

```bash
find /path/to/waveforms -maxdepth 2 -type f | head
```

2. Adjust `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py` if suffix, channel names, or filename token indices differ.

3. Run picker:

```bash
mkdir -p outputs/pnsn
python seismo_skill/skills/pnsn_phase_detection/pnsn/picker.py \
  -i /path/to/waveforms \
  -o outputs/pnsn/picks \
  -m seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.jit \
  -d cpu
```

4. Verify:

```bash
wc -l outputs/pnsn/picks.txt
tail -20 outputs/pnsn/picks.log
tail -20 outputs/pnsn/picks.err
```

Report how many picks were produced and whether any files were skipped.

## Event Association

After picking, associate phases into earthquake events if the user asks for monitoring, catalog building, event detection, or earthquake location.

Station file format for FastLink:

```text
NET STA LOC longitude latitude elevation
SC AXX 00 110.00 38.00 1000.00
```

FastLink:

```bash
python seismo_skill/skills/pnsn_phase_detection/pnsn/fastlinker.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_fastlink.txt \
  -s /path/to/stations.txt \
  -d cpu
```

REAL:

```bash
python seismo_skill/skills/pnsn_phase_detection/pnsn/reallinker.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_real \
  -s /path/to/stations.txt
```

GaMMA:

```bash
python seismo_skill/skills/pnsn_phase_detection/pnsn/gammalink.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_gamma.txt \
  -s /path/to/stations.txt \
  -d cpu
```

Prefer FastLink for quick monitoring workflows, REAL for classical association/location-style workflows, and GaMMA when probabilistic association is requested.

## Model Selection

- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.jit`: default Pg/Sg/Pn/Sn picker.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.diff.jit`: differential-input pnsn v3 model.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v1.jit`: legacy engineering model.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/phasenet.jit`: fast Pg/Sg-style PhaseNet picker.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/eqtransformer.stead.jit`: EQTransformer-style picker.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/rnn.jit`: RNN picker, useful for high-recall Pg/Sg workflows.
- `seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/lppnt.jit`, `lppnm.jit`, `lppnl.jit`: lightweight LPPN variants.

Do not use `tele.jit` as the default teleseismic picker. Prefer the pnsn family for a unified local/regional/distant workflow unless the user explicitly asks for the tele model.

## Common Failure Modes

- **No picks:** check sampling rate, component order, channel names, and bandpass settings; try `pnsn.v3.diff.jit` or lower the threshold for ONNX workflows.
- **Many skipped stations:** fewer or more than three components are present in each station directory; reorganize files or update `config/picker.py`.
- **CUDA error:** rerun with `-d cpu`.
- **Poor grouping:** `namekeyindex` or `channelindex` does not match filename tokens.
- **Need probability traces:** use ONNX models and external post-processing instead of `.jit`.
- **FastLink asks about existing temporary files:** remove `fastdata/` or handle the prompt before running unattended.

## Response Expectations

When using this skill, the assistant should provide:

- the exact model path used;
- the waveform directory or files processed;
- any configuration changes made to `seismo_skill/skills/pnsn_phase_detection/pnsn/config/picker.py`;
- command or Python script executed;
- number of picks/events generated;
- paths to `.txt`, `.log`, `.err`, figures, and associated event catalogs;
- warnings about skipped files, sampling rate mismatch, or missing components.
