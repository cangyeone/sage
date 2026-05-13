---
name: waveform_visualization
category: visualization
keywords: plot, draw, visualization, waveform plot, visualize, plot waveform, particle motion, plot_stream, plot_spectrogram, plot_psd, plot_particle_motion, savefig, 绘制, 画图, 波形图, 可视化, 粒子运动, 质点运动
---

# Waveform Visualization

## Description

Plot waveform time series, amplitude spectra, power spectral density, and particle motion diagrams, with support for phase pick annotations.

---

## ⚠️ Critical Note: Toolkit Functions Require No Import, plt.show() Forbidden

Functions like `plot_stream`, `plot_spectrogram`, `plot_psd`, `plot_particle_motion`, and `savefig`
have been **pre-injected into the execution environment** via `from seismo_code.toolkit import *` and can be **called directly**.

The server has **no display**, so `plt.show()` will cause the program to hang. **Must use** `savefig()` or `plot_*` functions to save images.

```python
# ✅ Correct workflow (no import needed)
st = read_stream_from_dir("/data/event_001/")
st = detrend_stream(st)
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)
plot_stream(st, title="Three-component waveform")   # Auto-save and display to interface

# ❌ Forbidden
# from obspy import plot_stream        # plot_stream is NOT an obspy function!
# plt.show()                           # Server has no display, will hang!
```

---

## Approach 1: Using Built-in Toolkit (Recommended)

### `plot_stream(st, title="", outfile=None, picks=None)`

Plot multi-component waveform (one trace per row, vertically arranged).

**Parameters:**
- `st` : obspy.Stream
- `title` : str — Figure title
- `outfile` : str — Save path; if None, auto-saves to SAGE_OUTDIR (**recommended to omit**)
- `picks` : list[dict] — Phase annotations

```python
# ✅ Call directly (no import needed)
st = read_stream_from_dir("/data/event_001/")
st = detrend_stream(st)
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

# Omit outfile → system auto-saves and displays to interface (recommended)
plot_stream(st, title="Three-component waveform")
```

---

## Plotting Existing Phase Picks

When the user asks to draw or overlay "this pick result" on a waveform, treat it
as a visualization task. Do **not** assume the file is named `picks_table.csv`,
do **not** choose arbitrary `data.csv` files, and do **not** re-pick the waveform
with STA/LTA unless the user explicitly asks for a new STA/LTA picking run.

Search in this order:

1. explicit file paths mentioned by the user or available in prior outputs;
2. `SAGE_OUTDIR`;
3. current execution directory;
4. authorized waveform/data directories and their parent directories.

Candidate pick filenames include `*pick*.csv`, `*pick*.txt`, `pnsn_picks.csv`,
`sage_picks_*.txt`, `phase_picks.*`, and `picks.*`. A candidate must contain
pick-like columns or rows before use.

Accepted pick schemas:

- CSV/table columns:
  - phase: `phase`, `phase_name`, `type`
  - absolute time: `time_abs`, `absolute_time`, `time`, `timestamp`
  - relative time: `time_rel_s`, `relative_time_s`, `t`, `arrival_time_s`
  - optional: `station`, `channel`, `confidence`, `snr`, `amp`
- PNSN text format:

```text
# path/to/waveform/file
phase_name,relative_time_s,confidence,absolute_time,SNR,AMP,station,extra
```

Skip comment lines beginning with `#`; parse comma-separated data rows. If the
first row is not a header, infer the PNSN order above.

Minimal robust parser pattern:

```python
import csv
from pathlib import Path

def parse_pick_file(path):
    path = Path(path)
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    data_lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not data_lines:
        return rows

    header = [c.strip().lower() for c in data_lines[0].split(",")]
    has_header = any(c in header for c in ["phase", "phase_name", "time_abs", "absolute_time", "relative_time_s", "time_rel_s"])
    if has_header:
        for r in csv.DictReader(data_lines):
            row = {k.strip().lower(): v for k, v in r.items() if k}
            rows.append({
                "phase": row.get("phase") or row.get("phase_name") or row.get("type") or "",
                "time_abs": row.get("time_abs") or row.get("absolute_time") or row.get("time") or row.get("timestamp"),
                "time_rel_s": row.get("time_rel_s") or row.get("relative_time_s") or row.get("t") or row.get("arrival_time_s"),
                "station": row.get("station"),
                "confidence": row.get("confidence"),
            })
    else:
        # PNSN text order: phase_name, relative_time_s, confidence, absolute_time, SNR, AMP, station, extra
        for ln in data_lines:
            parts = [p.strip() for p in ln.split(",")]
            if len(parts) >= 4:
                rows.append({
                    "phase": parts[0],
                    "time_rel_s": parts[1],
                    "confidence": parts[2],
                    "time_abs": parts[3],
                    "station": parts[6] if len(parts) > 6 else None,
                })
    return [r for r in rows if r.get("time_abs") or r.get("time_rel_s")]
```

After parsing, pass the normalized list to `plot_stream(st, picks=picks,
title="Waveform with phase picks")`, or manually draw vertical lines at
`time_rel_s`. Print a concrete validation line:

If the same script just ran `PNSNPicker.pick_stream(st)`, the returned pick
dictionaries can be passed directly to `plot_stream`; they already include
`time_abs` and `time_rel_s`. Do not filter them out because the station label is
compound, for example `X1.53085.01`.

```python
print(f"[SAGE_TEST] plotted {len(picks)} picks from {pick_file}")
```

If no valid pick file is found, fail with a clear message listing searched
directories and candidate files. Do not silently use an unrelated table.

---

### `plot_spectrogram(tr, outfile=None, wlen=None, per_lap=0.9)`

Plot time-frequency spectrogram for single trace.

```python
st = read_stream_from_dir("/data/event_001/")
tr = st.select(channel="*Z")[0]   # Select vertical component
plot_spectrogram(tr)
```

---

### `plot_psd(tr, outfile=None)`

Plot power spectral density (PSD) curve.

```python
tr = st.select(channel="*Z")[0]
plot_psd(tr)
```

---

### `plot_particle_motion(st, outfile=None)`

Plot particle motion diagram (requires three-component data).

```python
st = read_stream_from_dir("/data/event_001/")
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)
plot_particle_motion(st)
```

---

## Approach 2: Using Native ObsPy + Matplotlib (Manual Plotting)

When complete custom plotting is needed, use native obspy to read data and matplotlib to plot.
**Note: Must use `savefig()` instead of `plt.show()`**

```python
from obspy import read   # obspy.read is a legitimate obspy function
import matplotlib.pyplot as plt

# Read a single file
st = read("/data/event_001/YN.YSW03..HHZ.sac")
tr = st[0]

times = tr.times()   # Relative time axis (seconds)
data = tr.data

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, data, linewidth=0.8, color='black')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude")
ax.set_title(f"{tr.id}")
ax.grid(True, alpha=0.3)
plt.tight_layout()

# ✅ Must use savefig() to save, cannot use plt.show()
savefig("waveform.png")   # savefig is pre-injected, auto-reports to interface
```

---

## Complete Chained Example: Read + Preprocess + Multi-plot Visualization in One Script

```python
# Single script chaining multiple skill steps (recommended approach)
st = read_stream_from_dir("/data/event_001/")
stream_info(st)                                          # Print station/channel info

st = detrend_stream(st)
st = taper_stream(st)
st = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

# Multi-component waveform plot
plot_stream(st, title="Event waveform (1-10 Hz)")

# Vertical component spectrogram + PSD
tr_z = st.select(channel="*Z")[0]
plot_spectrogram(tr_z)
freqs, psd, _ = plot_psd(tr_z)
import numpy as np
print("Peak frequency: " + str(round(float(freqs[np.argmax(psd)]), 2)) + " Hz")

# Particle motion
plot_particle_motion(st)
```

---

## Notes

- **All plot_* functions and savefig are pre-injected, no import needed, and cannot be imported from obspy**
- Server environment must set `show=False` or omit (default no popup); **forbidden to call plt.show()**
- Particle motion plotting requires three-component data (Z/N/E or Z/1/2); skip missing projections
- When outfile is omitted, auto-saves to `SAGE_OUTDIR` and outputs `[FIGURE] /path` for interface capture
