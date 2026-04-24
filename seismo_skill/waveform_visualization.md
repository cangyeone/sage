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
