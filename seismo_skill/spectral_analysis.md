---
name: spectral_analysis
category: analysis
keywords: spectrum, power spectrum, amplitude spectrum, HVSR, spectral ratio method, site response, predominant frequency, Fourier, FFT, compute_spectrum, compute_hvsr, plot_psd, 频谱, 功率谱, 振幅谱, 谱比法, 场地响应, 卓越周期
---

# Spectral Analysis and HVSR

## Description

Compute amplitude spectrum, power spectrum, and horizontal-to-vertical spectral ratio (HVSR) for site response and dominant frequency analysis.

---

## Main Functions

### `compute_spectrum(tr, method="fft")`

Compute amplitude spectrum for a single trace.

**Parameters:**
- `tr` : obspy.Trace
- `method` : str — `"fft"` (Fast Fourier Transform)

**Returns:** `(freqs, amps)` — numpy arrays of frequency (Hz) and corresponding amplitude

```python
tr = st.select(channel="HHZ")[0]
freqs, amps = compute_spectrum(tr)

import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.semilogy(freqs, amps)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Amplitude spectrum")
plt.xlim(0.1, 50)
plt.grid(True, alpha=0.3)
plt.savefig("spectrum.png", dpi=150)
print(f"Peak frequency: {freqs[np.argmax(amps)]:.2f} Hz")
```

---

### `compute_hvsr(st, method="konno_ohmachi", f_min=0.1, f_max=20.0, smooth_coeff=40, window_length=20.0, overlap=0.5)`

Compute horizontal-to-vertical spectral ratio (HVSR / H/V) for estimating site predominant frequency and amplification.

**Parameters:**
- `st` : obspy.Stream — Must contain three components (Z + two horizontals)
- `method` : str — Smoothing method, `"konno_ohmachi"` / `"constant"` / `"proportional"`
- `f_min` : float — Lower frequency limit (Hz), default 0.1
- `f_max` : float — Upper frequency limit (Hz), default 20.0
- `smooth_coeff` : float — Konno-Ohmachi smoothing coefficient b, default 40
- `window_length` : float — Window length (seconds), default 20.0
- `overlap` : float — Window overlap ratio, default 0.5

**Returns:** `(freqs, hvsr_mean, hvsr_std)` — Frequency array, H/V mean, standard deviation

```python
st = read_stream_from_dir("/data/station/", pattern="**/*.mseed")
st = detrend_stream(st)
st = taper_stream(st)

freqs, hvsr, hvsr_std = compute_hvsr(
    st,
    method="konno_ohmachi",
    f_min=0.1,
    f_max=20.0,
    window_length=30.0,
    overlap=0.5,
)

# Find predominant frequency
import numpy as np
f0_idx = np.argmax(hvsr)
f0 = freqs[f0_idx]
print(f"Site predominant frequency f0 = {f0:.3f} Hz  (T0 = {1/f0:.2f} s)")
print(f"H/V at f0 = {hvsr[f0_idx]:.2f} ± {hvsr_std[f0_idx]:.2f}")

# Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(freqs, hvsr, "b-", lw=2, label="H/V mean")
ax.fill_between(freqs, hvsr - hvsr_std, hvsr + hvsr_std, alpha=0.3, color="blue", label="±1σ")
ax.axvline(f0, color="red", ls="--", label=f"f0 = {f0:.3f} Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("H/V ratio")
ax.set_title("HVSR spectral ratio curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("hvsr.png", dpi=150, bbox_inches="tight")
print("HVSR plot saved: hvsr.png")
```

---

## PSD Compared with Global Noise Model

```python
# Plot PSD and compare with NLNM/NHNM (built-in reference models)
tr = st.select(channel="*Z")[0]
fig = plot_psd(tr, outfile="psd_nlnm.png")
print("PSD plot saved: psd_nlnm.png")
```

---

## Multi-station Spectrum Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 5))
for tr in st:
    freqs, amps = compute_spectrum(tr)
    ax.semilogy(freqs, amps, label=tr.id, alpha=0.8)

ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")
ax.set_title("Multi-station amplitude spectrum comparison")
ax.legend(fontsize=8)
ax.set_xlim(0.1, 50)
ax.grid(True, alpha=0.3)
plt.savefig("multi_spectrum.png", dpi=150, bbox_inches="tight")
```

---

## Notes

- HVSR analysis requires record length >= 10 times window length; recommend >= 30 minutes of background noise
- Three-component sampling rate must be consistent; use `resample_stream` to unify if needed
- `f_min` should be >= 3 / record_length to avoid insufficient frequency resolution
