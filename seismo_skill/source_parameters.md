---
name: source_parameters
category: analysis
keywords: magnitude, moment magnitude, seismic moment, source parameters, corner frequency, stress drop, local magnitude, ML, estimate_magnitude_ml, estimate_corner_freq, estimate_seismic_moment, moment_to_mw, estimate_stress_drop, 震级, 矩震级, 地震矩, 震源参数, 角频率, 应力降, ML震级
---

# Source Parameter Estimation

## Description

Estimate local magnitude (ML), seismic moment (M₀), moment magnitude (Mw), corner frequency (fc), and stress drop (Δσ), calculated directly from ObsPy Trace.

---

## Main Functions

### `estimate_magnitude_ml(tr, dist_km, station_correction=0.0)`

Estimate local magnitude (ML, Richter 1935) based on Wood-Anderson simulation.

**Parameters:**
- `tr` : obspy.Trace — Velocity record (single component)
- `dist_km` : float — Epicentral distance (kilometers)
- `station_correction` : float — Station correction value, default 0

**Returns:** float — ML magnitude

```python
tr = st.select(channel="HHZ")[0]
dist_km = 85.3  # Epicentral distance

ml = estimate_magnitude_ml(tr, dist_km=dist_km)
print(f"ML = {ml:.2f}")
```

---

### `estimate_corner_freq(tr, dist_km, method="brune", density=2700, vp=6000)`

Estimate corner frequency (fc) using Brune source model.

**Parameters:**
- `tr` : obspy.Trace — Velocity record (within P-wave window)
- `dist_km` : float — Epicentral distance (kilometers)
- `method` : str — Source model, `"brune"`
- `density` : float — Medium density (kg/m³), default 2700
- `vp` : float — P-wave velocity (m/s), default 6000

**Returns:** `(fc, omega0)` — Corner frequency (Hz) and low-frequency spectral amplitude Ω₀

```python
fc, omega0 = estimate_corner_freq(tr, dist_km=dist_km)
print(f"Corner frequency fc = {fc:.2f} Hz")
print(f"Low-frequency spectral amplitude Omega0 = {omega0:.3e}")
```

---

### `estimate_seismic_moment(tr, dist_km, density=2700, vp=6000, radiation=0.52)`

Estimate seismic moment M₀ using far-field displacement spectrum.

**Parameters:**
- `tr` : obspy.Trace — Displacement record (instrument response removed)
- `dist_km` : float — Epicentral distance (kilometers)
- `density` : float — Medium density (kg/m³)
- `vp` : float — P-wave velocity (m/s)
- `radiation` : float — Radiation pattern factor, default 0.52 for P-waves

**Returns:** float — Seismic moment M₀ (N·m)

```python
M0 = estimate_seismic_moment(tr, dist_km=dist_km)
print(f"Seismic moment M0 = {M0:.3e} N·m")
```

---

### `moment_to_mw(M0)`

Convert seismic moment to moment magnitude.

**Formula:** Mw = (2/3) × log₁₀(M₀) − 6.07

**Parameters:**
- `M0` : float — Seismic moment (N·m)

**Returns:** float — Moment magnitude Mw

```python
Mw = moment_to_mw(M0)
print(f"Moment magnitude Mw = {Mw:.2f}")
```

---

### `estimate_stress_drop(M0, fc, vs=3500)`

Estimate static stress drop Δσ using Brune model.

**Formula:** Δσ = (7/16) × M₀ × (fc/vs)³

**Parameters:**
- `M0` : float — Seismic moment (N·m)
- `fc` : float — Corner frequency (Hz)
- `vs` : float — S-wave velocity (m/s), default 3500

**Returns:** float — Stress drop (MPa)

```python
delta_sigma = estimate_stress_drop(M0, fc, vs=3500)
print(f"Stress drop Delta_sigma = {delta_sigma:.2f} MPa")
```

---

## Complete Source Parameter Analysis Example

```python
from obspy import read_inventory

# 1. Read waveform and remove instrument response
st = read_stream("/data/event.mseed")
inv = read_inventory("/data/station.xml")

st_vel = st.copy()
st_vel = detrend_stream(st_vel)
st_vel = taper_stream(st_vel)
st_vel = remove_response(st_vel, inv, output="VEL", pre_filt=(0.01, 0.02, 45, 50))

st_disp = st.copy()
st_disp = detrend_stream(st_disp)
st_disp = taper_stream(st_disp)
st_disp = remove_response(st_disp, inv, output="DISP", pre_filt=(0.01, 0.02, 45, 50))

# 2. Select vertical component
dist_km = 120.0
tr_vel  = st_vel.select(channel="*Z")[0]
tr_disp = st_disp.select(channel="*Z")[0]

# 3. Calculate source parameters
ML  = estimate_magnitude_ml(tr_vel, dist_km=dist_km)
fc, omega0 = estimate_corner_freq(tr_vel, dist_km=dist_km)
M0  = estimate_seismic_moment(tr_disp, dist_km=dist_km)
Mw  = moment_to_mw(M0)
ds  = estimate_stress_drop(M0, fc)

print(f"--- Source Parameter Summary ---")
print(f"Local magnitude ML = {ML:.2f}")
print(f"Moment magnitude Mw = {Mw:.2f}")
print(f"Seismic moment M0 = {M0:.3e} N·m")
print(f"Corner frequency fc = {fc:.2f} Hz")
print(f"Stress drop Delta_sigma = {ds:.2f} MPa")
```

---

## Notes

- `estimate_magnitude_ml` internally performs Wood-Anderson instrument simulation; input must be velocity record
- `estimate_seismic_moment` requires displacement record input (instrument response removed); otherwise results will be significantly biased
- Single-station estimates have large uncertainty; averaging across multiple stations recommended
