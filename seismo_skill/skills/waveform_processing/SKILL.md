---
name: waveform_processing
category: waveform
keywords: filtering, bandpass filter, lowpass filter, highpass filter, detrend, demean, resample, taper, instrument response, preprocessing, filter_stream, detrend_stream, resample_stream, trim_stream, 滤波, 带通滤波, 去趋势, 去均值, 重采样, 仪器响应, 预处理
---

# Waveform Preprocessing

## Description

Perform standard preprocessing operations on ObsPy Stream/Trace: detrending, demeaning, tapering, filtering, resampling, and windowing.

---

## Main Functions

### `detrend_stream(st, type="demean")`

Remove trend or mean from waveform.

**Parameters:**
- `st` : obspy.Stream
- `type` : str — `"demean"` (remove mean) / `"linear"` (remove linear trend) / `"constant"` (same as demean)

**Returns:** obspy.Stream (modified in place and returned)

```python
st = read_stream("/data/wave.mseed")
st = detrend_stream(st, type="demean")     # Remove mean
st = detrend_stream(st, type="linear")    # Remove linear trend
```

---

### `taper_stream(st, max_percentage=0.05, type="cosine")`

Apply cosine taper window to both ends of waveform to reduce spectral leakage.

**Parameters:**
- `st` : obspy.Stream
- `max_percentage` : float — Taper fraction of total length, default 0.05 (5%)
- `type` : str — Window type, `"cosine"` / `"hann"` / `"hamming"`, etc.

```python
st = taper_stream(st, max_percentage=0.05)
```

---

### `filter_stream(st, filter_type, freqmin=None, freqmax=None, corners=4, zerophase=True)`

Apply frequency domain filtering to Stream.

**Parameters:**
- `st` : obspy.Stream
- `filter_type` : str — `"bandpass"` / `"lowpass"` / `"highpass"` / `"bandstop"`
- `freqmin` : float — Low cut frequency (Hz), required for bandpass/highpass
- `freqmax` : float — High cut frequency (Hz), required for bandpass/lowpass
- `corners` : int — Filter order, default 4
- `zerophase` : bool — Zero-phase filtering (forward + reverse), default True

**Returns:** obspy.Stream

```python
st = read_stream("/data/wave.mseed")
st = detrend_stream(st)
st = taper_stream(st)

# Bandpass 1-10 Hz
st_bp = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

# Lowpass 5 Hz
st_lp = filter_stream(st, "lowpass", freqmax=5.0)

# Highpass 0.5 Hz
st_hp = filter_stream(st, "highpass", freqmin=0.5)
```

---

### `resample_stream(st, sampling_rate)`

Resample to specified sampling rate.

**Parameters:**
- `st` : obspy.Stream
- `sampling_rate` : float — Target sampling rate (Hz)

**Returns:** obspy.Stream

```python
# Downsample to 50 Hz
st = resample_stream(st, sampling_rate=50.0)
```

---

### `trim_stream(st, starttime=None, endtime=None, pad=True)`

Extract specified time window.

**Parameters:**
- `st` : obspy.Stream
- `starttime` : obspy.UTCDateTime or str, e.g. `"2024-01-01T00:00:00"`
- `endtime` : obspy.UTCDateTime or str
- `pad` : bool — Pad with zeros if insufficient data, default True

```python
from obspy import UTCDateTime
t0 = UTCDateTime("2024-03-15T08:30:00")
st = trim_stream(st, starttime=t0, endtime=t0 + 60)  # Extract 60 seconds
```

---

### `merge_stream(st)`

Merge multiple traces of the same channel (fill gaps).

**Returns:** obspy.Stream

```python
st = merge_stream(st)
```

---

### `remove_response(st, inventory_or_paz, output="VEL", pre_filt=None)`

Remove instrument response and convert to physical quantity (displacement / velocity / acceleration).

**Parameters:**
- `st` : obspy.Stream
- `inventory_or_paz` : obspy.Inventory or PAZ dict
- `output` : str — `"DISP"` (displacement) / `"VEL"` (velocity) / `"ACC"` (acceleration)
- `pre_filt` : tuple — Water-level pre-filtering, e.g. `(0.005, 0.01, 45, 50)`

```python
from obspy import read_inventory
inv = read_inventory("station.xml")
st = remove_response(st, inv, output="VEL", pre_filt=(0.005, 0.01, 45, 50))
```

---

## Standard Preprocessing Workflow

```python
st = read_stream("/data/wave.mseed")

# Standard four-step preprocessing
st = detrend_stream(st, type="demean")
st = detrend_stream(st, type="linear")
st = taper_stream(st, max_percentage=0.05)
st_filtered = filter_stream(st, "bandpass", freqmin=1.0, freqmax=10.0)

print("Preprocessing complete")
stream_info(st_filtered)
```

---

## Notes

- Always detrend and taper before filtering, otherwise edge effects will contaminate the spectrum
- `zerophase=True` introduces no phase shift and is strongly recommended in seismology
- Apply lowpass filtering (cutoff < new sampling rate / 2) before resampling to avoid aliasing
