---
name: waveform_io
category: waveform
keywords: read waveform, load waveform, mseed, sac, miniseed, seed, waveform file, read_stream, stream_info, picks_to_dict, csv, txt, 读取波形, 加载波形, 波形文件
---

# Waveform Reading and Information Query

## Description

Read local waveform files (SAC / MiniSEED / SEED, etc.), retrieve stream information, and load phase pick results.

---

## ⚠️ Critical Note: Toolkit Functions Require No Import

Functions like `read_stream`, `read_stream_from_dir`, `stream_info`, and `picks_to_dict` have been
**pre-injected into the execution environment** via `from seismo_code.toolkit import *` and can be **called directly**:

```python
# ✅ Correct: Call directly, no import needed
st = read_stream_from_dir("/data/event_001/")
```

**❌ Forbidden patterns (will raise ImportError):**
```python
# Error: read_stream_from_dir is NOT an obspy function!
from obspy import read_stream_from_dir

# Error: Already pre-injected, no need to import again
from seismo_code.toolkit import read_stream_from_dir
```

---

## Main Functions

### `read_stream(path)`

Read a single waveform file and return an `obspy.Stream`.

**Parameters:**
- `path` : str — Path to waveform file (supports .mseed / .sac / .seed / .msd, etc.)

**Returns:** `obspy.Stream`

```python
# ✅ Call directly (no import needed)
st = read_stream("/data/station/HHZ.mseed")
print(st)
# Output: 1 Trace(s) in Stream: NET.STA.LOC.HHZ ...
```

---

### `read_stream_from_dir(directory, pattern="**/*")`

Recursively read all waveform files in a directory (SAC / MiniSEED / SEED all supported) and merge into a single Stream.

**Parameters:**
- `directory` : str — Directory path
- `pattern` : str — Glob pattern, default `**/*` (**reads all formats**, recommended to use default)

**Returns:** `obspy.Stream`

> ⚠️ **Important**: `read_stream` can only read **a single file** and does not support glob (`*`) paths.
> When reading an entire directory, **must use** `read_stream_from_dir(directory)`.

```python
# ✅ Correct: Read all waveforms in directory (call directly, no import needed)
st = read_stream_from_dir("/data/event_001/")
print(f"Loaded {len(st)} traces")

# ✅ Can also read only .sac files
st = read_stream_from_dir("/data/event_001/", pattern="**/*.sac")

# ❌ Error: Cannot pass glob path to read_stream
# st = read_stream("/data/event_001/*.mseed")  # Will error!

# ❌ Error: Cannot import this function from obspy
# from obspy import read_stream_from_dir  # ImportError!
```

---

### Native ObsPy Alternative (Reading Single SAC File)

If you only need to read a single file, you can also use native obspy:

```python
from obspy import read   # obspy.read is a legitimate obspy function

# Read a single file
st = read("/data/event_001/YN.YSW03..HHZ.sac")
tr = st[0]
print(f"Station: {tr.id}, Sampling rate: {tr.stats.sampling_rate} Hz")
print(f"Duration: {tr.stats.npts / tr.stats.sampling_rate:.1f} seconds")
```

---

### `stream_info(st)`

Print detailed statistical information about a Stream (stations, channels, time range, sampling rate).

```python
# ✅ Call directly (no import needed)
st = read_stream_from_dir("/data/event_001/")
info = stream_info(st)
print(info)
```

---

### `picks_to_dict(picks_file)`

Read phase pick results from a CSV file.

```python
# ✅ Call directly (no import needed)
picks = picks_to_dict("results/picks.csv")
for p in picks:
    print(f"{p['phase']}  t={p['rel_time']:.2f}s  conf={p['confidence']:.2f}")
```

---

## Notes

- All toolkit functions are pre-injected; **do not use `from obspy import ...` to import them**
- Use `from obspy import read` when native obspy functionality is needed (this is obspy's own function)
- Three-component files (HHZ/HHN/HHE) should be placed in the same directory and loaded at once using `read_stream_from_dir`
- If the file path does not exist, a `FileNotFoundError` will be raised; wrap in `try/except` for safety
