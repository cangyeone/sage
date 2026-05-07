---
name: pnsn_phase_detection
description: Use this skill for seismic phase picking, earthquake monitoring, Pg/Sg/Pn/Sn detection, P/S arrival picking, continuous waveform scanning, pnsn deep-learning pickers, PhaseNet/EQTransformer/RNN/LPPN pickers, and downstream phase association with FastLink, REAL, or GaMMA. Trigger when the user asks to detect phases, pick arrivals, monitor earthquakes, process continuous 3-component SAC/MSEED/SEED waveforms, or associate picks into earthquake events.
---

# PNSN Phase Detection and Earthquake Monitoring

This skill wraps the local `pnsn/` project for automatic seismic phase picking and event monitoring. Use it when a task involves:

- detecting or picking Pg, Sg, Pn, Sn, P, or S arrivals;
- running deep-learning pickers on continuous 3-component waveforms;
- scanning SAC, MSEED, SEED, or miniseed waveform directories;
- converting picks into earthquake events with FastLink, REAL, or GaMMA;
- building an earthquake monitoring workflow from waveform data.

Project root assumptions:

- Run commands from the SAGE repo root.
- The pnsn code lives at `pnsn/`.
- Default models live at `pnsn/pickers/`.
- Picker configuration lives at `pnsn/config/picker.py`.

## Default Tool Choice

Prefer the TorchScript pnsn v3 picker for most tasks:

```bash
python pnsn/picker.py \
  -i /path/to/waveforms \
  -o outputs/picks/pnsn_picks \
  -m pnsn/pickers/pnsn.v3.jit \
  -d cpu
```

Use `pnsn/pickers/pnsn.v3.diff.jit` when the user wants the paper differential-input model or when high-frequency transients are important. Use `pnsn/pickers/pnsn.v1.jit` only for legacy engineering compatibility.

For GPU, set `-d cuda:0`; otherwise use `-d cpu` for safer local execution.

## Required Input Format

The picker expects one station to have exactly three synchronized components in the same directory.

Typical assumptions:

- sampling rate: 100 Hz;
- channels: E/N/Z, commonly `BHE/BHN/BHZ`;
- waveform extensions: usually `.sac`, `.mseed`, `.seed`;
- one station group should not mix many time segments in the same directory unless `config/picker.py` has been adapted.

Before running, inspect a few filenames and update `pnsn/config/picker.py` if needed:

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

`pnsn/picker.py` writes three files using the `-o` prefix:

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

model_path = Path("pnsn/pickers/pnsn.v3.jit")
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

2. Adjust `pnsn/config/picker.py` if suffix, channel names, or filename token indices differ.

3. Run picker:

```bash
mkdir -p outputs/pnsn
python pnsn/picker.py \
  -i /path/to/waveforms \
  -o outputs/pnsn/picks \
  -m pnsn/pickers/pnsn.v3.jit \
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
python pnsn/fastlinker.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_fastlink.txt \
  -s /path/to/stations.txt \
  -d cpu
```

REAL:

```bash
python pnsn/reallinker.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_real \
  -s /path/to/stations.txt
```

GaMMA:

```bash
python pnsn/gammalink.py \
  -i outputs/pnsn/picks.txt \
  -o outputs/pnsn/events_gamma.txt \
  -s /path/to/stations.txt \
  -d cpu
```

Prefer FastLink for quick monitoring workflows, REAL for classical association/location-style workflows, and GaMMA when probabilistic association is requested.

## Model Selection

- `pnsn/pickers/pnsn.v3.jit`: default Pg/Sg/Pn/Sn picker.
- `pnsn/pickers/pnsn.v3.diff.jit`: differential-input pnsn v3 model.
- `pnsn/pickers/pnsn.v1.jit`: legacy engineering model.
- `pnsn/pickers/phasenet.jit`: fast Pg/Sg-style PhaseNet picker.
- `pnsn/pickers/eqtransformer.stead.jit`: EQTransformer-style picker.
- `pnsn/pickers/rnn.jit`: RNN picker, useful for high-recall Pg/Sg workflows.
- `pnsn/pickers/lppnt.jit`, `lppnm.jit`, `lppnl.jit`: lightweight LPPN variants.

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
- any configuration changes made to `pnsn/config/picker.py`;
- command or Python script executed;
- number of picks/events generated;
- paths to `.txt`, `.log`, `.err`, figures, and associated event catalogs;
- warnings about skipped files, sampling rate mismatch, or missing components.

