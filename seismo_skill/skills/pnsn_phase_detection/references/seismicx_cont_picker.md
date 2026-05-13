# SeismicX-Cont Continuous Picker Reference

This reference describes how to use the PNSN phase-detection skill with
SeismicX-Cont style continuous HDF5 data. It is intentionally stored inside the
skill folder so chat, science-analysis, and parameter-optimization agents can
use it even when a demo folder such as `publish_mini` is not present.

## When to Use

Use this reference when the task mentions:

- `publish_mini`, SeismicX-Cont, continuous HDF5 waveform data, or `data/hdf5`;
- `picker.py`, `run_picker_to_jsonl.py`, `pnsn.v3.diff.jit`, `phasenet.jit`, or
  `skynet.jit`;
- phase-pick recall, precision, residuals, benchmark statistics, or annotation
  plots;
- converting a continuous waveform folder into a research dataset.

## Data Location

The large waveform data are not bundled in the repository. Download them from:

[https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont](https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont)

Suggested command:

```bash
modelscope download --dataset cangyeone/SeismicX-Cont --local_dir /path/to/SeismicX-Cont
```

Expected project layout:

```text
project_root/
  data/
    hdf5/
      continuous_waveform_*.h5
    label/
      annotations_mini_two_hours.json
    index/
      waveform_index.sqlite              # optional
    picks/
      pnsn.v3.diff.phase.jsonl           # generated
  pickers/
    phasenet.jit                         # optional local models
    pnsn.v3.diff.jit
    skynet.jit
  scripts/
    run_picker_to_jsonl.py               # optional project script
    evaluate_picks.py                    # optional project script
  utils/
    hdf5_waveform_dataset.py             # optional project dataloader
```

If project scripts are missing, generate equivalent code using the contracts
below. Do not require `publish_mini` itself to exist.

## Output Pick JSONL Contract

Write one JSON object per pick. Keep fields stable so evaluation, plotting, and
association workflows can consume the same file.

```json
{
  "record_type": "phase_pick",
  "station_id": "CI.CLC.--",
  "phase_name": "Pg",
  "phase_time": "2019-07-06T04:13:08.824000Z",
  "phase_prob": 0.92,
  "h5_file": "data/hdf5/continuous_waveform_usa_20190706_04.h5",
  "sample_index": 12345,
  "sampling_rate": 100.0,
  "model": "pnsn.v3.diff.jit"
}
```

Phase map used by the PNSN/mini scripts:

- `0`: Pg
- `1`: Sg
- `2`: Pn
- `3`: Sn
- `4`: P
- `5`: S

## Recommended Picker Execution

If a project has `scripts/run_picker_to_jsonl.py`, prefer it because it already
knows the local dataloader and resume semantics:

```bash
python scripts/run_picker_to_jsonl.py \
  --h5_input "data/hdf5/continuous_waveform_*.h5" \
  --picker_model seismo_skill/skills/pnsn_phase_detection/pnsn/pickers/pnsn.v3.diff.jit \
  --picker_backend pnsn \
  --output_jsonl data/picks/pnsn.v3.diff.phase.jsonl \
  --device cpu \
  --batch_size 1 \
  --num_workers 0 \
  --target_sampling_rate 100 \
  --canonical_input_length 360000 \
  --min_confidence 0.3 \
  --resume
```

If the project does not have that script, write a custom Python script that:

1. loads HDF5 files with `utils/hdf5_waveform_dataset.py` when available;
2. otherwise traverses HDF5 datasets with `h5py`;
3. normalizes waveform arrays to `[n_samples, 3]`;
4. resamples to 100 Hz when needed;
5. loads the TorchScript picker from the skill-local `pnsn/pickers/` folder;
6. writes JSONL and a small manifest;
7. prints `[SAGE_TEST]` with input counts, output counts, and file paths.

## Benchmark Metrics

When labels are available, compute at least:

- recall = matched reference picks / reference picks;
- precision = matched automatic picks / automatic picks;
- median residual and MAD residual by phase;
- per-phase recall and precision;
- per-station and per-hour recall;
- residual histogram and annotation overlay figures.

Recommended matching rule:

- Match by compatible station alias (`NET.STA.LOC`, `NET.STA`, `--`/`00` location
  aliases), phase family (`Pg/Pn/P` as P, `Sg/Sn/S` as S), and nearest time.
- Use a default tolerance of 1.5 s for recall/precision.
- Report unmatched reference picks separately from low-confidence auto picks.

## Mini Tests

Before a full run, execute a small smoke test:

```bash
python - <<'PY'
from pathlib import Path
import glob

root = Path(".")
h5 = glob.glob("data/hdf5/*.h5")
labels = root / "data/label/annotations_mini_two_hours.json"
print("[SAGE_TEST] h5_files", len(h5))
print("[SAGE_TEST] label_exists", labels.exists())
assert h5, "No HDF5 files found"
assert labels.exists(), "No label JSON found"
PY
```

For any generated picker code, verify:

- the first waveform sample can be loaded;
- waveform shape is `[n_samples, 3]`;
- sampling rate is known;
- output JSONL exists, even if it contains zero picks;
- a plot manifest exists when plotting is requested.

