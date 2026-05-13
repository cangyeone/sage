# SeismicX-Cont Mini Two-Hour Subset

This folder is a compact, Zenodo-friendly trial release for SeismicX-Cont. It is
designed for quick download, tutorial use, software smoke tests, and checking
that the HDF5, annotation, SQLite, dataloader, picker, and evaluation workflow
all fit together before using the full 14-day benchmark.

## Relation to the Full Dataset

This mini release is a two-hour subset of the complete SeismicX-Cont benchmark.
It is intended for quick trials and examples, not for reporting full benchmark
results. The complete dataset should be downloaded from ModelScope:

```text
https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont
```

The mini release contains only two one-hour windows:

- Dense Ridgecrest hour: `2019-07-06 04:00-05:00 UTC`, 6948 reference picks.
- Quiet-period hour: `2021-11-14 16:00-17:00 UTC`, 7 reference picks.

The mini package is about 3.0 GiB. The two waveform files are about 1.7 GiB
and 1.2 GiB, and the response JSON is about 119 MiB, so all examples below are
intended to run on a laptop.

## 1. Folder Layout

Run commands from this folder unless noted otherwise:

```bash
cd publish_mini
```

Main files:

```text
data/
  hdf5/
    continuous_waveform_usa_20190706_04.h5
    continuous_waveform_usa_20211114_16.h5
  index/
    waveform_index.sqlite
  label/
    annotations_mini_two_hours.json
    stations.csv
  response/
    instrument_responses_mini.json
  picks/
    # generated picker JSONL files are written here
notebooks/
  seismicx_cont_mini_quickstart.ipynb
pickers/
  phasenet.jit
  pnsn.v3.diff.jit
  skynet.jit
scripts/
  verify_mini_subset.py
  example_query_mini.py
  example_dataloader.py
  run_picker_to_jsonl_mini.sh
  run_pickers_mini.sh
  run_evaluate_picks_mini.sh
  run_consensus_picks_mini.sh
utils/
  hdf5_waveform_dataset.py
  hdf5_waveform_index.py
  waveform_index_api.py
```

## 2. Environment

Use the same Python environment as the complete SeismicX-Cont release. The basic
inspection examples need `h5py`, `numpy`, and `torch`; picker and evaluation
examples also use the dependencies imported by the scripts.

Quick check:

```bash
python - <<'PY'
import h5py, numpy, torch
print("h5py", h5py.__version__)
print("numpy", numpy.__version__)
print("torch", torch.__version__)
PY
```

If shell scripts are not executable after download, run:

```bash
chmod +x scripts/*.sh scripts/*.py
```

The scripts also respect a `PYTHON` environment variable. For example:

```bash
PYTHON=/Users/anaconda3/bin/python3 ./scripts/verify_mini_subset.py
```

## 3. First Five-Minute Check

These commands verify that the package is complete and that the SQLite index can
find waveform segments.

```bash
python scripts/verify_mini_subset.py
python scripts/example_query_mini.py --limit 5
```

Expected high-level counts from `verify_mini_subset.py`:

```text
HDF5 files: 2
waveform_segments: 22908
stations: 971
n_picks: 6955
2019_dense: 6948
2021_quiet: 7
```

If these counts are close to the above values, the mini release is ready for the
examples below.

## 4. Query Waveforms From SQLite

The SQLite database is a waveform coverage index. It tells you which HDF5 file
and dataset path contain a requested station-channel-time interval.

List a few raw segment records:

```bash
python scripts/example_query_mini.py \
  --starttime 2019-07-06T04:00:00 \
  --endtime 2019-07-06T04:05:00 \
  --limit 10
```

Query and merge one channel into an in-memory waveform array:

```bash
python scripts/example_query_waveform.py \
  --db data/index/waveform_index.sqlite \
  --network BK \
  --station BDM \
  --channel BHZ \
  --starttime 2019-07-06T04:00:00 \
  --endtime 2019-07-06T04:01:00 \
  --limit 3
```

This is the simplest way to confirm that `data/index/waveform_index.sqlite` and
`data/hdf5/*.h5` are mutually consistent.

## 5. Inspect Annotations

The two-hour annotation subset is:

```text
data/label/annotations_mini_two_hours.json
```

It preserves the original event, station, phase, status, and provenance fields
for the selected one-hour windows. A quick summary:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("data/label/annotations_mini_two_hours.json")
obj = json.loads(path.read_text())
print(json.dumps(obj["summary"], indent=2, ensure_ascii=False))
PY
```

Use the 2019 hour to test dense aftershock behavior and the 2021 hour to test
low-event-rate monitoring behavior. The 2021 hour is not pure noise; it is the
least labeled nonzero hour selected so that the quiet subset still has reference
annotations.

## 6. Use the PyTorch HDF5 Dataloader

Run the minimal dataloader example:

```bash
python scripts/example_dataloader.py --n_samples 2
```

The loader reads the HDF5 files, groups components by station-time window,
resamples to the requested sampling rate, and returns tensors in the form:

```text
waveform: torch.Tensor [time, 3]
station_id: network.station.location
channels: selected component channels
starttime: waveform start time
sampling_rate: output sampling rate
```

To point it at a single HDF5 file:

```bash
python scripts/example_dataloader.py \
  --h5_input data/hdf5/continuous_waveform_usa_20190706_04.h5 \
  --n_samples 2
```

The mini package also includes an instrument-response JSON derived from the
complete response inventory:

```text
data/response/instrument_responses_mini.json
```

Use it when you want the dataloader to remove the native instrument response:

```python
from utils.hdf5_waveform_dataset import HDF5WaveformDataset

dataset = HDF5WaveformDataset(
    h5_file="data/hdf5/continuous_waveform_usa_20190706_04.h5",
    mode="three",
    instrument_response_json="data/response/instrument_responses_mini.json",
    remove_instrument_response=True,
    response_output="VEL",
    response_pre_filt=(0.2, 0.5, 20.0, 45.0),
    response_water_level=None,
)

item = dataset[0]
print(item["waveform"].shape)
print(item["instrument_processing"])
```

To remove the native response and then simulate a target response, select the
target response by `response_id` or by station-channel metadata:

```python
dataset = HDF5WaveformDataset(
    h5_file="data/hdf5/continuous_waveform_usa_20190706_04.h5",
    mode="single",
    instrument_response_json="data/response/instrument_responses_mini.json",
    remove_instrument_response=True,
    response_output="VEL",
    response_pre_filt=(0.2, 0.5, 20.0, 45.0),
    response_water_level=None,
    simulate_instrument_response=True,
    simulation_response_selector={
        "network": "BK",
        "station": "BDM",
        "location": "00",
        "channel": "BHZ",
        "time": "2019-07-06T04:00:00Z",
    },
    simulation_output="VEL",
)
```

The response correction and simulation are computed with ObsPy. The mini
dataloader supplies the JSON lookup, station-channel-time matching, and the
`instrument_processing` metadata returned with each sample.

## 7. Run One Picker

The quickest picker test uses the bundled PhaseNet TorchScript model:

```bash
./scripts/run_picker_to_jsonl_mini.sh
```

Default output:

```text
data/picks/phasenet.mini.phase.jsonl
```

Run a different bundled model by passing the model path and output path:

```bash
./scripts/run_picker_to_jsonl_mini.sh \
  pickers/pnsn.v3.diff.jit \
  data/picks/pnsn_v3_diff.mini.phase.jsonl
```

For GPU or Apple Silicon MPS, set `DEVICE` when using the full three-model
script below, or pass `--device` directly to `scripts/run_picker_to_jsonl.py`.

## 8. Run the Three Bundled Pickers

This runs PhaseNet, PNSN v3 diff, and SkyNet on the two HDF5 files:

```bash
DEVICE=cpu ./scripts/run_pickers_mini.sh
```

Use `DEVICE=mps` on Apple Silicon or `DEVICE=cuda` on CUDA machines:

```bash
DEVICE=mps ./scripts/run_pickers_mini.sh
```

Generated JSONL files:

```text
data/picks/phasenet.mini.phase.jsonl
data/picks/pnsn_v3_diff.mini.phase.jsonl
data/picks/skynet.mini.phase.jsonl
```

Each JSONL line is one station-time inference record with picker outputs. These
files are generated products and are intentionally not included as required
input files.

## 9. Evaluate Picker Output

After generating a picker JSONL file, evaluate it against the mini annotation
subset:

```bash
./scripts/run_evaluate_picks_mini.sh \
  data/picks/pnsn_v3_diff.mini.phase.jsonl
```

Outputs are written under:

```text
eval_picks/example/
```

The evaluation script uses:

- `data/label/annotations_mini_two_hours.json` as the reference annotation set.
- `data/index/waveform_index.sqlite` as the waveform coverage index.
- A 1.5 s true-positive tolerance.
- A 5.0 s residual fitting window.

The mini evaluation is only a smoke test. It should not be interpreted as a
model ranking because the subset has just two one-hour windows.

## 10. Build Optional Consensus Candidates

If the three picker JSONL files exist, build consensus-supported candidate
arrivals:

```bash
./scripts/run_consensus_picks_mini.sh
```

Default output:

```text
data/label/consensus_mini_picks.json
```

The default rule requires at least three models to agree within 1.5 s on the
same station and phase. This layer is optional. It is useful for auditing
candidate detections and testing workflow extensions, but it is not the primary
benchmark label set.

## 11. Run REAL Association

The mini package also includes a bridge from SeismicX picker JSONL output to
the REAL association algorithm. It compiles `scripts/real/real.c`, converts
phase-pick JSONL records to REAL input files, runs association-based event
detection, and writes a workflow JSONL that contains both time-window phase
candidates and associated earthquake events. Downstream earthquake location can
be attached by users through the associated-arrival list in each event record.

Run from the `publish_mini/` folder:

```bash
python scripts/run_real_association.py \
  --picks-jsonl data/picks/pnsn_v3_diff.mini.phase.jsonl \
  --output-jsonl data/associated/real_events.jsonl \
  --starttime 2019-07-06T04:00:00Z \
  --endtime 2019-07-06T05:00:00Z
```

Default output:

```text
data/associated/real_events.jsonl
data/associated/real_events.summary.json
```

The JSONL contains one `real_association_run` metadata record, one
`real_input_pick` record for each phase pick that entered REAL after filtering
and NMS, and one `real_event` record for each associated event. Each event
record contains an origin-time estimate, optional preliminary
latitude/longitude/depth when provided by the associator, association counts,
the parameters used, and the associated arrivals. Each arrival keeps a
`source_pick.real_input_pick_id` that points back to the corresponding
time-window phase record, as well as the original picker JSONL line, station id,
phase name, probability, channel family, and source HDF5 file.

The default settings are conservative smoke-test settings: `--min-prob 0.85`,
0.5 s station-phase NMS, and a small REAL trigger requirement. Lowering
`--min-prob` can recover more candidate arrivals but will increase runtime
quickly during dense aftershock periods. Use `--keep-work-dir` if you want to
inspect the generated REAL `station.dat`, pick files, `catalog_sel.txt`, and
`phase_sel.txt`. Use `--no-include-input-picks` for an event-only JSONL in a
large production run.

For details, see:

```text
scripts/README_real_association.md
```

Compare the associated events with the mini catalog:

```bash
python scripts/compare_associated_events.py \
  --pred-jsonl data/associated/real_events.jsonl \
  --catalog data/label/annotations_mini_two_hours.json \
  --outdir eval_events/real_vs_catalog_mini
```

An event is counted as a true positive when the epicentral distance error is
less than 20 km and the origin-time error is less than 3 s. The generic event
JSONL format accepted by the comparison script is documented in:

```text
scripts/README_event_comparison.md
```

## 12. Notebook Walkthrough

For an interactive version of the workflow, open:

```text
notebooks/seismicx_cont_mini_quickstart.ipynb
```

The notebook covers:

- checking package files and sizes;
- reading the annotation summary;
- querying the SQLite waveform index;
- plotting one HDF5 waveform segment;
- instantiating the PyTorch dataloader;
- preparing picker, evaluation, and consensus commands.

## 13. Rebuild the Mini Package

Most users do not need to rebuild the mini package. Use this section only if you
have the complete SeismicX-Cont annotation file and the source MiniSEED archive.

Re-extract annotations from the full annotation JSON:

```bash
python scripts/extract_mini_annotations.py \
  --source ../data/label/annotations_for_continuous_hdf5.json \
  --output data/label/annotations_mini_two_hours.json
```

Rebuild the two hourly HDF5 files from the source waveform archive:

```bash
SEISMICX_RAW_ROOT=/path/to/continous_usa/data \
  ./scripts/make_mini_hdf5_subsets.sh
```

The script expects source waveform folders such as:

```text
$SEISMICX_RAW_ROOT/07/06
$SEISMICX_RAW_ROOT/11/14
```

Rebuild the SQLite waveform index:

```bash
./scripts/build_mini_waveform_index.sh
```

This script writes the SQLite database to local temporary storage first and then
copies it into `data/index/`, because some external drives are unreliable for
SQLite database creation. You can choose the temporary path explicitly:

```bash
LOCAL_DB=/tmp/seismicx_cont_mini_waveform_index.sqlite \
  ./scripts/build_mini_waveform_index.sh
```

Update checksums:

```bash
./scripts/write_manifest.sh
```

## 14. Troubleshooting

`ModuleNotFoundError: No module named 'utils'`
: Run commands from the `publish_mini` folder, or use the provided scripts rather
than moving files into another directory.

`Permission denied` for shell scripts
: Run `chmod +x scripts/*.sh scripts/*.py`.

SQLite errors on an external drive
: Build the database on local storage with `LOCAL_DB=/tmp/...` and let
`build_mini_waveform_index.sh` copy the finished database back into
`data/index/`.

Picker is slow on CPU
: Start with one model using `run_picker_to_jsonl_mini.sh`. Use `DEVICE=mps` or
`DEVICE=cuda` with `run_pickers_mini.sh` when hardware is available.

Missing source waveform archive during rebuild
: The downloaded mini package already includes HDF5 files. Rebuilding HDF5 is
only possible if you also have the original MiniSEED source archive.

`REAL association is slow`
: Start with the default one-hour command and `--min-prob 0.85`. Lower
probability thresholds can greatly increase the number of input picks and the
REAL search time.

## 15. What to Cite

When using this mini subset in examples or tutorials, cite the complete
SeismicX-Cont release and the original waveform/data sources described in the
paper. The complete dataset is available at:

```text
https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont
```

The mini subset is a convenience package derived from the complete
SeismicX-Cont benchmark release.
