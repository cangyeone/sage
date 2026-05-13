---
name: continuous_picker_benchmark
title: Continuous Picker Benchmark / 连续波形震相拾取评估
version: "1.0"
description: Run a user-provided phase picker on a SeismicX-Cont-style continuous dataset and produce phase recall, residual, station coverage, and optional event-level association metrics.
keywords:
  - phase picker
  - picker benchmark
  - pick recall
  - precision
  - residual
  - REAL association
  - 震相拾取
  - 查全率
  - 震相关联
skills:
  - name: continuous_seismic_monitoring
    role: Picker JSONL contract, evaluation scripts, recall statistics, and association metrics.
  - name: pnsn_phase_detection
    role: Optional domain reference for P/S phase picking and monitoring-style evaluation.
steps:
  - id: verify_dataset
    skill: continuous_seismic_monitoring
    description: Confirm HDF5, SQLite index, station metadata, and labels are available.
  - id: wrap_picker
    skill: continuous_seismic_monitoring
    description: Adapt the user-provided picker/checkpoint/API into the standard phase-pick JSONL contract.
    depends_on: [verify_dataset]
  - id: run_picker
    skill: continuous_seismic_monitoring
    description: Run the picker over selected waveform windows, with resume support and progress logs.
    depends_on: [wrap_picker]
  - id: evaluate_picks
    skill: continuous_seismic_monitoring
    description: Compare predicted picks with labels and compute phase recall, residuals, station coverage, and missed-pick diagnostics.
    depends_on: [run_picker]
  - id: associate_events
    skill: continuous_seismic_monitoring
    description: Optionally associate picks with REAL and compare event recall when reference events exist.
    depends_on: [evaluate_picks]
  - id: write_benchmark_report
    skill: continuous_seismic_monitoring
    description: Save metrics, figures, failure cases, and reproducible commands for science analysis or parameter optimization.
    depends_on: [associate_events]
---

# Continuous Picker Benchmark

Use this workflow when the user provides a picker and wants objective statistics such as P/S recall, timing residuals, station coverage, and event-level recall.

## Inputs

- HDF5 waveform files and SQLite index produced by `continuous_dataset_build` or downloaded from SeismicX-Cont.
- Station metadata, usually `data/label/stations.csv`.
- Ground-truth phase labels and optional event catalog.
- Picker implementation: TorchScript, ONNX, Python callable, API endpoint, or external command.
- Evaluation settings: phase tolerance, residual window, score threshold, device, time range, and station subset.

## Picker Output Contract

Write one JSON object per line. Required fields: `station_id`, `phase_name`,
`phase_time`, `phase_score`, `source_hdf5`, `window_start`, and
`picker_name`. For example, one line may describe a P pick at
`2019-07-06T04:12:34.567Z` from `NET.STA.LOC.CHA` with a score of `0.98`.

If the picker emits another format, write an adapter in the project output directory and keep the raw output for debugging.

## Procedure

1. Verify dataset integrity with `scripts/verify_mini_subset.py` or equivalent HDF5/index checks.
2. Inspect picker inputs/outputs and make a tiny one-window smoke test before running the full benchmark.
3. Run picker inference. Prefer `publish_mini/scripts/run_picker_to_jsonl.py` when compatible; otherwise call the picker through a small adapter.
4. Evaluate phase picks with `publish_mini/scripts/evaluate_picks.py`. Recommended defaults:

```bash
python scripts/evaluate_picks.py \
  --pred data/picks/user_picker.phase.jsonl \
  --label data/label/annotations_mini_two_hours.json \
  --stations data/label/stations.csv \
  --tp-tol 1.5 \
  --err-window 5.0 \
  --plot \
  --out eval_picks/user_picker
```

5. If multiple pickers are provided, build consensus picks with `scripts/build_consensus_picks_json.py` and evaluate both single-picker and consensus outputs.
6. Optionally run association with `scripts/run_real_association.py`, then compare associated events with `scripts/compare_associated_events.py` when a reference catalog exists.
7. Produce figures: phase residual histogram, recall by phase/station/time, missed-pick distribution, pick density timeline, and optional event-recall map/timeline.

## Required Outputs

- `*.phase.jsonl` picker output.
- `summary.json` and `summary.tsv` pick metrics.
- `matches.jsonl` or equivalent match table.
- Residual and recall figures.
- Optional `real_events.jsonl`, `real_events.summary.json`, and event comparison tables.
- `picker_benchmark_report.md` with verified metrics and failure cases.

## Evidence Rules

Only report recall when ground-truth labels exist. If the picker fails on some stations or windows, include those windows in the missed/failed case table. Do not treat associated events as ground truth unless a reference catalog is explicitly available.
