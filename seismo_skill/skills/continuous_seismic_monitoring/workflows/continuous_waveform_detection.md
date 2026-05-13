---
name: continuous_waveform_detection
title: Continuous Waveform Earthquake Detection / 连续波形地震检测
version: "1.0"
description: Detect earthquakes from continuous waveform data by reusing a SeismicX-Cont-style dataset, a phase picker, and phase association; generate catalogs, QC figures, and evidence reports.
keywords:
  - earthquake detection
  - continuous waveform
  - event association
  - REAL
  - catalog
  - 地震检测
  - 连续波形检测
  - 地震目录
  - 震相关联
skills:
  - name: continuous_seismic_monitoring
    role: End-to-end continuous waveform detection, picker execution, association, QC, and report outputs.
  - name: pnsn_phase_detection
    role: Optional phase detection and seismic monitoring concepts.
steps:
  - id: load_dataset
    skill: continuous_seismic_monitoring
    description: Load or build a SeismicX-Cont-style HDF5/SQLite dataset and select the detection time range.
  - id: run_picker
    skill: continuous_seismic_monitoring
    description: Run the selected picker over continuous waveform windows and write phase JSONL.
    depends_on: [load_dataset]
  - id: filter_picks
    skill: continuous_seismic_monitoring
    description: Apply score thresholds, non-maximum suppression, station coverage checks, and duplicate-pick filtering.
    depends_on: [run_picker]
  - id: associate_events
    skill: continuous_seismic_monitoring
    description: Associate filtered picks into candidate earthquakes with REAL or a user-selected associator.
    depends_on: [filter_picks]
  - id: validate_candidates
    skill: continuous_seismic_monitoring
    description: Compare with labels/catalogs when available; otherwise produce internal QC and manual-review evidence.
    depends_on: [associate_events]
  - id: write_detection_report
    skill: continuous_seismic_monitoring
    description: Save catalog, figures, evidence tables, missing information, and reusable outputs for Science Agent and Parameter Optimizer.
    depends_on: [validate_candidates]
---

# Continuous Waveform Earthquake Detection

Use this workflow when the user gives continuous waveforms and asks to detect earthquakes. It assumes the data either already follows the dataset contract from `continuous_dataset_build` or can be converted first.

## Inputs

- HDF5 continuous waveform files and SQLite index.
- Station metadata.
- Picker model/API or picker output JSONL.
- Association method: default REAL, or a user-specified associator.
- Detection configuration: time range, stations, picker threshold, association thresholds, travel-time table, output directory.
- Optional labels/catalogs for recall and validation.

## Procedure

1. If no dataset/index exists, run `continuous_dataset_build` first.
2. Query the selected time range and stations using the SQLite index. Confirm that waveform coverage is non-empty.
3. Run the picker or load existing picker JSONL. Keep raw picker logs and runtime settings.
4. Filter picks:
   - Remove invalid timestamps and unknown stations.
   - Apply score thresholds per phase.
   - Merge duplicate picks within a small time window.
   - Summarize pick counts by station, phase, and time.
5. Associate events. Prefer `publish_mini/scripts/run_real_association.py` for REAL-based association:

```bash
python scripts/run_real_association.py \
  --picks data/picks/user_picker.phase.jsonl \
  --stations data/label/stations.csv \
  --out data/associated/user_picker_real
```

6. Validate candidate events:
   - If labels/catalogs exist, compute event recall and residuals.
   - If labels are absent, report candidate events with internal QC only and mark them as requiring manual review.
7. Generate figures: pick timeline, station contribution, candidate event timeline, spatial map when coordinates exist, depth/magnitude proxies if available, and false-negative/uncertain-event diagnostics when labels exist.
8. Save outputs for later workflows. Parameter Optimizer should be able to reuse threshold settings and objective metrics; Science Agent should be able to reuse catalogs, figures, and evidence tables.

## Required Outputs

- `phase_picks.jsonl` or a named picker output JSONL.
- `associated_events.jsonl` and `associated_events.summary.json`.
- Optional event-comparison metrics when reference events exist.
- QC figures and tables.
- `continuous_detection_report.md`.
- `detection_run_config.json` with all thresholds, data paths, and picker/associator versions.

## Scientific Use

This workflow can feed Science Agent with reproducible evidence: event-rate changes, spatial clustering, station coverage, missed detections, and parameter sensitivity. It can also feed Parameter Optimizer with objective functions such as phase recall, event recall, residual MAD, false associations, and runtime.
