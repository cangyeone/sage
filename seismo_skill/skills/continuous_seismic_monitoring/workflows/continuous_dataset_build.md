---
name: continuous_dataset_build
title: Continuous Waveform Dataset Build / 连续波形数据集制作
version: "1.0"
description: Convert user-provided continuous waveforms, station metadata, and optional labels into a SeismicX-Cont-style HDF5 + SQLite dataset with validation artifacts.
keywords:
  - continuous waveform
  - SeismicX-Cont
  - HDF5
  - SQLite
  - dataloader
  - 连续波形
  - 数据集制作
  - 台站元数据
  - 震相标签
skills:
  - name: continuous_seismic_monitoring
    role: Dataset schema, HDF5/index conversion, dataloader checks, and QC outputs.
steps:
  - id: inspect_inputs
    skill: continuous_seismic_monitoring
    description: Traverse the user project directory, identify waveform files, station metadata, labels, and existing SeismicX-Cont artifacts.
  - id: infer_schema
    skill: continuous_seismic_monitoring
    description: Sample files and infer channel naming, time coverage, station IDs, sample rates, and label formats with LLM-assisted reasoning.
    depends_on: [inspect_inputs]
  - id: normalize_metadata
    skill: continuous_seismic_monitoring
    description: Create or normalize station metadata and write a data schema manifest.
    depends_on: [infer_schema]
  - id: build_hdf5
    skill: continuous_seismic_monitoring
    description: Convert continuous waveform windows to HDF5 using publish_mini scripts or a compatible adapter.
    depends_on: [normalize_metadata]
  - id: build_index
    skill: continuous_seismic_monitoring
    description: Build the SQLite waveform index for station/time-window queries.
    depends_on: [build_hdf5]
  - id: normalize_labels
    skill: continuous_seismic_monitoring
    description: Normalize optional phase/event labels into JSON/CSV contracts; if labels are absent, record that recall cannot be computed.
    depends_on: [build_index]
  - id: validate_dataset
    skill: continuous_seismic_monitoring
    description: Run mini tests for HDF5 readability, SQLite query, dataloader sample, label consistency, and QC figures.
    depends_on: [normalize_labels]
  - id: write_outputs
    skill: continuous_seismic_monitoring
    description: Save dataset manifest, IO contracts, QC report, and reproducible commands for later chat/science/optimization workflows.
    depends_on: [validate_dataset]
---

# Continuous Waveform Dataset Build

Use this workflow when a user provides continuous waveform data and wants it converted into a reusable continuous-data dataset.

如果用户只是想使用公开测试集，先提示从 ModelScope 下载：

<https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont>

The downloaded `data/` directory may be placed under `publish_mini/data/` or inside the user project. It is external data and should not be committed.

## Inputs

- Continuous waveform root: miniSEED, SAC, HDF5, or a mixed archive.
- Station metadata: CSV, StationXML, JSON, or inventory files.
- Optional labels: phase picks, event catalog, or associated events.
- Output root: project-controlled data/output directory.
- Optional segment policy: window length, stride, allowed gaps, channel family, and sample rate.

## Procedure

1. List files and infer roles. Treat waveform files, station files, label files, and data notes separately.
2. Sample a few waveform files with ObsPy or HDF5 readers. Record sample rate, start/end time, channel count, station naming, and gaps.
3. Create `data_schema_manifest.json` and `io_contracts.md` before conversion. These files must state the inferred input and output contracts.
4. Normalize station metadata into a `stations.csv`-style table with station ID, network, station, location, channel family, longitude, latitude, elevation, and optional start/end time.
5. Build HDF5 waveform segments. Prefer `publish_mini/scripts/makeh5_flex_seg.py` when the raw layout matches; otherwise write a small adapter inside the project output directory.
6. Build a SQLite index with `publish_mini/utils/hdf5_waveform_index.py` or `publish_mini/scripts/build_mini_waveform_index.sh`.
7. Normalize phase/event labels when present. If no labels exist, explicitly mark `label_status: absent` in the manifest.
8. Validate with small tests:
   - HDF5 files open and contain non-empty waveform arrays.
   - SQLite index returns at least one station/time query.
   - Dataloader can read one batch/window.
   - Label times fall inside waveform time coverage when labels exist.
9. Write QC artifacts: station coverage map/table, channel completeness, waveform snippet plots, time coverage histogram, and conversion command log.

## Required Outputs

- `data/hdf5/*.h5` or project-equivalent HDF5 files.
- `data/index/waveform_index.sqlite`.
- `data/label/stations.csv` and optional normalized labels.
- `dataset_manifest.json`.
- `data_schema_manifest.json`.
- `io_contracts.md`.
- `dataset_qc_report.md`.
- QC figures/tables.

## Failure Handling

If a format reader fails, do not abandon the workflow. Sample fewer files, inspect headers with command-line tools, write a format note, and retry with a narrower adapter. If labels are unavailable, continue dataset construction but disable recall metrics in downstream workflows.
