---
name: continuous_seismic_monitoring
description: Use this skill for SeismicX-Cont/publish_mini continuous waveform dataset building, HDF5/SQLite dataloaders, picker benchmarking, phase-pick recall/statistics, REAL phase association, and continuous earthquake detection workflows.
---

# Continuous Seismic Monitoring / 连续波形地震监测

Use this skill when the task involves continuous waveform archives, SeismicX-Cont data, picker evaluation, phase association, or end-to-end earthquake detection from continuous data.

适用场景：连续波形数据集制作、HDF5/SQLite 数据加载、给定 picker 后统计震相查全率/误差、REAL 震相关联、连续波形地震检测，以及把检测结果交给科学分析或参数优化页面继续写报告/论文。

## Data Policy / 数据策略

The reusable code lives in `publish_mini/`, but the large `data/` directory and picker model weights are external artifacts and must not be committed.

可复现实验数据从 ModelScope 下载：

<https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont>

Recommended local layout:

```text
publish_mini/
  data/                 # downloaded externally, git-ignored
    hdf5/
    index/
    label/
    picks/
    associated/
  scripts/
  utils/
  pickers/              # optional user-provided checkpoints, git-ignored
```

If `data/` or picker weights are missing, guide the user to download or provide them instead of fabricating metrics.

## Code Map / 代码入口

Prefer reusing the scripts in `publish_mini/scripts/` and utilities in `publish_mini/utils/`:

- Dataset checks: `scripts/verify_mini_subset.py`, `scripts/example_query_mini.py`, `scripts/example_query_waveform.py`
- HDF5/index building: `scripts/makeh5_flex_seg.py`, `scripts/make_mini_hdf5_subsets.sh`, `scripts/build_mini_waveform_index.sh`, `utils/hdf5_waveform_index.py`
- Data loading: `utils/hdf5_waveform_dataset.py`, `utils/waveform_index_api.py`
- Picker inference: `scripts/run_picker_to_jsonl.py`, `scripts/run_picker_to_jsonl_mini.sh`, `scripts/run_pickers_mini.sh`
- Pick evaluation: `scripts/evaluate_picks.py`, `scripts/run_evaluate_picks_mini.sh`
- Multi-picker consensus: `scripts/build_consensus_picks_json.py`, `scripts/run_consensus_picks_mini.sh`
- Association and event comparison: `scripts/run_real_association.py`, `scripts/compare_associated_events.py`

See `references/publish_mini_code_map.md` for a compact command reference.

## Workflows / 工作流

This skill exposes three reusable workflows from `workflows/`:

1. `continuous_dataset_build`: turn user-provided continuous waveforms into a SeismicX-Cont-style dataset.
2. `continuous_picker_benchmark`: run a provided picker and compute phase/event recall, residuals, and QC statistics.
3. `continuous_waveform_detection`: detect earthquakes from continuous waveform data using the dataset/index built by the first workflow.

These workflows are discoverable by chat, Science Agent, and Parameter Optimizer through the shared workflow loader.

## Input and Output Contracts / 输入输出约定

Dataset build outputs should include:

- HDF5 waveform files under `data/hdf5/` or a user-selected project data directory.
- SQLite waveform index such as `data/index/waveform_index.sqlite`.
- Station metadata and optional labels, usually `data/label/stations.csv` and `data/label/*.json`.
- `dataset_manifest.json`, `data_schema_manifest.json`, and `io_contracts.md`.
- QC tables/figures showing trace counts, station coverage, time coverage, missing channels, and sample waveform snippets.

Picker benchmark outputs should include:

- Picker output in JSONL, one pick per line.
- Pick-level summary in JSON/TSV, including recall, matched picks, residual median/MAD, phase-specific metrics, and station/time coverage.
- Optional event-level comparison after association.
- QC figures such as residual histograms, pick timelines, station coverage, and false-negative summaries.

Continuous detection outputs should include:

- Phase JSONL, associated event catalog JSONL/CSV, and association summary.
- Timeline, spatial map when coordinates exist, station contribution plots, and confidence/QC tables.
- A Markdown detection report that separates verified evidence, candidate events, and missing information.

## Validation Rules / 验证规则

Always run a small validation before long jobs:

```bash
python scripts/verify_mini_subset.py
python scripts/example_query_mini.py
python scripts/example_query_waveform.py
```

For generated datasets, validate:

- HDF5 files can be opened and contain non-empty waveform arrays.
- SQLite index exists and returns the expected stations/time windows.
- Dataloader can fetch at least one batch/window.
- Pick JSONL lines have parseable timestamps, station IDs, phase labels, and probabilities/scores.
- Evaluation summaries are derived from label files or catalog files, not from text speculation.

## Evidence Discipline / 证据约束

Do not invent recall, precision, event counts, or detection quality. Every number must come from a script output, label file, event catalog, or logged command result. If labels are absent, say that recall cannot be computed and report only internal QC or manual-review candidates.

When used inside Science Agent, persist intermediate files so they can become evidence for reports and papers. When used inside Parameter Optimizer, record parameter settings, objective metrics, and every trial result so the optimization history is reproducible.
