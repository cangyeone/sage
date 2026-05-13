# Workflow: Benchmark a Given Picker

Use this workflow when the user provides a picker or asks for phase-picking
recall, precision, residuals, and visual annotation checks.

## Inputs

- Continuous HDF5 waveforms.
- Label JSON.
- One or more picker models or a user-provided picker script.
- Optional station metadata.

## Procedure

1. Confirm the dataset contract from `data_schema_manifest.json` or infer it
   from data notes and HDF5 samples.
2. Run the picker on a small sample first.
3. Validate picker JSONL schema and pick counts.
4. Run the full picker job.
5. Match automatic picks to labels by station alias, phase family, and time.
6. Compute:
   - per-phase recall and precision;
   - median residual and MAD;
   - per-station recall;
   - per-hour recall;
   - confidence-threshold sensitivity.
7. Generate figures:
   - residual histograms;
   - recall/precision by phase;
   - recall by station/time;
   - waveform annotation panels with manual and automatic picks.
8. Write a benchmark report explaining what the model is reliable for and where
   it fails.

## Recommended Plotting Command

```bash
python seismo_skill/skills/pnsn_phase_detection/scripts/plot_picks_and_labels.py \
  --project-root . \
  --h5-input "data/hdf5/*.h5" \
  --label-json data/label/annotations_mini_two_hours.json \
  --auto-jsonl data/picks/pnsn.v3.diff.phase.jsonl \
  --outdir outputs/picker_benchmark/annotation_plots \
  --max-panels 16 \
  --window-seconds 180
```

## Outputs

```text
outputs/picker_benchmark/
  picks.jsonl
  picker_metrics.json
  picker_metrics.csv
  residual_histogram.png
  recall_precision_by_phase.png
  annotation_plots/
  benchmark_report.md
```

## Scientific-Analysis Use

When this workflow is used inside Science Agent, the figures and metrics should
feed into the paper as evidence for model validity, not merely as UI logs.

