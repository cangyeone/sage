# publish_mini Code Map

`publish_mini/` contains the reusable SeismicX-Cont mini benchmark code. The `data/` directory is intentionally external and should be downloaded from:

<https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont>

## Dataset Verification

```bash
cd publish_mini
python scripts/verify_mini_subset.py
python scripts/example_query_mini.py
python scripts/example_query_waveform.py
python scripts/example_dataloader.py
```

## Dataset Construction

- `scripts/makeh5_flex_seg.py`: flexible waveform-to-HDF5 segment builder.
- `scripts/make_mini_hdf5_subsets.sh`: example mini subset HDF5 construction wrapper.
- `scripts/build_mini_waveform_index.sh`: builds the SQLite waveform index.
- `utils/hdf5_waveform_index.py`: index builder implementation.
- `utils/hdf5_waveform_dataset.py`: HDF5 dataset/dataloader helpers.
- `utils/waveform_index_api.py`: SQLite query API.

## Picker Inference

- `scripts/run_picker_to_jsonl.py`: generic picker-to-phase-JSONL runner.
- `scripts/run_picker_to_jsonl_mini.sh`: mini benchmark example wrapper.
- `scripts/run_pickers_mini.sh`: runs several configured pickers on the mini data.

The expected phase-pick JSONL contract is one pick per line with station/channel identity, phase name, pick time, score/probability, and source waveform/window information.

## Pick-Level Evaluation

- `scripts/evaluate_picks.py`: compares picker JSONL with labels and writes summary metrics.
- `scripts/run_evaluate_picks_mini.sh`: example wrapper for the mini subset.
- `scripts/build_consensus_picks_json.py`: combines multiple pickers into consensus picks.
- `scripts/run_consensus_picks_mini.sh`: mini subset consensus wrapper.

Typical metrics:

- P/S recall within tolerance.
- Median/MAD residuals.
- Matched/missed pick counts.
- Station/time coverage.

## Association and Event-Level Evaluation

- `scripts/run_real_association.py`: converts picks and stations into REAL input, runs association, and emits event JSONL/summary.
- `scripts/real/real.c`: bundled REAL source. Compile locally when needed.
- `scripts/real/tt_db/tdb.txt`: example travel-time database.
- `scripts/compare_associated_events.py`: compares associated events with a reference event catalog.
- `scripts/README_real_association.md`: REAL usage notes.
- `scripts/README_event_comparison.md`: event comparison notes.

## Output Hygiene

Write generated outputs to a project-specific output directory when running from SAGE. Do not commit:

- `publish_mini/data/`
- `publish_mini/pickers/`
- `publish_mini/eval_picks/`
- `publish_mini/eval_events/`
- generated manifests, figures, and caches
