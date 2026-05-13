# Annotation Plotting Reference

Use this reference whenever the user asks to visualize phase picks, show
waveform annotations, compare automatic and manual picks, or make figures for a
science-analysis report.

## Skill-Local Plotting Script

Use the script stored inside this skill:

```bash
python seismo_skill/skills/pnsn_phase_detection/scripts/plot_picks_and_labels.py \
  --project-root /path/to/project \
  --h5-input "data/hdf5/*.h5" \
  --label-json data/label/annotations_mini_two_hours.json \
  --auto-jsonl data/picks/pnsn.v3.diff.phase.jsonl \
  --outdir "$SAGE_OUTDIR/annotation_plots" \
  --max-panels 12 \
  --window-seconds 180 \
  --min-confidence 0.3
```

The script works without `publish_mini`. If a project contains
`utils/hdf5_waveform_dataset.py`, it uses that dataloader. Otherwise it falls
back to generic HDF5 traversal.

## Visual Convention

- Manual/curated labels: dashed vertical lines.
- Automatic picks: solid vertical lines.
- P/Pg/Pn phases: red colors.
- S/Sg/Sn phases: blue colors.
- Each output panel is saved as PNG.
- `annotation_plot_manifest.json` records the generated figures and source
  waveform files.

## Expected Outputs

```text
annotation_plots/
  annotation_panel_001_CI_CLC_--.png
  annotation_panel_002_BK_BDM_00.png
  annotation_plot_manifest.json
```

The manifest is important for web rendering and later paper drafting.

## Troubleshooting

- **No figures:** check that the HDF5 pattern matches files and that label/auto
  station IDs overlap the waveform station IDs.
- **Only label lines or only auto lines:** the missing side is acceptable, but
  report it clearly.
- **Empty windows:** increase `--window-seconds` or lower
  `--min-confidence`.
- **Station ID mismatch:** normalize aliases such as `NET.STA.00`,
  `NET.STA.--`, and `NET.STA`.
- **Large memory use:** use a smaller `--max-panels`; do not load all HDF5 files
  into memory at once.

## Paper-Ready Figure Guidance

For reports and manuscripts, do not only plot quality-control histograms. Use
annotation plots to support claims such as:

- whether P and S arrivals are clear enough for a proposed picker;
- whether false positives cluster near noisy intervals;
- whether a model misses small emergent phases;
- whether confidence thresholds trade recall for precision.

Always connect annotation examples to a quantitative table, such as per-phase
recall/precision and residual statistics.

