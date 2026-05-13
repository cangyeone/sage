# Workflow: Build a Continuous Waveform Dataset

Use this workflow when the user provides continuous waveform files and wants a
dataset suitable for phase picking, detection, evaluation, or model
optimization.

## Inputs

- Raw waveform files or HDF5 files.
- Optional station metadata.
- Optional manual/curated phase labels.
- Optional data notes describing time windows, components, and sampling rates.

## Procedure

1. Inventory the project directory. Classify files as waveform data, labels,
   station metadata, scripts, or papers.
2. Read data notes first. Infer station-ID conventions, component order,
   sampling rate, and time windows.
3. If raw waveform files are provided, convert them to a continuous HDF5 layout:
   one station/channel group per time window, with `starttime`, `endtime`, and
   `sampling_rate` attributes.
4. If labels exist, normalize them to a JSON structure with station IDs, phase
   names, absolute UTC pick times, score/status, and event IDs.
5. Write:
   - `data_schema_manifest.json`
   - `io_contracts.md`
   - `dataset_manifest.json`
6. Run smoke tests:
   - first waveform sample loads;
   - waveform shape is `[n_samples, 3]`;
   - station/time attributes are present;
   - labels can be parsed;
   - a small annotation plot is generated.
7. Save diagnostic figures:
   - station/time coverage;
   - label counts by phase;
   - waveform amplitude examples;
   - annotation panels.

## Outputs

```text
data/
  hdf5/
  label/
  index/
outputs/
  dataset_manifest.json
  data_schema_manifest.json
  io_contracts.md
  annotation_plots/
```

## Agent Requirements

- Keep all generated files under the current project directory.
- If a large dataset is needed, point the user to
  `https://www.modelscope.cn/datasets/cangyeone/SeismicX-Cont` instead of
  copying data into the repository.
- If fields are uncertain, mark them as `inferred` in the manifest rather than
  presenting them as verified.

