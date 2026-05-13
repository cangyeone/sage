# Workflow: Detect Earthquakes From Continuous Waveforms

Use this workflow when the user gives continuous waveform data and wants a
detected earthquake catalog.

## Inputs

- Continuous HDF5 or waveform directory.
- Picker model or picker script.
- Station metadata with coordinates.
- Optional velocity model and association parameters.

## Procedure

1. Build or verify the continuous dataset.
2. Run phase picking on all selected windows.
3. Apply confidence thresholds and optional non-maximum suppression.
4. Associate picks into candidate events using FastLink, REAL, or GaMMA.
5. Validate events:
   - minimum number of P picks;
   - minimum number of S picks;
   - station azimuthal coverage;
   - residual/time-window sanity checks.
6. Generate figures:
   - pick raster through time;
   - station/event map;
   - event magnitude or pick-count timeline;
   - example annotation panels for accepted and rejected events.
7. Write an event-detection report with assumptions and known limitations.

## Recommended Association Choice

- FastLink: quick monitoring and high-throughput association.
- REAL: classical association/location-style workflows.
- GaMMA: probabilistic association when uncertainty matters.

## Outputs

```text
outputs/continuous_detection/
  picks.jsonl
  associated_events.jsonl
  event_catalog.csv
  event_detection_summary.md
  figures/
```

## Agent Requirements

- Report the picker model, thresholds, association method, and station metadata
  source.
- If no station coordinates are available, stop before claiming event locations.
- Clearly distinguish phase detections from located earthquake events.

