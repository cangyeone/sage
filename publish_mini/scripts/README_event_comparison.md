# Event-Level Catalog Comparison

`scripts/compare_associated_events.py` compares associated earthquake events
against a reference catalog. It is designed for `run_real_association.py`
outputs, but it can also evaluate any event JSONL file that follows the generic
event schema below.

## True-Positive Rule

An associated event is counted as a true-positive event when it can be matched
one-to-one to a reference event with both:

- epicentral distance error `< 20 km`;
- origin-time error `< 3 s`.

The script uses greedy one-to-one matching sorted by a combined normalized
time-and-distance score. This prevents one reference event from being counted as
multiple true positives.

## Generic Event JSONL Format

Any JSONL file can be evaluated if each event record contains the following
fields:

```json
{
  "record_type": "event",
  "event_id": "example_000001",
  "origin_time_iso": "2019-07-06T04:13:06.130Z",
  "latitude": 35.9092,
  "longitude": -117.6893,
  "depth_km": 8.39,
  "magnitude": 3.63
}
```

Required fields:

| Field | Description |
| --- | --- |
| `event_id` | Unique event id. `id` is also accepted. |
| `origin_time_iso` | Origin time. `origin_time`, `event_time`, or `time` are also accepted. |
| `latitude` | Event latitude in degrees. |
| `longitude` | Event longitude in degrees. |

Optional fields:

| Field | Description |
| --- | --- |
| `record_type` | Use `real_event`, `event`, or `catalog_event` for predicted events. |
| `depth_km` | Event depth in kilometers. `depth_m` is also accepted and converted. |
| `magnitude` | Event magnitude. `mag` or `magnitude_median` are also accepted. |

`run_real_association.py` already writes compatible `real_event` records. Its
workflow JSONL may also contain `real_association_run` and `real_input_pick`
records; the comparison script ignores those non-event records.

## SeismicX Annotation Catalog

The default reference format is the SeismicX annotation JSON:

```text
data/label/annotations_for_continuous_hdf5.json
```

The script extracts event id, preferred origin time, latitude, longitude, depth,
and magnitude from the nested annotation structure.

## Mini Example

Run from the `publish_mini/` folder:

```bash
python scripts/compare_associated_events.py \
  --pred-jsonl data/associated/real_events.jsonl \
  --catalog data/label/annotations_mini_two_hours.json \
  --outdir eval_events/real_vs_catalog_mini
```

The predicted-event time window is inferred from the `real_association_run`
record written by `run_real_association.py`. If the input JSONL does not contain
that metadata record, provide the time window explicitly:

```bash
python scripts/compare_associated_events.py \
  --pred-jsonl data/associated/my_events.jsonl \
  --catalog data/label/annotations_for_continuous_hdf5.json \
  --starttime 2019-07-06T04:00:00Z \
  --endtime 2019-07-06T05:00:00Z \
  --outdir eval_events/my_events_vs_catalog
```

## Full-Dataset Example

```bash
python scripts/compare_associated_events.py \
  --pred-jsonl data/associated/real_events.jsonl \
  --catalog data/label/annotations_for_continuous_hdf5.json \
  --outdir eval_events/real_vs_catalog
```

## Generic Event-JSONL Reference Catalog

Use `--catalog-format event-jsonl` if the reference catalog is also a JSONL file
following the generic event schema:

```bash
python scripts/compare_associated_events.py \
  --pred-jsonl data/associated/real_events.jsonl \
  --catalog external_catalog.jsonl \
  --catalog-format event-jsonl \
  --outdir eval_events/real_vs_external_catalog
```

## Output Files

The output directory contains:

| File | Description |
| --- | --- |
| `event_match_summary.json` | Counts, precision/recall/F1, time-window metadata, and error statistics for true-positive events. |
| `event_match_summary.tsv` | Flat metric table for quick inspection. |
| `event_matches.jsonl` | One record per TP, FP, and FN event. |

Output record types in `event_matches.jsonl`:

- `event_true_positive`: predicted event matched to a reference event.
- `event_false_positive`: predicted event with no matching reference event.
- `event_false_negative`: reference event missed by the predicted event set.

Error statistics are reported for origin-time error, epicentral distance error,
depth error when both depths are available, and magnitude error when both
magnitudes are available.
