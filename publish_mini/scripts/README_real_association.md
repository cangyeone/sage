# REAL Association Workflow

This folder includes a SeismicX-Cont bridge for the REAL association algorithm:

- `scripts/run_real_association.py` converts SeismicX picker JSONL files to REAL
  input files, compiles `scripts/real/real.c`, runs REAL, and converts
  `phase_sel.txt` back to workflow JSONL.
- `scripts/real/real.c` is the REAL source file used by the wrapper.
- `scripts/real/tt_db/tdb.txt` is the travel-time table used by the default
  example.

The output is a workflow product, not an authoritative earthquake catalog. It
is intended to demonstrate that SeismicX-Cont can support the downstream chain
from continuous waveform loading to phase picking, association-based event
detection, and catalog-style JSONL export. Downstream earthquake location is
left to users: the associated-arrival list in each event is the interface to an
external locator.

## Input Pick Format

The wrapper reads one JSON record per line from `run_picker_to_jsonl.py`. It
uses records with:

- `record_type == "phase_pick"`;
- `phase_time`, `phase_name`, and `phase_prob`;
- `station_id` and/or `station_info`;
- station latitude and longitude in `station_info`, with `data/label/stations.csv`
  used as a fallback;
- optional amplitude, channel, and HDF5 provenance fields.

Phase names are normalized before association:

- `Pg`, `Pn`, and `P` become REAL `P`;
- `Sg`, `Sn`, and `S` become REAL `S`.

REAL's station input is keyed by `network.station`, so location codes are
collapsed for the REAL run. The output keeps the original SeismicX
`station_id`, location code, channel family, probability, and HDF5 provenance
inside each associated pick's `source_pick` object.

## Mini Example

Run from the `publish_mini/` folder:

```bash
python scripts/run_real_association.py \
  --picks-jsonl data/picks/pnsn_v3_diff.mini.phase.jsonl \
  --output-jsonl data/associated/real_events.jsonl \
  --starttime 2019-07-06T04:00:00Z \
  --endtime 2019-07-06T05:00:00Z
```

The default mini smoke-test settings use:

- `--min-prob 0.85`;
- station-phase non-maximum suppression over `--nms-window 0.50` s;
- REAL search parameters `--real-s 3/0/3/0/1.0/0.1/1.5`;
- automatic compilation of `scripts/real/real.c` with `cc -O2 -lm`.

On the current mini pick file, this command accepts 298 picks, keeps 188 after
NMS, writes those 188 time-window phase candidates as `real_input_pick` records,
and writes 23 associated events for the 2019 Ridgecrest hour.

## Full-Dataset Example

Run from the full SeismicX-Cont repository:

```bash
python scripts/run_real_association.py \
  --picks-jsonl data/picks/pnsn_v3_diff.phase.jsonl \
  --output-jsonl data/associated/real_events.jsonl \
  --starttime 2019-07-01T00:00:00Z \
  --endtime 2019-07-08T00:00:00Z \
  --timeout-per-day 600
```

For more complete association, lower `--min-prob` or tune REAL's `--real-s`,
`--real-r`, and `--real-g` parameters. Runtime can grow quickly when many
low-probability picks are included, especially during the dense 2019 aftershock
interval.

## Output Files

The wrapper writes:

```text
data/associated/real_events.jsonl
data/associated/real_events.summary.json
```

The JSONL contains three record types:

- `real_association_run`: one metadata record for the run, including time
  window, filters, REAL parameters, and counts.
- `real_input_pick`: one record for each time-window phase pick that entered
  REAL after probability filtering and station-phase NMS. These records preserve
  unassociated picks as well as picks associated to events.
- `real_event`: one record for each event associated by REAL.

A `real_input_pick` record looks like:

```json
{
  "record_type": "real_input_pick",
  "real_input_pick_id": "REAL_PICK_20190706_00000001",
  "associated": true,
  "associated_event_ids": ["REAL_20190706_000018"],
  "station_id": "BK.BRIC.00",
  "phase": "P",
  "phase_name": "Pg",
  "phase_time": "2019-07-06T04:52:30.440Z",
  "phase_prob": 0.8977,
  "channel_family": "HH",
  "source_h5_file": "data/hdf5/continuous_waveform_usa_20190706_04.h5"
}
```

Each `real_event` record keeps the associated arrivals inside `picks`:

```json
{
  "record_type": "real_event",
  "event_id": "REAL_20190706_000001",
  "origin_time_iso": "2019-07-06T04:01:02.728Z",
  "latitude": 36.0488,
  "longitude": -117.6665,
  "depth_km": 4.0,
  "n_p_picks": 3,
  "n_s_picks": 0,
  "n_associated_picks": 3,
  "association_algorithm": "REAL",
  "association_parameters": {
    "min_prob": 0.85,
    "nms_window": 0.5
  },
  "picks": [
    {
      "network": "CI",
      "station": "TOW2",
      "phase": "P",
      "phase_time": "2019-07-06T04:01:08.257Z",
      "travel_time_sec": 5.5303,
      "residual_sec": 0.0098,
      "source_pick": {
        "real_input_pick_id": "REAL_PICK_20190706_00000115",
        "line_no": 67570,
        "station_id": "CI.TOW2.--",
        "phase_name": "Pg",
        "phase_prob": 0.9157,
        "channel_family": "HN",
        "source_h5_file": "data/hdf5/continuous_waveform_usa_20190706_04.h5"
      }
    }
  ]
}
```

`real_events.summary.json` records the input file, filter counts, REAL command
lines, event count, associated-pick count, and how many time-window input picks
were or were not associated.

## Association JSONL Schema for Downstream Use

The schema is intentionally centered on association rather than on a prescribed
location method. A downstream locator only needs the `real_event` records and
their `picks` arrays.

Minimum event record:

```json
{
  "record_type": "real_event",
  "event_id": "REAL_20190706_000001",
  "origin_time_iso": "2019-07-06T04:01:02.728Z",
  "association_algorithm": "REAL",
  "association_parameters": {"min_prob": 0.85},
  "picks": [
    {
      "network": "CI",
      "station": "TOW2",
      "phase": "P",
      "phase_time": "2019-07-06T04:01:08.257Z",
      "source_pick": {
        "real_input_pick_id": "REAL_PICK_20190706_00000115",
        "station_id": "CI.TOW2.--",
        "phase_name": "Pg",
        "phase_prob": 0.9157,
        "source_h5_file": "data/hdf5/continuous_waveform_usa_20190706_04.h5"
      }
    }
  ]
}
```

Required event fields for association exchange:

| Field | Meaning |
| --- | --- |
| `record_type` | Use `real_event` for events written by this wrapper. Other associators can use `event` or `catalog_event` if they follow the same structure. |
| `event_id` | Unique event id within the file. |
| `origin_time_iso` | Event origin-time estimate from the associator. |
| `picks` | Associated station-phase-time arrivals. |

Required fields for each associated arrival:

| Field | Meaning |
| --- | --- |
| `network`, `station` | Station identity used by the associator. |
| `phase` | Normalized phase used by the associator, usually `P` or `S`. |
| `phase_time` | Arrival time in ISO format. |
| `source_pick.station_id` | Original SeismicX station id, including location code. |

Optional but recommended arrival fields:

| Field | Meaning |
| --- | --- |
| `source_pick.real_input_pick_id` | Link to the corresponding `real_input_pick` record in the same JSONL file. |
| `source_pick.phase_name` | Original picker phase name, for example `Pg`, `Pn`, `Sg`, or `Sn`. |
| `source_pick.phase_prob` | Picker confidence. |
| `source_pick.channel_family`, `source_pick.channels` | Input channel family and channel list. |
| `source_pick.source_h5_file` | HDF5 waveform file used by the picker. |

Optional event fields such as `latitude`, `longitude`, `depth_km`,
`magnitude_median`, residual summaries, or azimuthal gap are kept when the
associator provides them. They are useful diagnostics but are not required for a
user-supplied location workflow.

## Debugging and Intermediate Files

By default, the temporary REAL work directory is removed after a successful run.
Use `--keep-work-dir` to inspect the generated REAL files:

```bash
python scripts/run_real_association.py \
  --picks-jsonl data/picks/pnsn_v3_diff.mini.phase.jsonl \
  --output-jsonl data/associated/real_events.jsonl \
  --starttime 2019-07-06T04:00:00Z \
  --endtime 2019-07-06T05:00:00Z \
  --keep-work-dir
```

The retained work directory contains `station.dat`, the per-day REAL pick
folders, compiled `real` binary, `catalog_sel.txt`, `phase_sel.txt`, stdout, and
stderr.

## Notes

- REAL magnitudes and preliminary hypocentre fields in this workflow are
  diagnostic values from the associator. Treat them as optional metadata unless
  a calibrated location or magnitude workflow is added.
- The default travel-time table is provided for a quick reusable example.
  Regional studies should verify or replace it for their target velocity model.
- The association output is useful for end-to-end workflow testing and candidate
  auditing. It should not replace the primary SeismicX-Cont reference labels.
- Use `--no-include-input-picks` if you need an event-only JSONL for a very large
  production run.
