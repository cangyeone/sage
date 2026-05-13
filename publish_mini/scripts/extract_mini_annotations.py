#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the two-hour SeismicX-Cont mini annotation subset."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_WINDOWS = [
    ("2019_dense", "2019-07-06T04:00:00Z", "2019-07-06T05:00:00Z"),
    ("2021_quiet", "2021-11-14T16:00:00Z", "2021-11-14T17:00:00Z"),
]


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def in_windows(dt: datetime, windows):
    for name, start, end in windows:
        if start <= dt < end:
            return name
    return None


def prune_station(station_obj, kept_picks):
    out = copy.deepcopy(station_obj)
    out["picks"] = kept_picks
    out["counts"] = {
        "picks": len(kept_picks),
        "phase": dict(Counter(p.get("phase", "unknown") for p in kept_picks)),
        "status": dict(Counter(p.get("status", "unknown") for p in kept_picks)),
    }
    return out


def extract_subset(source: Path, output: Path, windows):
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out = {
        "format": data.get("format", "seismicx_cont_annotations"),
        "subset_name": "SeismicX-Cont mini two-hour subset",
        "subset_description": (
            "Two one-hour annotation subsets paired with hourly HDF5 waveform "
            "files: the densest labeled 2019 hour and the least labeled nonzero "
            "2021 hour."
        ),
        "source_annotation_file": str(source),
        "hdf5_hierarchy": data.get("hdf5_hierarchy"),
        "station_id_format": data.get("station_id_format"),
        "empty_location_value": data.get("empty_location_value"),
        "subset_windows": [
            {"name": name, "starttime": start.isoformat().replace("+00:00", "Z"),
             "endtime": end.isoformat().replace("+00:00", "Z")}
            for name, start, end in windows
        ],
        "years": {},
        "summary": {},
    }

    global_phase = Counter()
    global_status = Counter()
    global_window = Counter()
    global_counts = Counter()
    by_window_phase = defaultdict(Counter)
    by_window_status = defaultdict(Counter)

    for year_id, year_obj in (data.get("years") or {}).items():
        out_year = {"utc_time": year_obj.get("utc_time", year_id), "days": {}}

        for day_id, day_obj in (year_obj.get("days") or {}).items():
            out_day = {"utc_time": day_obj.get("utc_time", day_id), "events": {}}

            for event_id, event_obj in (day_obj.get("events") or {}).items():
                out_event = copy.deepcopy(event_obj)
                out_event["stations"] = {}

                for station_id, station_obj in (event_obj.get("stations") or {}).items():
                    kept = []
                    for pick in (station_obj.get("picks") or []):
                        pick_time = pick.get("time")
                        if not pick_time:
                            continue
                        window_name = in_windows(parse_utc(pick_time), windows)
                        if window_name is None:
                            continue
                        kept_pick = copy.deepcopy(pick)
                        kept_pick["mini_subset_window"] = window_name
                        kept.append(kept_pick)

                        phase = kept_pick.get("phase", "unknown")
                        status = kept_pick.get("status", "unknown")
                        global_counts["picks"] += 1
                        global_phase[phase] += 1
                        global_status[status] += 1
                        global_window[window_name] += 1
                        by_window_phase[window_name][phase] += 1
                        by_window_status[window_name][status] += 1

                    if kept:
                        out_event["stations"][station_id] = prune_station(station_obj, kept)

                if out_event["stations"]:
                    out_event["counts"] = {
                        "stations": len(out_event["stations"]),
                        "picks": sum(len(s.get("picks") or []) for s in out_event["stations"].values()),
                    }
                    out_day["events"][event_id] = out_event

            if out_day["events"]:
                out_year["days"][day_id] = out_day

        if out_year["days"]:
            out["years"][year_id] = out_year

    out["summary"] = {
        "n_windows": len(windows),
        "n_years": len(out["years"]),
        "n_days": sum(len(y["days"]) for y in out["years"].values()),
        "n_events": sum(len(d["events"]) for y in out["years"].values() for d in y["days"].values()),
        "n_stations_with_picks": sum(
            len(e["stations"])
            for y in out["years"].values()
            for d in y["days"].values()
            for e in d["events"].values()
        ),
        "n_picks": global_counts["picks"],
        "phase_counts": dict(global_phase),
        "status_counts": dict(global_status),
        "window_counts": dict(global_window),
        "window_phase_counts": {k: dict(v) for k, v in by_window_phase.items()},
        "window_status_counts": {k: dict(v) for k, v in by_window_status.items()},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("../data/label/annotations_for_continuous_hdf5.json"),
        help="Full SeismicX-Cont annotation JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/label/annotations_mini_two_hours.json"),
        help="Output mini annotation JSON.",
    )
    parser.add_argument(
        "--window",
        action="append",
        nargs=3,
        metavar=("NAME", "START", "END"),
        help="Optional custom window. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    raw_windows = args.window or DEFAULT_WINDOWS
    windows = [(name, parse_utc(start), parse_utc(end)) for name, start, end in raw_windows]
    extract_subset(args.source, args.output, windows)


if __name__ == "__main__":
    main()
