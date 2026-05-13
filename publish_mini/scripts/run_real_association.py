#!/usr/bin/env python3
"""Run REAL association on SeismicX-Cont picker JSONL output.

The script bridges the SeismicX-Cont JSONL pick format to the REAL C program:

1. read one-pick-per-line JSONL records from ``run_picker_to_jsonl.py``;
2. normalize Pg/Pn to P and Sg/Sn to S;
3. write REAL station and day/station/phase pick files;
4. compile and run ``scripts/real/real.c``;
5. convert REAL ``phase_sel.txt`` outputs to workflow JSONL records.

The output JSONL includes run metadata, the time-window phase picks that were
sent to REAL, and the associated REAL events. It is intended as a reusable
workflow product, not as an authoritative catalog.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def parse_iso_time(value: str) -> dt.datetime:
    value = str(value).replace("Z", "+00:00")
    out = dt.datetime.fromisoformat(value)
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out


def iso_z(value: dt.datetime) -> str:
    return value.replace(tzinfo=dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def day_key(value: dt.datetime) -> str:
    return value.strftime("%Y%m%d")


def normalize_phase(phase_name: str) -> str | None:
    phase = str(phase_name or "").upper()
    if phase.startswith("P"):
        return "P"
    if phase.startswith("S"):
        return "S"
    return None


def load_station_csv(path: Path | None) -> dict[tuple[str, str], dict]:
    if path is None or not path.exists():
        return {}

    stations = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            net = (row.get("net") or row.get("network") or "").strip()
            sta = (row.get("sta") or row.get("station") or "").strip()
            if not net or not sta:
                continue
            stations[(net, sta)] = {
                "network": net,
                "station": sta,
                "latitude": float(row.get("lat") or row.get("latitude")),
                "longitude": float(row.get("lon") or row.get("longitude")),
                "elevation_m": float(row.get("elev_m") or row.get("elevation") or 0.0),
            }
    return stations


def load_picks(args) -> tuple[list[dict], dict[tuple[str, str], dict], dict]:
    station_csv = load_station_csv(args.station_csv)
    starttime = parse_iso_time(args.starttime) if args.starttime else None
    endtime = parse_iso_time(args.endtime) if args.endtime else None

    picks = []
    stations = {}
    stats = Counter()

    with args.picks_jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                stats["bad_json"] += 1
                continue
            if record.get("record_type") != "phase_pick":
                stats[f"skip_record_type:{record.get('record_type', '')}"] += 1
                continue

            prob = float(record.get("phase_prob") or 0.0)
            if prob < args.min_prob:
                stats["skip_low_probability"] += 1
                continue

            phase = normalize_phase(record.get("phase_name", ""))
            if phase is None:
                stats["skip_unknown_phase"] += 1
                continue

            try:
                pick_time = parse_iso_time(record["phase_time"])
            except Exception:
                stats["skip_bad_time"] += 1
                continue

            if starttime is not None and pick_time < starttime:
                stats["skip_before_starttime"] += 1
                continue
            if endtime is not None and pick_time >= endtime:
                stats["skip_after_endtime"] += 1
                continue

            station_id = str(record.get("station_id") or "")
            station_info = record.get("station_info") or {}
            parts = station_id.split(".")
            network = str(station_info.get("network") or (parts[0] if len(parts) > 0 else ""))
            station = str(station_info.get("station") or (parts[1] if len(parts) > 1 else ""))
            location = str(station_info.get("location") or (parts[2] if len(parts) > 2 else "--"))
            if not network or not station:
                stats["skip_missing_station_id"] += 1
                continue

            fallback_station = station_csv.get((network, station), {})
            latitude = station_info.get("latitude", fallback_station.get("latitude"))
            longitude = station_info.get("longitude", fallback_station.get("longitude"))
            elevation_m = station_info.get("elevation", fallback_station.get("elevation_m", 0.0))
            if latitude is None or longitude is None:
                stats["skip_missing_station_location"] += 1
                continue

            base = dt.datetime(pick_time.year, pick_time.month, pick_time.day)
            sec_of_day = (pick_time - base).total_seconds()
            amplitude = abs(float(record.get("amplitude") or 0.0))
            if amplitude <= 0:
                amplitude = 1.0

            item = {
                "line_no": line_no,
                "network": network,
                "station": station,
                "location": location,
                "station_id": station_id or f"{network}.{station}.{location}",
                "phase": phase,
                "phase_name": record.get("phase_name", ""),
                "phase_time": pick_time,
                "phase_time_iso": iso_z(pick_time),
                "sec_of_day": sec_of_day,
                "probability": prob,
                "amplitude": amplitude,
                "day": day_key(pick_time),
                "channel_family": record.get("channel_family", ""),
                "channels": record.get("channels", []),
                "source_h5_file": record.get("h5_file", ""),
            }
            picks.append(item)

            skey = (network, station)
            stations.setdefault(
                skey,
                {
                    "network": network,
                    "station": station,
                    "latitude": float(latitude),
                    "longitude": float(longitude),
                    "elevation_m": float(elevation_m or 0.0),
                    "locations": set(),
                },
            )
            stations[skey]["locations"].add(location)
            stats["accepted"] += 1
            stats[f"accepted_{phase}"] += 1
            stats[f"accepted_day_{item['day']}"] += 1

    return picks, stations, dict(stats)


def apply_nms(picks: list[dict], nms_window: float) -> list[dict]:
    if nms_window <= 0:
        return sorted(picks, key=lambda x: (x["day"], x["network"], x["station"], x["phase"], x["sec_of_day"]))

    grouped = defaultdict(list)
    for pick in picks:
        key = (pick["day"], pick["network"], pick["station"], pick["phase"])
        grouped[key].append(pick)

    kept = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda x: x["sec_of_day"])
        cluster = []
        cluster_start = None
        for pick in group:
            if cluster_start is None:
                cluster = [pick]
                cluster_start = pick["sec_of_day"]
                continue
            if pick["sec_of_day"] - cluster_start <= nms_window:
                cluster.append(pick)
                continue
            kept.append(max(cluster, key=lambda x: x["probability"]))
            cluster = [pick]
            cluster_start = pick["sec_of_day"]
        if cluster:
            kept.append(max(cluster, key=lambda x: x["probability"]))

    return sorted(kept, key=lambda x: (x["day"], x["network"], x["station"], x["phase"], x["sec_of_day"]))


def assign_input_pick_ids(picks: list[dict]) -> None:
    for idx, pick in enumerate(picks, 1):
        pick["real_input_pick_id"] = f"REAL_PICK_{pick['day']}_{idx:08d}"


def write_real_inputs(picks: list[dict], stations: dict, work_dir: Path) -> tuple[Path, Path, dict]:
    pick_dir = work_dir / "real_picks"
    station_file = work_dir / "station.dat"
    pick_dir.mkdir(parents=True, exist_ok=True)

    day_counts = Counter()
    for pick in picks:
        day_path = pick_dir / pick["day"]
        day_path.mkdir(parents=True, exist_ok=True)
        out = day_path / f"{pick['network']}.{pick['station']}.{pick['phase']}.txt"
        with out.open("a", encoding="utf-8") as f:
            f.write(
                f"{pick['sec_of_day']:.3f} {pick['probability']:.3f} "
                f"{pick['amplitude']:.6e}\n"
            )
        day_counts[(pick["day"], pick["phase"])] += 1

    with station_file.open("w", encoding="utf-8") as f:
        for key in sorted(stations):
            sta = stations[key]
            elev_km = float(sta.get("elevation_m") or 0.0) / 1000.0
            f.write(
                f"{sta['longitude']:.4f} {sta['latitude']:.4f} "
                f"{sta['network']} {sta['station']} BHZ {elev_km:.4f}\n"
            )

    return pick_dir, station_file, {"day_phase_counts": dict(day_counts)}


def compile_real(args, work_dir: Path) -> Path:
    if args.real_bin:
        real_bin = resolve_path(args.real_bin)
        if not real_bin.exists():
            raise FileNotFoundError(real_bin)
        return real_bin

    real_src = args.real_src
    if not real_src.exists():
        raise FileNotFoundError(real_src)

    out = work_dir / "bin" / "real"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [args.cc, "-O2", str(real_src), "-lm", "-o", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            "REAL compilation failed\n"
            f"command: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return out


def median_latitude(stations: dict) -> float:
    values = [float(item["latitude"]) for item in stations.values()]
    if not values:
        return 0.0
    return float(statistics.median(values))


def run_real_for_days(args, real_bin: Path, pick_dir: Path, station_file: Path, stations: dict, work_dir: Path) -> dict:
    run_root = work_dir / "real_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    outputs = {}
    days = sorted(p.name for p in pick_dir.iterdir() if p.is_dir())
    lat_center = args.lat_center if args.lat_center is not None else median_latitude(stations)

    for day in days:
        year = day[:4]
        month = day[4:6]
        dd = day[6:8]
        run_dir = run_root / day
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(real_bin),
            f"-D{year}/{month}/{dd}/{lat_center:.4f}",
            f"-R{args.real_r}",
            f"-G{args.real_g}",
            f"-S{args.real_s}",
            f"-V{args.real_v}",
            str(station_file),
            str(pick_dir / day),
            str(args.ttdb),
        ]
        proc = subprocess.run(
            cmd,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=args.timeout_per_day,
            check=False,
        )
        (run_dir / "real.stdout").write_text(proc.stdout or "", encoding="utf-8")
        (run_dir / "real.stderr").write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode != 0:
            raise RuntimeError(
                f"REAL failed for {day} with exit code {proc.returncode}. "
                f"See {run_dir / 'real.stderr'}"
            )

        outputs[day] = {
            "run_dir": str(run_dir),
            "catalog_sel": str(run_dir / "catalog_sel.txt"),
            "phase_sel": str(run_dir / "phase_sel.txt"),
            "command": cmd,
        }

    return outputs


def build_pick_lookup(picks: list[dict]):
    lookup = defaultdict(list)
    for pick in picks:
        lookup[(pick["day"], pick["network"], pick["station"], pick["phase"])].append(pick)
    for key in lookup:
        lookup[key].sort(key=lambda x: x["sec_of_day"])
    return lookup


def nearest_pick(lookup, day: str, network: str, station: str, phase: str, sec: float, tolerance: float):
    group = lookup.get((day, network, station, phase), [])
    if not group:
        return None
    values = [item["sec_of_day"] for item in group]
    idx = bisect.bisect_left(values, sec)
    candidates = []
    if idx < len(group):
        candidates.append(group[idx])
    if idx > 0:
        candidates.append(group[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda x: abs(x["sec_of_day"] - sec))
    if abs(best["sec_of_day"] - sec) <= tolerance:
        return best
    return None


def parse_event_header(fields: list[str]) -> dict:
    if len(fields) < 17:
        raise ValueError(f"Unexpected REAL event header: {' '.join(fields)}")
    event_num = int(fields[0])
    year, month, day = int(fields[1]), int(fields[2]), int(fields[3])
    time_token = fields[4]
    origin_time = parse_iso_time(f"{year:04d}-{month:02d}-{day:02d}T{time_token}Z")
    offset = 5
    return {
        "real_event_number": event_num,
        "origin_time": origin_time,
        "origin_time_iso": iso_z(origin_time),
        "origin_offset_sec": float(fields[offset]),
        "std_sec": float(fields[offset + 1]),
        "latitude": float(fields[offset + 2]),
        "longitude": float(fields[offset + 3]),
        "depth_km": float(fields[offset + 4]),
        "magnitude_median": float(fields[offset + 5]),
        "magnitude_std": float(fields[offset + 6]),
        "n_p_picks": int(fields[offset + 7]),
        "n_s_picks": int(fields[offset + 8]),
        "n_picks": int(fields[offset + 9]),
        "n_stations_with_ps": int(fields[offset + 10]),
        "azimuthal_gap_deg": float(fields[offset + 11]),
        "picks": [],
    }


def parse_real_outputs(outputs: dict, picks: list[dict], args) -> list[dict]:
    lookup = build_pick_lookup(picks)
    events = []

    for day in sorted(outputs):
        phase_sel = Path(outputs[day]["phase_sel"])
        if not phase_sel.exists():
            continue

        current = None
        with phase_sel.open("r", encoding="utf-8") as f:
            for raw in f:
                fields = raw.split()
                if not fields:
                    continue
                if fields[0].isdigit():
                    if current is not None:
                        events.append(current)
                    current = parse_event_header(fields)
                    current["day"] = day
                    continue

                if current is None or len(fields) < 9:
                    continue
                network, station, phase = fields[0], fields[1], fields[2]
                sec = float(fields[3])
                pick_time = dt.datetime.strptime(day, "%Y%m%d") + dt.timedelta(seconds=sec)
                original = nearest_pick(
                    lookup,
                    day=day,
                    network=network,
                    station=station,
                    phase=phase,
                    sec=sec,
                    tolerance=args.match_tolerance,
                )
                arrival = {
                    "network": network,
                    "station": station,
                    "phase": phase,
                    "phase_time": iso_z(pick_time),
                    "phase_offset_sec": sec,
                    "travel_time_sec": float(fields[4]),
                    "amplitude": float(fields[5]),
                    "residual_sec": float(fields[6]),
                    "weight": float(fields[7]),
                    "back_azimuth_deg": float(fields[8]),
                }
                if original is not None:
                    arrival["source_pick"] = {
                        "real_input_pick_id": original.get("real_input_pick_id"),
                        "line_no": original["line_no"],
                        "station_id": original["station_id"],
                        "location": original["location"],
                        "phase_name": original["phase_name"],
                        "phase_prob": original["probability"],
                        "phase_time": original["phase_time_iso"],
                        "channel_family": original["channel_family"],
                        "channels": original["channels"],
                        "source_h5_file": original["source_h5_file"],
                    }
                current["picks"].append(arrival)

        if current is not None:
            events.append(current)

    for idx, event in enumerate(events, 1):
        event["record_type"] = "real_event"
        event["event_id"] = f"REAL_{event['day']}_{idx:06d}"
        event["association_algorithm"] = "REAL"
        event["association_parameters"] = {
            "min_prob": args.min_prob,
            "nms_window": args.nms_window,
            "real_r": args.real_r,
            "real_g": args.real_g,
            "real_s": args.real_s,
            "real_v": args.real_v,
        }
        event["n_associated_picks"] = len(event["picks"])
        event.pop("origin_time", None)

    return events


def compact_input_pick(pick: dict, associated_event_ids: list[str] | None = None) -> dict:
    associated_event_ids = associated_event_ids or []
    return {
        "record_type": "real_input_pick",
        "real_input_pick_id": pick["real_input_pick_id"],
        "associated": bool(associated_event_ids),
        "associated_event_ids": associated_event_ids,
        "line_no": pick["line_no"],
        "day": pick["day"],
        "network": pick["network"],
        "station": pick["station"],
        "location": pick["location"],
        "station_id": pick["station_id"],
        "phase": pick["phase"],
        "phase_name": pick["phase_name"],
        "phase_time": pick["phase_time_iso"],
        "phase_offset_sec": pick["sec_of_day"],
        "phase_prob": pick["probability"],
        "amplitude": pick["amplitude"],
        "channel_family": pick["channel_family"],
        "channels": pick["channels"],
        "source_h5_file": pick["source_h5_file"],
    }


def build_associated_input_pick_map(events: list[dict]) -> dict[str, list[str]]:
    associated = defaultdict(list)
    for event in events:
        event_id = event.get("event_id")
        if not event_id:
            continue
        for arrival in event.get("picks", []):
            pick_id = (arrival.get("source_pick") or {}).get("real_input_pick_id")
            if pick_id:
                associated[pick_id].append(event_id)
    return dict(associated)


def build_run_record(
    args,
    raw_stats: dict,
    picks_before_nms: int,
    picks_after_nms: int,
    stations: dict,
    events: list[dict],
) -> dict:
    associated_input_pick_count = len(build_associated_input_pick_map(events))
    return {
        "record_type": "real_association_run",
        "format": "seismicx_real_association_jsonl_v1",
        "picks_jsonl": str(args.picks_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "time_window": {
            "starttime": args.starttime,
            "endtime": args.endtime,
        },
        "filters": {
            "min_prob": args.min_prob,
            "nms_window": args.nms_window,
            "match_tolerance": args.match_tolerance,
        },
        "real_parameters": {
            "real_r": args.real_r,
            "real_g": args.real_g,
            "real_s": args.real_s,
            "real_v": args.real_v,
        },
        "raw_pick_stats": raw_stats,
        "picks_before_nms": picks_before_nms,
        "real_input_pick_count": picks_after_nms,
        "station_count_net_sta": len(stations),
        "event_count": len(events),
        "associated_pick_count": sum(len(event.get("picks", [])) for event in events),
        "unique_associated_input_pick_count": associated_input_pick_count,
        "unassociated_input_pick_count": picks_after_nms - associated_input_pick_count,
        "includes_real_input_picks": bool(args.include_input_picks),
    }


def write_jsonl(
    events: list[dict],
    input_picks: list[dict],
    output_jsonl: Path,
    args,
    raw_stats: dict,
    picks_before_nms: int,
    stations: dict,
):
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    associated_map = build_associated_input_pick_map(events)
    with output_jsonl.open("w", encoding="utf-8") as f:
        run_record = build_run_record(
            args=args,
            raw_stats=raw_stats,
            picks_before_nms=picks_before_nms,
            picks_after_nms=len(input_picks),
            stations=stations,
            events=events,
        )
        f.write(json.dumps(run_record, ensure_ascii=False, separators=(",", ":")) + "\n")
        if args.include_input_picks:
            for pick in input_picks:
                record = compact_input_pick(
                    pick,
                    associated_event_ids=associated_map.get(pick["real_input_pick_id"], []),
                )
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_summary(path: Path, args, raw_stats: dict, picks_before_nms: int, picks_after_nms: int, stations: dict, outputs: dict, events: list[dict]):
    associated_input_pick_count = len(build_associated_input_pick_map(events))
    summary = {
        "picks_jsonl": str(args.picks_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "work_dir": str(args.work_dir),
        "work_dir_retained": bool(args.keep_work_dir),
        "raw_pick_stats": raw_stats,
        "picks_before_nms": picks_before_nms,
        "real_input_pick_count": picks_after_nms,
        "includes_real_input_picks": bool(args.include_input_picks),
        "jsonl_record_counts": {
            "real_association_run": 1,
            "real_input_pick": picks_after_nms if args.include_input_picks else 0,
            "real_event": len(events),
        },
        "station_count_net_sta": len(stations),
        "real_outputs": outputs,
        "event_count": len(events),
        "associated_pick_count": sum(len(event.get("picks", [])) for event in events),
        "unique_associated_input_pick_count": associated_input_pick_count,
        "unassociated_input_pick_count": picks_after_nms - associated_input_pick_count,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Associate SeismicX-Cont picker JSONL picks with REAL and write earthquake JSONL."
    )
    parser.add_argument("--picks-jsonl", type=Path, default=Path("data/picks/pnsn_v3_diff.mini.phase.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/associated/real_events.jsonl"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--station-csv", type=Path, default=Path("data/label/stations.csv"))
    parser.add_argument("--starttime", default=None)
    parser.add_argument("--endtime", default=None)
    parser.add_argument("--min-prob", type=float, default=0.85)
    parser.add_argument("--nms-window", type=float, default=0.50)
    parser.add_argument("--match-tolerance", type=float, default=0.05)
    parser.add_argument("--real-src", type=Path, default=ROOT / "scripts/real/real.c")
    parser.add_argument("--real-bin", type=Path, default=None)
    parser.add_argument("--ttdb", type=Path, default=ROOT / "scripts/real/tt_db/tdb.txt")
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    parser.add_argument("--timeout-per-day", type=float, default=300.0)
    parser.add_argument("--lat-center", type=float, default=None)
    parser.add_argument("--real-r", default="0.4/20/0.02/2/5")
    parser.add_argument("--real-g", default="2.0/20/0.01/1")
    parser.add_argument("--real-s", default="3/0/3/0/1.0/0.1/1.5")
    parser.add_argument("--real-v", default="6.2/3.5")
    parser.add_argument("--include-input-picks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    args.picks_jsonl = resolve_path(args.picks_jsonl)
    args.output_jsonl = resolve_path(args.output_jsonl)
    args.station_csv = resolve_path(args.station_csv) if args.station_csv else None
    args.real_src = resolve_path(args.real_src)
    args.ttdb = resolve_path(args.ttdb)
    if args.real_bin:
        args.real_bin = resolve_path(args.real_bin)
    if args.work_dir is None:
        args.work_dir = args.output_jsonl.with_suffix("").parent / (
            args.output_jsonl.with_suffix("").name + ".real_work"
        )
    else:
        args.work_dir = resolve_path(args.work_dir)

    if not args.picks_jsonl.exists():
        raise FileNotFoundError(args.picks_jsonl)
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir, ignore_errors=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    picks, stations, raw_stats = load_picks(args)
    picks_before_nms = len(picks)
    picks = apply_nms(picks, args.nms_window)
    assign_input_pick_ids(picks)
    picks_after_nms = len(picks)
    if not picks:
        raise RuntimeError("No picks left after filtering.")

    pick_dir, station_file, _input_stats = write_real_inputs(picks, stations, args.work_dir)
    real_bin = compile_real(args, args.work_dir)
    outputs = run_real_for_days(args, real_bin, pick_dir, station_file, stations, args.work_dir)
    events = parse_real_outputs(outputs, picks, args)
    write_jsonl(
        events=events,
        input_picks=picks,
        output_jsonl=args.output_jsonl,
        args=args,
        raw_stats=raw_stats,
        picks_before_nms=picks_before_nms,
        stations=stations,
    )
    write_summary(
        args.output_jsonl.with_suffix(".summary.json"),
        args,
        raw_stats,
        picks_before_nms,
        picks_after_nms,
        stations,
        outputs,
        events,
    )

    print(f"[OK] accepted picks before NMS: {picks_before_nms}")
    print(f"[OK] picks after NMS: {picks_after_nms}")
    print(f"[OK] stations: {len(stations)}")
    print(f"[OK] REAL events: {len(events)}")
    print(f"[OK] wrote: {args.output_jsonl}")
    print(f"[OK] summary: {args.output_jsonl.with_suffix('.summary.json')}")
    if args.keep_work_dir:
        print(f"[INFO] work directory retained for inspection: {args.work_dir}")
    else:
        shutil.rmtree(args.work_dir, ignore_errors=True)
        print(f"[INFO] removed intermediate work directory: {args.work_dir}")


if __name__ == "__main__":
    main()
