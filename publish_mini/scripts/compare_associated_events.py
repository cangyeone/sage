#!/usr/bin/env python3
"""Compare associated earthquake JSONL against a reference catalog.

The default use case compares ``run_real_association.py`` output with the
SeismicX-Cont annotation JSON. Generic event JSONL files are also supported as
long as each event has an event id, origin time, latitude, longitude, and depth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
from collections import Counter
from pathlib import Path


def parse_iso_time(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    out = dt.datetime.fromisoformat(text)
    if out.tzinfo is not None:
        out = out.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return out


def iso_z(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def as_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(min(1.0, a)))


def event_time(record: dict) -> dt.datetime | None:
    return parse_iso_time(
        record.get("origin_time_iso")
        or record.get("origin_time")
        or record.get("event_time")
        or record.get("time")
    )


def event_depth_km(record: dict):
    depth = record.get("depth_km")
    if depth is not None:
        return as_float(depth)
    depth_m = record.get("depth_m")
    if depth_m is not None:
        return as_float(depth_m) / 1000.0
    return None


def event_magnitude(record: dict):
    for key in ("magnitude", "mag", "magnitude_median"):
        value = as_float(record.get(key))
        if value is not None:
            return value
    preferred = record.get("preferred_magnitude") or {}
    return as_float(preferred.get("mag"))


def normalize_event(record: dict, source: str, fallback_id: str) -> dict | None:
    origin_time = event_time(record)
    latitude = as_float(record.get("latitude"))
    longitude = as_float(record.get("longitude"))
    depth_km = event_depth_km(record)
    if origin_time is None or latitude is None or longitude is None:
        return None
    event_id = (
        record.get("event_id")
        or record.get("id")
        or record.get("real_event_number")
        or fallback_id
    )
    return {
        "event_id": str(event_id),
        "source": source,
        "origin_time": origin_time,
        "origin_time_iso": iso_z(origin_time),
        "latitude": latitude,
        "longitude": longitude,
        "depth_km": depth_km,
        "magnitude": event_magnitude(record),
        "raw_record_type": record.get("record_type"),
    }


def load_event_jsonl(path: Path, accepted_record_types: set[str] | None = None) -> tuple[list[dict], dict]:
    events = []
    run_record = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "real_association_run":
                run_record = record
                continue
            if accepted_record_types and record_type not in accepted_record_types:
                continue
            event = normalize_event(record, source=str(path), fallback_id=f"{path.name}:{line_no}")
            if event is not None:
                events.append(event)
    return events, run_record


def load_seismicx_annotation_catalog(path: Path) -> list[dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    events = []
    years = obj.get("years", {})
    for year_value in years.values():
        for day_value in year_value.get("days", {}).values():
            day_events = day_value.get("events", {})
            iterator = day_events.values() if isinstance(day_events, dict) else day_events
            for item in iterator:
                source_event = item.get("event") or item
                preferred_origin = item.get("preferred_origin") or {}
                merged = {
                    **source_event,
                    "event_id": item.get("event_id") or source_event.get("event_id"),
                    "origin_time_iso": preferred_origin.get("time") or source_event.get("event_time"),
                    "latitude": preferred_origin.get("latitude", source_event.get("latitude")),
                    "longitude": preferred_origin.get("longitude", source_event.get("longitude")),
                    "depth_m": preferred_origin.get("depth_m"),
                    "depth_km": source_event.get("depth_km"),
                    "magnitude": (item.get("preferred_magnitude") or {}).get(
                        "mag", source_event.get("magnitude")
                    ),
                }
                event = normalize_event(merged, source=str(path), fallback_id=f"catalog:{len(events)+1}")
                if event is not None:
                    events.append(event)
    return events


def load_catalog(path: Path, catalog_format: str) -> list[dict]:
    if catalog_format == "seismicx-annotation-json":
        return load_seismicx_annotation_catalog(path)
    if catalog_format == "event-jsonl":
        events, _run = load_event_jsonl(path, accepted_record_types=None)
        return events
    raise ValueError(f"Unsupported catalog format: {catalog_format}")


def infer_time_window(pred_events: list[dict], run_record: dict, args) -> tuple[dt.datetime | None, dt.datetime | None]:
    start = parse_iso_time(args.starttime) if args.starttime else None
    end = parse_iso_time(args.endtime) if args.endtime else None
    if start is None or end is None:
        tw = run_record.get("time_window") or {}
        start = start or parse_iso_time(tw.get("starttime"))
        end = end or parse_iso_time(tw.get("endtime"))
    if (start is None or end is None) and pred_events:
        times = [item["origin_time"] for item in pred_events]
        start = start or min(times)
        end = end or max(times)
    return start, end


def filter_time_window(events: list[dict], start: dt.datetime | None, end: dt.datetime | None, margin_s: float) -> list[dict]:
    if start is None and end is None:
        return events
    out = []
    margin = dt.timedelta(seconds=margin_s)
    for event in events:
        t = event["origin_time"]
        if start is not None and t < start - margin:
            continue
        if end is not None and t >= end + margin:
            continue
        out.append(event)
    return out


def candidate_pair(pred: dict, ref: dict, args) -> dict | None:
    distance_km = haversine_km(pred["latitude"], pred["longitude"], ref["latitude"], ref["longitude"])
    time_error_s = abs((pred["origin_time"] - ref["origin_time"]).total_seconds())
    if distance_km > args.max_epicentral_distance_km:
        return None
    if time_error_s > args.max_origin_time_error_s:
        return None
    depth_error_km = None
    if pred.get("depth_km") is not None and ref.get("depth_km") is not None:
        depth_error_km = abs(pred["depth_km"] - ref["depth_km"])
    magnitude_error = None
    if pred.get("magnitude") is not None and ref.get("magnitude") is not None:
        magnitude_error = pred["magnitude"] - ref["magnitude"]
    score = (
        time_error_s / max(args.max_origin_time_error_s, 1e-9)
        + distance_km / max(args.max_epicentral_distance_km, 1e-9)
    )
    return {
        "pred_event_id": pred["event_id"],
        "ref_event_id": ref["event_id"],
        "time_error_s": time_error_s,
        "epicentral_distance_error_km": distance_km,
        "depth_error_km": depth_error_km,
        "magnitude_error": magnitude_error,
        "score": score,
    }


def match_events(pred_events: list[dict], ref_events: list[dict], args) -> tuple[list[dict], list[dict], list[dict]]:
    candidates = []
    for pred_idx, pred in enumerate(pred_events):
        for ref_idx, ref in enumerate(ref_events):
            pair = candidate_pair(pred, ref, args)
            if pair is None:
                continue
            pair["pred_idx"] = pred_idx
            pair["ref_idx"] = ref_idx
            candidates.append(pair)

    candidates.sort(
        key=lambda x: (
            x["score"],
            x["time_error_s"],
            x["epicentral_distance_error_km"],
            x["pred_event_id"],
            x["ref_event_id"],
        )
    )

    matched_pred = set()
    matched_ref = set()
    matches = []
    for pair in candidates:
        if pair["pred_idx"] in matched_pred or pair["ref_idx"] in matched_ref:
            continue
        pred = pred_events[pair["pred_idx"]]
        ref = ref_events[pair["ref_idx"]]
        matched_pred.add(pair["pred_idx"])
        matched_ref.add(pair["ref_idx"])
        pair = dict(pair)
        pair.pop("pred_idx", None)
        pair.pop("ref_idx", None)
        pair["pred_event"] = event_public_fields(pred)
        pair["ref_event"] = event_public_fields(ref)
        matches.append(pair)

    false_positives = [
        {"record_type": "event_false_positive", "pred_event": event_public_fields(event)}
        for idx, event in enumerate(pred_events)
        if idx not in matched_pred
    ]
    false_negatives = [
        {"record_type": "event_false_negative", "ref_event": event_public_fields(event)}
        for idx, event in enumerate(ref_events)
        if idx not in matched_ref
    ]
    return matches, false_positives, false_negatives


def event_public_fields(event: dict) -> dict:
    return {
        "event_id": event["event_id"],
        "origin_time": event["origin_time_iso"],
        "latitude": event["latitude"],
        "longitude": event["longitude"],
        "depth_km": event.get("depth_km"),
        "magnitude": event.get("magnitude"),
        "source": event.get("source"),
    }


def describe_values(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    sorted_values = sorted(values)
    def percentile(q: float):
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return sorted_values[int(pos)]
        return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
    }


def build_summary(pred_events: list[dict], ref_events: list[dict], matches: list[dict], fp: list[dict], fn: list[dict], args, time_window):
    tp = len(matches)
    n_pred = len(pred_events)
    n_ref = len(ref_events)
    precision = tp / n_pred if n_pred else None
    recall = tp / n_ref if n_ref else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    return {
        "event_matching_protocol": {
            "max_epicentral_distance_km": args.max_epicentral_distance_km,
            "max_origin_time_error_s": args.max_origin_time_error_s,
            "one_to_one_matching": True,
        },
        "time_window": {
            "starttime": iso_z(time_window[0]),
            "endtime": iso_z(time_window[1]),
            "catalog_time_margin_s": args.catalog_time_margin_s,
        },
        "counts": {
            "predicted_events": n_pred,
            "reference_events": n_ref,
            "true_positive_events": tp,
            "false_positive_events": len(fp),
            "false_negative_events": len(fn),
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "errors_for_true_positive_events": {
            "origin_time_error_s": describe_values([m["time_error_s"] for m in matches]),
            "epicentral_distance_error_km": describe_values(
                [m["epicentral_distance_error_km"] for m in matches]
            ),
            "depth_error_km": describe_values(
                [m["depth_error_km"] for m in matches if m.get("depth_error_km") is not None]
            ),
            "magnitude_error": describe_values(
                [m["magnitude_error"] for m in matches if m.get("magnitude_error") is not None]
            ),
        },
    }


def write_outputs(outdir: Path, summary: dict, matches: list[dict], fp: list[dict], fn: list[dict]):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "event_match_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (outdir / "event_matches.jsonl").open("w", encoding="utf-8") as f:
        for match in matches:
            record = {"record_type": "event_true_positive", **match}
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        for record in fp:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        for record in fn:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    with (outdir / "event_match_summary.tsv").open("w", encoding="utf-8") as f:
        counts = summary["counts"]
        metrics = summary["metrics"]
        f.write("metric\tvalue\n")
        for key, value in counts.items():
            f.write(f"{key}\t{value}\n")
        for key, value in metrics.items():
            f.write(f"{key}\t{'' if value is None else value}\n")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Compare associated earthquake events against a reference catalog."
    )
    parser.add_argument("--pred-jsonl", type=Path, default=Path("data/associated/real_events.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/label/annotations_mini_two_hours.json"))
    parser.add_argument(
        "--catalog-format",
        choices=("seismicx-annotation-json", "event-jsonl"),
        default="seismicx-annotation-json",
    )
    parser.add_argument("--outdir", type=Path, default=Path("eval_events/real_vs_catalog"))
    parser.add_argument("--starttime", default=None)
    parser.add_argument("--endtime", default=None)
    parser.add_argument("--catalog-time-margin-s", type=float, default=3.0)
    parser.add_argument("--max-epicentral-distance-km", type=float, default=20.0)
    parser.add_argument("--max-origin-time-error-s", type=float, default=3.0)
    parser.add_argument(
        "--pred-record-types",
        default="real_event,event,catalog_event",
        help="Comma-separated record_type values treated as predicted events. Empty means any record with event fields.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    pred_types = {x.strip() for x in args.pred_record_types.split(",") if x.strip()}
    pred_events, run_record = load_event_jsonl(args.pred_jsonl, accepted_record_types=pred_types or None)
    catalog_events = load_catalog(args.catalog, args.catalog_format)
    time_window = infer_time_window(pred_events, run_record, args)
    catalog_events = filter_time_window(
        catalog_events,
        start=time_window[0],
        end=time_window[1],
        margin_s=args.catalog_time_margin_s,
    )
    pred_events = filter_time_window(pred_events, start=time_window[0], end=time_window[1], margin_s=0.0)

    matches, fp, fn = match_events(pred_events, catalog_events, args)
    summary = build_summary(pred_events, catalog_events, matches, fp, fn, args, time_window)
    write_outputs(args.outdir, summary, matches, fp, fn)

    counts = summary["counts"]
    metrics = summary["metrics"]
    print(f"[OK] predicted events : {counts['predicted_events']}")
    print(f"[OK] reference events : {counts['reference_events']}")
    print(f"[OK] TP/FP/FN         : {counts['true_positive_events']}/{counts['false_positive_events']}/{counts['false_negative_events']}")
    print(f"[OK] precision/recall/F1: {metrics['precision']}/{metrics['recall']}/{metrics['f1']}")
    print(f"[OK] wrote: {args.outdir / 'event_match_summary.json'}")
    print(f"[OK] wrote: {args.outdir / 'event_matches.jsonl'}")


if __name__ == "__main__":
    main()
