#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import h5py
import numpy as np
from obspy import read, UTCDateTime


DEFAULT_LOCATION = "--"


def normalize_location(location, default=DEFAULT_LOCATION):
    if location is None:
        return default
    location = str(location).strip()
    return location if location else default


def make_station_id(network, station, location, default_location=DEFAULT_LOCATION):
    network = str(network or "").strip()
    station = str(station or "").strip()
    location = normalize_location(location, default_location)
    return f"{network}.{station}.{location}"


def make_station_key(network, station):
    network = str(network or "").strip()
    station = str(station or "").strip()
    return f"{network}.{station}"


def split_station_id(station_id, default_location=DEFAULT_LOCATION):
    parts = str(station_id).split(".")
    network = parts[0] if len(parts) > 0 else ""
    station = parts[1] if len(parts) > 1 else ""
    location = parts[2] if len(parts) > 2 else default_location
    return network, station, normalize_location(location, default_location)


def parse_utc_or_none(value):
    value = str(value or "").strip()
    if not value:
        return None
    return UTCDateTime(value)


def utc_to_group_id(t: UTCDateTime, level: str) -> str:
    if level == "year":
        return f"{t.year:04d}-01-01T00:00:00.000000Z"
    if level == "day":
        return f"{t.year:04d}-{t.month:02d}-{t.day:02d}T00:00:00.000000Z"
    raise ValueError(f"Unsupported level: {level}")


def floor_utc_to_interval(t: UTCDateTime, interval_seconds):
    """Return the UTC start time of the fixed-length interval containing t."""
    if interval_seconds is None:
        return None
    interval_seconds = int(interval_seconds)
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    ts = float(t.timestamp)
    start_ts = int(ts // interval_seconds) * interval_seconds
    return UTCDateTime(start_ts)


def interval_file_id(t: UTCDateTime, interval_seconds) -> str:
    """Create a safe file id from the interval start time."""
    if interval_seconds is None:
        return "single"

    interval_start = floor_utc_to_interval(t, interval_seconds)

    if int(interval_seconds) == 86400:
        return f"{interval_start.year:04d}{interval_start.month:02d}{interval_start.day:02d}"

    if int(interval_seconds) == 3600:
        return (
            f"{interval_start.year:04d}{interval_start.month:02d}{interval_start.day:02d}_"
            f"{interval_start.hour:02d}"
        )

    return (
        f"{interval_start.year:04d}{interval_start.month:02d}{interval_start.day:02d}T"
        f"{interval_start.hour:02d}{interval_start.minute:02d}{interval_start.second:02d}_"
        f"{int(interval_seconds)}s"
    )


def split_mode_from_seconds(interval_seconds):
    if interval_seconds is None:
        return "single"
    if int(interval_seconds) == 86400:
        return "day"
    if int(interval_seconds) == 3600:
        return "hour"
    return "custom"


def set_common_attrs(obj, level, node_type, parent_type):
    obj.attrs["level"] = level
    obj.attrs["type"] = node_type
    obj.attrs["parent_type"] = parent_type


def load_station_locations_csv(loc_file, default_location=DEFAULT_LOCATION):
    """
    支持两种格式：

    1. 有表头：
       net,sta,lat,lon,elev_m,start,end

    2. 无表头：
       CI,WBM,35.60839,-117.89049,892.0,1979-09-26T00:00:00.000000Z,3000-01-01T00:00:00.000000Z

    注意：
    位置匹配只使用 network.station，不使用 location。
    """
    locations = defaultdict(list)

    if not loc_file or not os.path.exists(loc_file):
        print(f"[WARN] Location CSV file not found: {loc_file}")
        return dict(locations)

    with open(loc_file, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.readline()
        f.seek(0)

        first_cols = [x.strip().lower() for x in sample.strip().split(",")]
        has_header = {"net", "sta", "lat", "lon"}.issubset(set(first_cols))

        if has_header:
            reader = csv.DictReader(f)

            for line_no, row in enumerate(reader, start=2):
                try:
                    net = row["net"].strip()
                    sta = row["sta"].strip()
                    loc = normalize_location(row.get("location", default_location), default_location)

                    start = parse_utc_or_none(row["start"])
                    end = parse_utc_or_none(row["end"])

                    key = make_station_key(net, sta)

                    locations[key].append(
                        {
                            "network": net,
                            "station": sta,
                            "location": loc,
                            "latitude": float(row["lat"]),
                            "longitude": float(row["lon"]),
                            "elevation": float(row["elev_m"]),
                            "start": start,
                            "end": end,
                            "starttime": str(start) if start is not None else "",
                            "endtime": str(end) if end is not None else "",
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Failed to parse location CSV line {line_no}: {row}, error={e}")

        else:
            reader = csv.reader(f)

            for line_no, row in enumerate(reader, start=1):
                if not row or len(row) < 7:
                    continue

                try:
                    net = row[0].strip()
                    sta = row[1].strip()
                    lat = float(row[2])
                    lon = float(row[3])
                    elev = float(row[4])
                    start = parse_utc_or_none(row[5])
                    end = parse_utc_or_none(row[6])

                    key = make_station_key(net, sta)

                    locations[key].append(
                        {
                            "network": net,
                            "station": sta,
                            "location": default_location,
                            "latitude": lat,
                            "longitude": lon,
                            "elevation": elev,
                            "start": start,
                            "end": end,
                            "starttime": str(start) if start is not None else "",
                            "endtime": str(end) if end is not None else "",
                        }
                    )
                except Exception as e:
                    print(f"[WARN] Failed to parse location CSV line {line_no}: {row}, error={e}")

    for key in locations:
        locations[key].sort(
            key=lambda x: x["start"] if x["start"] is not None else UTCDateTime(0)
        )

    return dict(locations)


def match_station_location(
    station_locations,
    station_id,
    trace_start=None,
    trace_end=None,
    allow_fallback=True,
):
    """
    只按 network.station 匹配台站位置。
    不使用 location code。

    例如：
        waveform station_id = BK.BDM.00
        location key        = BK.BDM
    """
    net, sta, _ = split_station_id(station_id)
    station_key = make_station_key(net, sta)

    records = station_locations.get(station_key, [])

    if not records:
        return None, "default_nan_no_station_record"

    if trace_start is None and trace_end is None:
        if allow_fallback:
            return records[-1], "fallback_nearest_time_network_station_only"
        return None, "default_nan_no_time_matched_position"

    matched = []

    for rec in records:
        rec_start = rec.get("start")
        rec_end = rec.get("end")

        left_ok = True if rec_end is None or trace_start is None else trace_start < rec_end
        right_ok = True if rec_start is None or trace_end is None else trace_end >= rec_start

        if left_ok and right_ok:
            matched.append(rec)

    if matched:
        def strict_score(rec):
            rec_start = rec.get("start")
            if rec_start is None or trace_start is None:
                return 0
            if rec_start <= trace_start:
                return abs(trace_start - rec_start)
            return abs(trace_start - rec_start) + 1e12

        return sorted(matched, key=strict_score)[0], "strict_time_matched_network_station_only"

    if not allow_fallback:
        return None, "default_nan_no_time_matched_position"

    def fallback_score(rec):
        if trace_start is None:
            return 0

        candidates = []
        if rec.get("start") is not None:
            candidates.append(abs(trace_start - rec["start"]))
        if rec.get("end") is not None:
            candidates.append(abs(trace_start - rec["end"]))

        return min(candidates) if candidates else 0

    return sorted(records, key=fallback_score)[0], "fallback_nearest_time_network_station_only"


def find_mseed_files(input_dir):
    input_dir = Path(input_dir)

    suffixes = {
        ".mseed", ".msd", ".miniseed", ".seed",
        ".MSEED", ".MSD", ".MINISEED", ".SEED",
    }

    return sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix in suffixes
    )


def build_record_from_trace_chunk(
    tr,
    mseed_file,
    station_id,
    channel,
    network,
    station,
    location,
    data,
    idx_start,
    idx_end,
    split_interval_seconds,
    default_location=DEFAULT_LOCATION,
):
    """Build one HDF5 segment record from a trace chunk.

    idx_end is exclusive. Samples are assigned to files by their sample time.
    This means an input trace crossing an hour/day/custom boundary is physically
    cut into multiple HDF5 datasets rather than merely written to the file of
    its first sample.
    """
    delta = float(tr.stats.delta)
    sr = float(tr.stats.sampling_rate)

    chunk_start = tr.stats.starttime + idx_start * delta
    chunk_end = tr.stats.starttime + (idx_end - 1) * delta
    interval_start = floor_utc_to_interval(chunk_start, split_interval_seconds)
    interval_end = (interval_start + int(split_interval_seconds)) if interval_start is not None else None

    return {
        "year_id": utc_to_group_id(chunk_start, "year"),
        "day_id": utc_to_group_id(chunk_start, "day"),
        "split_file_id": interval_file_id(chunk_start, split_interval_seconds),
        "split_interval_seconds": -1 if split_interval_seconds is None else int(split_interval_seconds),
        "split_interval_starttime": str(interval_start) if interval_start is not None else "",
        "split_interval_endtime": str(interval_end) if interval_end is not None else "",
        "station_id": station_id,
        "channel": channel,
        "starttime_obj": chunk_start,
        "endtime_obj": chunk_end,
        "starttime": str(chunk_start),
        "endtime": str(chunk_end),
        "sampling_rate": sr,
        "delta": delta,
        "npts": int(idx_end - idx_start),
        "network": network,
        "station": station,
        "location": location,
        "data": np.asarray(data[idx_start:idx_end]),
        "dtype": str(data.dtype),
        "source_file": str(mseed_file),
        "source_trace_starttime": str(tr.stats.starttime),
        "source_trace_endtime": str(tr.stats.endtime),
        "source_trace_npts": int(tr.stats.npts),
    }


def trace_to_records(tr, mseed_file, default_location=DEFAULT_LOCATION, split_interval_seconds=None):
    records = []

    net = tr.stats.network or ""
    sta = tr.stats.station or ""
    loc = normalize_location(tr.stats.location, default_location)
    cha = tr.stats.channel or ""
    station_id = make_station_id(net, sta, loc, default_location)

    data = np.asarray(tr.data)
    npts = int(tr.stats.npts)
    if npts <= 0:
        return records

    if split_interval_seconds is None:
        records.append(
            build_record_from_trace_chunk(
                tr=tr,
                mseed_file=mseed_file,
                station_id=station_id,
                channel=cha,
                network=net,
                station=sta,
                location=loc,
                data=data,
                idx_start=0,
                idx_end=npts,
                split_interval_seconds=None,
                default_location=default_location,
            )
        )
        return records

    split_interval_seconds = int(split_interval_seconds)
    if split_interval_seconds <= 0:
        raise ValueError("split_interval_seconds must be positive or None")

    sr = float(tr.stats.sampling_rate)
    idx_start = 0

    while idx_start < npts:
        sample_time = tr.stats.starttime + idx_start / sr
        interval_start = floor_utc_to_interval(sample_time, split_interval_seconds)
        next_boundary = interval_start + split_interval_seconds

        # First sample with sample_time >= next_boundary.
        idx_end = int(np.ceil((float(next_boundary - tr.stats.starttime) * sr) - 1e-9))
        idx_end = max(idx_start + 1, min(idx_end, npts))

        records.append(
            build_record_from_trace_chunk(
                tr=tr,
                mseed_file=mseed_file,
                station_id=station_id,
                channel=cha,
                network=net,
                station=sta,
                location=loc,
                data=data,
                idx_start=idx_start,
                idx_end=idx_end,
                split_interval_seconds=split_interval_seconds,
                default_location=default_location,
            )
        )

        idx_start = idx_end

    return records


def read_one_mseed(mseed_file, default_location=DEFAULT_LOCATION, split_interval_seconds=None):
    records = []

    try:
        st = read(str(mseed_file))
    except Exception as e:
        return records, f"[WARN] Failed to read {mseed_file}: {e}"

    for tr in st:
        records.extend(
            trace_to_records(
                tr=tr,
                mseed_file=mseed_file,
                default_location=default_location,
                split_interval_seconds=split_interval_seconds,
            )
        )

    return records, None


def write_position_attrs(obj, matched, match_mode):
    obj.attrs["position_match_mode"] = match_mode
    obj.attrs["position_is_fallback"] = "fallback" in str(match_mode)

    if matched is not None:
        obj.attrs["longitude"] = matched.get("longitude", np.nan)
        obj.attrs["latitude"] = matched.get("latitude", np.nan)
        obj.attrs["elevation"] = matched.get("elevation", np.nan)
        obj.attrs["location_available"] = True
        obj.attrs["location_source"] = match_mode
        obj.attrs["station_position_starttime"] = matched.get("starttime", "")
        obj.attrs["station_position_endtime"] = matched.get("endtime", "")
    else:
        obj.attrs["longitude"] = np.nan
        obj.attrs["latitude"] = np.nan
        obj.attrs["elevation"] = np.nan
        obj.attrs["location_available"] = False
        obj.attrs["location_source"] = match_mode
        obj.attrs["station_position_starttime"] = ""
        obj.attrs["station_position_endtime"] = ""


def write_station_position_history(station_grp, station_id, station_locations, default_location):
    if "position_history" in station_grp:
        return

    net, sta, _ = split_station_id(station_id, default_location)
    station_key = make_station_key(net, sta)

    pos_grp = station_grp.create_group("position_history")
    set_common_attrs(pos_grp, "position_history", "position_history_group", "station_group")

    records = station_locations.get(station_key, [])
    pos_grp.attrs["record_count"] = len(records)
    pos_grp.attrs["match_key"] = station_key
    pos_grp.attrs["match_rule"] = "network.station only; location ignored"

    for i, rec in enumerate(records):
        item_grp = pos_grp.create_group(str(i))
        set_common_attrs(item_grp, "position_record", "position_record_group", "position_history_group")

        item_grp.attrs["network"] = rec.get("network", "")
        item_grp.attrs["station"] = rec.get("station", "")
        item_grp.attrs["location"] = rec.get("location", default_location)
        item_grp.attrs["longitude"] = rec.get("longitude", np.nan)
        item_grp.attrs["latitude"] = rec.get("latitude", np.nan)
        item_grp.attrs["elevation"] = rec.get("elevation", np.nan)
        item_grp.attrs["starttime"] = rec.get("starttime", "")
        item_grp.attrs["endtime"] = rec.get("endtime", "")


def init_hdf5_root(h5, default_location, split_interval_seconds=None):
    set_common_attrs(h5, "root", "hdf5_file", "none")
    h5.attrs["description"] = "Continuous waveform dataset converted from MiniSEED"
    h5.attrs["station_id_format"] = "network.station.location"
    h5.attrs["station_location_match_rule"] = "network.station only; location ignored"
    h5.attrs["empty_location_value"] = default_location
    h5.attrs["missing_coordinate_value"] = "NaN"
    h5.attrs["station_location_format"] = (
        "CSV with header: net,sta,lat,lon,elev_m,start,end "
        "or no-header: net,sta,lat,lon,elev_m,start,end"
    )
    h5.attrs["split_mode"] = split_mode_from_seconds(split_interval_seconds)
    h5.attrs["split_interval_seconds"] = -1 if split_interval_seconds is None else int(split_interval_seconds)


def get_or_create_station_group(
    h5,
    year_id,
    day_id,
    station_id,
    station_locations,
    trace_start,
    trace_end,
    default_location,
):
    year_grp = h5.require_group(year_id)
    set_common_attrs(year_grp, "year", "year_group", "root")
    year_grp.attrs["utc_time"] = year_id

    day_grp = year_grp.require_group(day_id)
    set_common_attrs(day_grp, "day", "day_group", "year_group")
    day_grp.attrs["utc_time"] = day_id

    stations_grp = day_grp.require_group("stations")
    set_common_attrs(stations_grp, "stations", "stations_group", "day_group")
    stations_grp.attrs["description"] = "Container group for all stations under this day"

    station_grp = stations_grp.require_group(station_id)
    set_common_attrs(station_grp, "station", "station_group", "stations_group")

    network, station, location = split_station_id(station_id, default_location)
    station_grp.attrs["station_id"] = station_id
    station_grp.attrs["station_key"] = make_station_key(network, station)
    station_grp.attrs["network"] = network
    station_grp.attrs["station"] = station
    station_grp.attrs["location"] = location
    station_grp.attrs["location_default_value"] = default_location
    station_grp.attrs["location_is_default"] = location == default_location
    station_grp.attrs["instrument_time_range_start"] = str(trace_start)
    station_grp.attrs["instrument_time_range_end"] = str(trace_end)

    matched, match_mode = match_station_location(
        station_locations=station_locations,
        station_id=station_id,
        trace_start=trace_start,
        trace_end=trace_end,
        allow_fallback=True,
    )
    write_position_attrs(station_grp, matched, match_mode)

    write_station_position_history(
        station_grp=station_grp,
        station_id=station_id,
        station_locations=station_locations,
        default_location=default_location,
    )

    waveform_grp = station_grp.require_group("waveform")
    set_common_attrs(waveform_grp, "waveform", "waveform_group", "station_group")

    return station_grp, waveform_grp


def next_dataset_index(channel_grp):
    max_idx = -1
    for key in channel_grp.keys():
        if str(key).isdigit():
            max_idx = max(max_idx, int(key))
    return max_idx + 1


def update_channel_summary_attrs(channel_grp, rec):
    channel_grp.attrs["channel"] = rec["channel"]

    old_count = int(channel_grp.attrs.get("segment_count", 0))
    channel_grp.attrs["segment_count"] = old_count + 1

    rec_start = rec["starttime_obj"]
    rec_end = rec["endtime_obj"]

    old_start = channel_grp.attrs.get("starttime", "")
    old_end = channel_grp.attrs.get("endtime", "")

    if not old_start:
        channel_grp.attrs["starttime"] = str(rec_start)
    else:
        old_start_t = UTCDateTime(str(old_start))
        channel_grp.attrs["starttime"] = str(min(old_start_t, rec_start))

    if not old_end:
        channel_grp.attrs["endtime"] = str(rec_end)
    else:
        old_end_t = UTCDateTime(str(old_end))
        channel_grp.attrs["endtime"] = str(max(old_end_t, rec_end))


def write_one_record(
    h5,
    rec,
    station_locations,
    default_location,
    compression,
    compression_opts,
    shuffle,
):
    station_grp, waveform_grp = get_or_create_station_group(
        h5=h5,
        year_id=rec["year_id"],
        day_id=rec["day_id"],
        station_id=rec["station_id"],
        station_locations=station_locations,
        trace_start=rec["starttime_obj"],
        trace_end=rec["endtime_obj"],
        default_location=default_location,
    )

    channel_grp = waveform_grp.require_group(rec["channel"])
    set_common_attrs(channel_grp, "channel", "channel_group", "waveform_group")

    update_channel_summary_attrs(channel_grp, rec)

    matched, match_mode = match_station_location(
        station_locations=station_locations,
        station_id=rec["station_id"],
        trace_start=rec["starttime_obj"],
        trace_end=rec["endtime_obj"],
        allow_fallback=True,
    )
    write_position_attrs(channel_grp, matched, match_mode)

    ds_name = str(next_dataset_index(channel_grp))

    create_kwargs = {}
    if compression and compression.lower() != "none":
        create_kwargs["compression"] = compression
        if compression.lower() == "gzip":
            create_kwargs["compression_opts"] = compression_opts
        create_kwargs["shuffle"] = shuffle

    ds = channel_grp.create_dataset(
        ds_name,
        data=rec["data"],
        **create_kwargs,
    )

    set_common_attrs(ds, "segment", "waveform_dataset", "channel_group")

    write_position_attrs(ds, matched, match_mode)

    ds.attrs["segment_index"] = int(ds_name)
    ds.attrs["network"] = rec["network"]
    ds.attrs["station"] = rec["station"]
    ds.attrs["station_key"] = make_station_key(rec["network"], rec["station"])
    ds.attrs["location"] = rec["location"]
    ds.attrs["location_is_default"] = rec["location"] == default_location
    ds.attrs["channel"] = rec["channel"]
    ds.attrs["sampling_rate"] = rec["sampling_rate"]
    ds.attrs["delta"] = rec["delta"]
    ds.attrs["npts"] = rec["npts"]
    ds.attrs["starttime"] = rec["starttime"]
    ds.attrs["endtime"] = rec["endtime"]
    ds.attrs["mseed_source_file"] = rec["source_file"]
    ds.attrs["dtype"] = rec["dtype"]
    ds.attrs["split_file_id"] = rec.get("split_file_id", "single")
    ds.attrs["split_interval_seconds"] = rec.get("split_interval_seconds", -1)
    ds.attrs["split_interval_starttime"] = rec.get("split_interval_starttime", "")
    ds.attrs["split_interval_endtime"] = rec.get("split_interval_endtime", "")
    ds.attrs["source_trace_starttime"] = rec.get("source_trace_starttime", rec["starttime"])
    ds.attrs["source_trace_endtime"] = rec.get("source_trace_endtime", rec["endtime"])
    ds.attrs["source_trace_npts"] = rec.get("source_trace_npts", rec["npts"])


def output_path_for_interval(output, split_file_id):
    output = Path(output)

    if output.suffix.lower() in [".h5", ".hdf5"]:
        out_dir = output.parent
        stem = output.stem
    else:
        out_dir = output
        stem = "continuous_waveform"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_{split_file_id}.h5"


def convert_mseed_to_hdf5_streaming(
    mseed_files,
    station_locations,
    output_file,
    num_workers=4,
    max_pending=16,
    default_location=DEFAULT_LOCATION,
    compression="gzip",
    compression_opts=4,
    shuffle=True,
    split_interval_seconds=86400,
    include_split_file_ids=None,
):
    total = len(mseed_files)
    submitted = 0
    finished = 0
    written_records = 0
    skipped_records = 0
    include_split_file_ids = set(include_split_file_ids or [])

    h5_files = {}

    def get_h5_for_record(rec):
        if split_interval_seconds is None:
            key = "__single__"
            if key not in h5_files:
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                h5 = h5py.File(output_path, "w")
                init_hdf5_root(h5, default_location, split_interval_seconds=None)
                h5_files[key] = h5
            return h5_files[key]

        key = rec["split_file_id"]
        if key not in h5_files:
            output_path = output_path_for_interval(output_file, key)
            h5 = h5py.File(output_path, "w")
            init_hdf5_root(h5, default_location, split_interval_seconds=split_interval_seconds)
            h5.attrs["split_file_id"] = key
            h5.attrs["split_interval_seconds"] = int(split_interval_seconds)
            h5.attrs["split_interval_starttime"] = rec.get("split_interval_starttime", "")
            h5.attrs["split_interval_endtime"] = rec.get("split_interval_endtime", "")
            h5_files[key] = h5
        return h5_files[key]

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            pending = set()

            def submit_more():
                nonlocal submitted
                while submitted < total and len(pending) < max_pending:
                    future = executor.submit(
                        read_one_mseed,
                        mseed_files[submitted],
                        default_location,
                        split_interval_seconds,
                    )
                    pending.add(future)
                    submitted += 1

            submit_more()

            while pending:
                done, pending_remaining = wait(pending, return_when=FIRST_COMPLETED)
                pending = pending_remaining

                for future in done:
                    finished += 1
                    records, warning = future.result()

                    if warning:
                        print(warning)

                    records.sort(
                        key=lambda r: (
                            r["split_file_id"],
                            r["year_id"],
                            r["day_id"],
                            r["station_id"],
                            r["channel"],
                            r["starttime_obj"],
                        )
                    )

                    for rec in records:
                        if include_split_file_ids and rec["split_file_id"] not in include_split_file_ids:
                            skipped_records += 1
                            continue

                        h5 = get_h5_for_record(rec)

                        write_one_record(
                            h5=h5,
                            rec=rec,
                            station_locations=station_locations,
                            default_location=default_location,
                            compression=compression,
                            compression_opts=compression_opts,
                            shuffle=shuffle,
                        )
                        written_records += 1

                    if finished % 100 == 0 or finished == total:
                        print(
                            f"[INFO] Progress: files {finished}/{total}, "
                            f"written waveform segments {written_records}, "
                            f"open hdf5 files {len(h5_files)}"
                        )

                    del records

                submit_more()

    finally:
        for h5 in h5_files.values():
            h5.close()

    print(f"[OK] Written waveform segments: {written_records}")
    if include_split_file_ids:
        print(
            f"[OK] Kept split_file_id(s): {', '.join(sorted(include_split_file_ids))}; "
            f"skipped waveform segments outside selection: {skipped_records}"
        )

    if split_interval_seconds is None:
        print(f"[OK] HDF5 written to: {output_file}")
    else:
        out_dir = Path(output_file).parent if Path(output_file).suffix else output_file
        print(
            f"[OK] HDF5 files written by {split_mode_from_seconds(split_interval_seconds)} "
            f"intervals ({int(split_interval_seconds)} s) under: {out_dir}"
        )


def resolve_split_interval_seconds(args):
    if getattr(args, "split_by_day", False):
        return 86400

    if args.split_interval == "single":
        return None
    if args.split_interval == "day":
        return 86400
    if args.split_interval == "hour":
        return 3600
    if args.split_interval == "custom":
        if args.custom_interval_seconds <= 0:
            raise ValueError("--custom_interval_seconds must be positive")
        return int(args.custom_interval_seconds)

    raise ValueError(f"Unsupported split interval: {args.split_interval}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MiniSEED files to hierarchical HDF5."
    )

    parser.add_argument(
        "--input_dir",
        default="/Volumes/Data/continous_dataset_tool/data/continous_usa/data/07/01",
        help="Directory containing MiniSEED files.",
    )

    parser.add_argument(
        "--loc_file",
        default="data/label/stations.csv",
        help="Station CSV file.",
    )

    parser.add_argument(
        "--output",
        default="data/hdf5_one_day_with_hour_segment/continuous_waveform_usa.h5",
        help=(
            "Output HDF5 file or filename prefix. With --split_interval day/hour/custom, "
            "this is used as a prefix, e.g. continuous_waveform_usa_20190701.h5 "
            "or continuous_waveform_usa_20190701_13.h5."
        ),
    )

    parser.add_argument(
        "--split_interval",
        default="hour",
        choices=["single", "day", "hour", "custom"],
        help=(
            "Output split interval. "
            "single = one HDF5 file; day = one file per day; "
            "hour = one file per hour; custom = use --custom_interval_seconds."
        ),
    )

    parser.add_argument(
        "--custom_interval_seconds",
        type=int,
        default=3600,
        help="Custom output interval length in seconds when --split_interval custom.",
    )

    parser.add_argument(
        "--split_by_day",
        action="store_true",
        help="Backward-compatible alias: equivalent to --split_interval day.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of threads for reading MiniSEED files.",
    )

    parser.add_argument(
        "--max_pending",
        type=int,
        default=4,
        help="Maximum number of pending read tasks.",
    )

    parser.add_argument(
        "--default_location",
        default=DEFAULT_LOCATION,
        help='Default location code when MiniSEED location is empty. Default: "--".',
    )

    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Dataset compression method.",
    )

    parser.add_argument(
        "--compression_opts",
        type=int,
        default=4,
        help="Compression level for gzip.",
    )

    parser.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Disable HDF5 shuffle filter.",
    )

    parser.add_argument(
        "--include_split_file_id",
        action="append",
        default=[],
        help=(
            "Only write selected split file id(s), e.g. 20190706_04. "
            "Can be supplied multiple times. Useful for building small hour subsets."
        ),
    )

    args = parser.parse_args()

    split_interval_seconds = resolve_split_interval_seconds(args)
    print(
        f"[INFO] Output split mode: {split_mode_from_seconds(split_interval_seconds)} "
        f"({split_interval_seconds if split_interval_seconds is not None else 'single file'} s)"
    )

    station_locations = load_station_locations_csv(
        args.loc_file,
        default_location=args.default_location,
    )
    print(f"[INFO] Loaded station location histories for {len(station_locations)} station keys.")

    mseed_files = find_mseed_files(args.input_dir)
    print(f"[INFO] Found {len(mseed_files)} MiniSEED files.")

    convert_mseed_to_hdf5_streaming(
        mseed_files=mseed_files,
        station_locations=station_locations,
        output_file=args.output,
        num_workers=args.num_workers,
        max_pending=args.max_pending,
        default_location=args.default_location,
        compression=args.compression,
        compression_opts=args.compression_opts,
        shuffle=not args.no_shuffle,
        split_interval_seconds=split_interval_seconds,
        include_split_file_ids=args.include_split_file_id,
    )


if __name__ == "__main__":
    main()
