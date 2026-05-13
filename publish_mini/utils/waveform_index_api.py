#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
from collections import defaultdict

import h5py
import numpy as np
from obspy import UTCDateTime


def parse_time_to_epoch(t):
    return float(UTCDateTime(t).timestamp)


def epoch_to_utc(t):
    return str(UTCDateTime(float(t)))


def wildcard_to_sql_like(pattern):
    """
    Convert wildcard pattern to SQL LIKE pattern.

    Examples:
        "*"   -> "%"
        "BK"  -> "BK"
        "B*"  -> "B%"
        "*Z"  -> "%Z"
        "BH*" -> "BH%"
    """
    if pattern is None or str(pattern).strip() == "":
        return "%"

    return str(pattern).strip().replace("*", "%")


def query_segments(
    db_file,
    network="*",
    station="*",
    location="*",
    channel="*",
    starttime=None,
    endtime=None,
    limit=None,
):
    """
    Query waveform segments from SQLite index.

    Parameters
    ----------
    db_file : str
        SQLite index database path.
    network : str
        Network code or wildcard pattern, e.g. "BK", "C*", "*".
    station : str
        Station code or wildcard pattern, e.g. "BDM", "BD*", "*".
    location : str
        Location code or wildcard pattern, e.g. "00", "--", "*".
    channel : str
        Channel code or wildcard pattern, e.g. "BHE", "BH*", "*Z", "*".
    starttime : str
        Query start time.
    endtime : str
        Query end time.
    limit : int or None
        Optional maximum number of returned segments.

    Returns
    -------
    rows : list[dict]
        Matched waveform segment metadata.
    """

    if starttime is None or endtime is None:
        raise ValueError("starttime and endtime are required.")

    query_start = parse_time_to_epoch(starttime)
    query_end = parse_time_to_epoch(endtime)

    sql = """
    SELECT *
    FROM waveform_segments
    WHERE network LIKE ?
      AND station LIKE ?
      AND location LIKE ?
      AND channel LIKE ?
      AND end_epoch >= ?
      AND start_epoch <= ?
    ORDER BY network, station, location, channel, start_epoch
    """

    params = [
        wildcard_to_sql_like(network),
        wildcard_to_sql_like(station),
        wildcard_to_sql_like(location),
        wildcard_to_sql_like(channel),
        query_start,
        query_end,
    ]

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row

    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return rows


def read_hdf5_segment(row):
    """
    Read a single waveform segment from HDF5 using h5_file and dataset_path.
    """
    with h5py.File(row["h5_file"], "r") as h5:
        data = h5[row["dataset_path"]][()]

    return np.asarray(data)


def trim_segment_to_window(data, row, starttime, endtime):
    """
    Trim one segment to the requested time window.
    """
    sr = float(row["sampling_rate"])

    seg_start = float(row["start_epoch"])
    seg_end = float(row["end_epoch"])

    query_start = parse_time_to_epoch(starttime)
    query_end = parse_time_to_epoch(endtime)

    use_start = max(seg_start, query_start)
    use_end = min(seg_end, query_end)

    if use_end < use_start:
        return data[:0], use_start, use_end

    i0 = int(round((use_start - seg_start) * sr))
    i1 = int(round((use_end - seg_start) * sr)) + 1

    i0 = max(i0, 0)
    i1 = min(i1, len(data))

    return data[i0:i1], use_start, use_end


def merge_segments(
    rows,
    starttime,
    endtime,
    fill_value=0.0,
    dtype=np.float32,
):
    """
    Merge multiple waveform segments into continuous arrays.

    Segments are grouped by:
        network.station.location.channel

    Missing samples are filled by fill_value.

    Returns
    -------
    merged : dict
        {
            "BK.BDM.00.BHE": {
                "data": np.ndarray,
                "network": "BK",
                "station": "BDM",
                "location": "00",
                "channel": "BHE",
                "starttime": "...",
                "endtime": "...",
                "sampling_rate": 100.0,
                "npts": 360001,
                "segments": [...]
            },
            ...
        }
    """

    if len(rows) == 0:
        return {}

    query_start = parse_time_to_epoch(starttime)
    query_end = parse_time_to_epoch(endtime)

    groups = defaultdict(list)

    for row in rows:
        key = (
            row["network"],
            row["station"],
            row["location"],
            row["channel"],
        )
        groups[key].append(row)

    merged = {}

    for key, seg_rows in groups.items():
        network, station, location, channel = key

        seg_rows = sorted(seg_rows, key=lambda r: float(r["start_epoch"]))

        sampling_rates = sorted(set(float(r["sampling_rate"]) for r in seg_rows))
        if len(sampling_rates) != 1:
            raise ValueError(
                f"Multiple sampling rates found for {network}.{station}.{location}.{channel}: "
                f"{sampling_rates}"
            )

        sr = sampling_rates[0]

        npts = int(round((query_end - query_start) * sr)) + 1
        out = np.full(npts, fill_value, dtype=dtype)
        filled = np.zeros(npts, dtype=bool)

        used_segments = []

        for row in seg_rows:
            data = read_hdf5_segment(row)
            data, use_start, use_end = trim_segment_to_window(
                data,
                row,
                starttime=starttime,
                endtime=endtime,
            )

            if len(data) == 0:
                continue

            i0 = int(round((use_start - query_start) * sr))
            i1 = i0 + len(data)

            if i0 < 0:
                data = data[-i0:]
                i0 = 0

            if i1 > npts:
                data = data[: npts - i0]
                i1 = npts

            if i0 >= i1:
                continue

            target = slice(i0, i1)
            mask = ~filled[target]

            out[target][mask] = data[: i1 - i0][mask].astype(dtype, copy=False)
            filled[target][mask] = True

            used_segments.append(
                {
                    "h5_file": row["h5_file"],
                    "dataset_path": row["dataset_path"],
                    "segment_starttime": row["starttime"],
                    "segment_endtime": row["endtime"],
                    "used_starttime": epoch_to_utc(use_start),
                    "used_endtime": epoch_to_utc(use_end),
                    "npts": int(len(data)),
                }
            )

        out_key = f"{network}.{station}.{location}.{channel}"

        merged[out_key] = {
            "data": out,
            "network": network,
            "station": station,
            "location": location,
            "channel": channel,
            "starttime": str(UTCDateTime(starttime)),
            "endtime": str(UTCDateTime(endtime)),
            "sampling_rate": sr,
            "npts": int(len(out)),
            "filled_ratio": float(filled.mean()) if len(filled) > 0 else 0.0,
            "segments": used_segments,
        }

    return merged


def query_and_merge(
    db_file,
    network="*",
    station="*",
    location="*",
    channel="*",
    starttime=None,
    endtime=None,
    fill_value=0.0,
    dtype=np.float32,
    limit=None,
):
    """
    Query waveform segments and merge them into continuous waveforms.
    """
    rows = query_segments(
        db_file=db_file,
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=starttime,
        endtime=endtime,
        limit=limit,
    )

    merged = merge_segments(
        rows=rows,
        starttime=starttime,
        endtime=endtime,
        fill_value=fill_value,
        dtype=dtype,
    )

    return merged, rows


def save_merged_to_npz(merged, output_npz):
    """
    Save merged waveforms and metadata to NPZ.
    """
    arrays = {}
    metadata = {}

    for key, item in merged.items():
        safe_key = key.replace(".", "_").replace("-", "_")
        arrays[safe_key] = item["data"]

        meta = dict(item)
        meta.pop("data", None)
        metadata[safe_key] = meta

    arrays["metadata_json"] = np.array(
        __import__("json").dumps(metadata, ensure_ascii=False)
    )

    np.savez(output_npz, **arrays)