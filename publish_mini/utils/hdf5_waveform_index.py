#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build and query a SQLite index for hierarchical HDF5 continuous waveform datasets.

Supported HDF5 structure:

/year_id
  /day_id
    /stations
      /network.station.location
        /waveform
          /channel
            /segment_dataset

Example usage:

1. Build index from a single HDF5 file:

python utils/hdf5_waveform_index.py build \
  --h5 data/continuous_waveform_usa.h5 \
  --db data/index/waveform_index.sqlite

2. Build index from daily HDF5 files:

python utils/hdf5_waveform_index.py build \
  --h5 "data/hdf5/continuous_waveform_usa_*.h5" \
  --db data/index/waveform_index.sqlite

3. Query one station and time range:

python utils/hdf5_waveform_index.py query \
  --db data/index/waveform_index.sqlite \
  --network BK \
  --station BDM \
  --starttime 2019-07-01T00:00:00 \
  --endtime 2019-07-01T01:00:00

4. Query specific channels:

python utils/hdf5_waveform_index.py query \
  --db data/index/waveform_index.sqlite \
  --network BK \
  --station BDM \
  --channels BHE,BHN,BHZ \
  --starttime 2019-07-01T00:00:00 \
  --endtime 2019-07-01T01:00:00

5. Read matched waveform segments into memory:

python utils/hdf5_waveform_index.py read \
  --db data/index/waveform_index.sqlite \
  --network BK \
  --station BDM \
  --channels BHE,BHN,BHZ \
  --starttime 2019-07-01T00:00:00 \
  --endtime 2019-07-01T01:00:00
"""

import argparse
import glob
import json
import os
import sqlite3
from pathlib import Path

import h5py
import numpy as np
from obspy import UTCDateTime


DEFAULT_LOCATION = "--"


# -----------------------------
# Basic utilities
# -----------------------------

def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="ignore")
    return value


def get_attr(obj, name, default=None):
    if name in obj.attrs:
        return decode_attr(obj.attrs[name])
    return default


def normalize_location(location, default=DEFAULT_LOCATION):
    location = decode_attr(location)
    if location is None:
        return default
    location = str(location).strip()
    return location if location else default


def split_station_id(station_id, default_location=DEFAULT_LOCATION):
    parts = str(station_id).split(".")
    network = parts[0] if len(parts) > 0 else ""
    station = parts[1] if len(parts) > 1 else ""
    location = parts[2] if len(parts) > 2 else default_location
    return network, station, normalize_location(location, default_location)


def make_station_key(network, station):
    return f"{str(network).strip()}.{str(station).strip()}"


def parse_time_to_epoch(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return float(UTCDateTime(value).timestamp)


def epoch_to_utc_string(epoch):
    if epoch is None:
        return ""
    return str(UTCDateTime(float(epoch)))


def resolve_h5_files(h5_input):
    """
    Supports:
        - single file
        - list of files
        - directory
        - glob pattern
    """
    if isinstance(h5_input, (list, tuple)):
        out = []
        for item in h5_input:
            out.extend(resolve_h5_files(item))
        return sorted(set(out))

    h5_input = str(h5_input)
    p = Path(h5_input)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        files = []
        files.extend(str(x) for x in p.glob("*.h5"))
        files.extend(str(x) for x in p.glob("*.hdf5"))
        return sorted(files)

    files = sorted(glob.glob(h5_input))
    if files:
        return files

    raise FileNotFoundError(f"No HDF5 files found from input: {h5_input}")


def comma_list(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


# -----------------------------
# SQLite schema
# -----------------------------

def connect_db(db_file):
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn, reset=False):
    cur = conn.cursor()

    if reset:
        cur.execute("DROP TABLE IF EXISTS waveform_segments")
        cur.execute("DROP TABLE IF EXISTS hdf5_files")
        cur.execute("DROP TABLE IF EXISTS stations")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hdf5_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            h5_file TEXT UNIQUE NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS stations (
            station_key TEXT PRIMARY KEY,
            network TEXT NOT NULL,
            station TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            location_available INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS waveform_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            file_id INTEGER NOT NULL,
            h5_file TEXT NOT NULL,
            dataset_path TEXT NOT NULL,

            year_id TEXT,
            day_id TEXT,

            station_id TEXT NOT NULL,
            station_key TEXT NOT NULL,
            network TEXT NOT NULL,
            station TEXT NOT NULL,
            location TEXT,

            channel TEXT NOT NULL,

            starttime TEXT NOT NULL,
            endtime TEXT NOT NULL,
            start_epoch REAL NOT NULL,
            end_epoch REAL NOT NULL,

            sampling_rate REAL,
            delta REAL,
            npts INTEGER,
            dtype TEXT,
            source_file TEXT,

            latitude REAL,
            longitude REAL,
            elevation REAL,
            location_available INTEGER,

            UNIQUE(h5_file, dataset_path)
        )
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_station_time
        ON waveform_segments (network, station, start_epoch, end_epoch)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_station_channel_time
        ON waveform_segments (network, station, channel, start_epoch, end_epoch)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_station_key_time
        ON waveform_segments (station_key, start_epoch, end_epoch)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_location
        ON waveform_segments (location)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_segments_h5_file
        ON waveform_segments (h5_file)
        """
    )

    conn.commit()


def get_or_insert_file_id(conn, h5_file):
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO hdf5_files (h5_file) VALUES (?)",
        (h5_file,),
    )
    conn.commit()

    cur.execute(
        "SELECT file_id FROM hdf5_files WHERE h5_file = ?",
        (h5_file,),
    )
    return int(cur.fetchone()["file_id"])


# -----------------------------
# HDF5 scanning
# -----------------------------

def iter_waveform_datasets(h5_file, default_location=DEFAULT_LOCATION):
    """
    Yield one record per waveform segment dataset.
    """
    with h5py.File(h5_file, "r") as h5:
        for year_id in sorted(h5.keys()):
            year_grp = h5[year_id]
            if not isinstance(year_grp, h5py.Group):
                continue

            for day_id in sorted(year_grp.keys()):
                day_grp = year_grp[day_id]
                if not isinstance(day_grp, h5py.Group):
                    continue

                if "stations" not in day_grp:
                    continue

                stations_grp = day_grp["stations"]

                for station_id in sorted(stations_grp.keys()):
                    station_grp = stations_grp[station_id]
                    if not isinstance(station_grp, h5py.Group):
                        continue

                    if "waveform" not in station_grp:
                        continue

                    waveform_grp = station_grp["waveform"]

                    for channel in sorted(waveform_grp.keys()):
                        channel_grp = waveform_grp[channel]
                        if not isinstance(channel_grp, h5py.Group):
                            continue

                        for ds_key in sorted(channel_grp.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
                            ds = channel_grp[ds_key]
                            if not isinstance(ds, h5py.Dataset):
                                continue

                            dataset_path = ds.name

                            network = get_attr(ds, "network", None)
                            station = get_attr(ds, "station", None)
                            location = get_attr(ds, "location", None)

                            if not network or not station:
                                net2, sta2, loc2 = split_station_id(station_id, default_location)
                                network = network or net2
                                station = station or sta2
                                location = location if location is not None else loc2

                            location = normalize_location(location, default_location)
                            station_key = make_station_key(network, station)

                            starttime = get_attr(ds, "starttime", "")
                            endtime = get_attr(ds, "endtime", "")

                            start_epoch = parse_time_to_epoch(starttime)
                            end_epoch = parse_time_to_epoch(endtime)

                            if start_epoch is None or end_epoch is None:
                                continue

                            yield {
                                "h5_file": str(h5_file),
                                "dataset_path": dataset_path,
                                "year_id": year_id,
                                "day_id": day_id,
                                "station_id": station_id,
                                "station_key": station_key,
                                "network": str(network),
                                "station": str(station),
                                "location": location,
                                "channel": str(get_attr(ds, "channel", channel)),
                                "starttime": str(starttime),
                                "endtime": str(endtime),
                                "start_epoch": float(start_epoch),
                                "end_epoch": float(end_epoch),
                                "sampling_rate": float(get_attr(ds, "sampling_rate", np.nan)),
                                "delta": float(get_attr(ds, "delta", np.nan)),
                                "npts": int(get_attr(ds, "npts", ds.shape[0])),
                                "dtype": str(get_attr(ds, "dtype", str(ds.dtype))),
                                "source_file": str(get_attr(ds, "mseed_source_file", "")),
                                "latitude": float(get_attr(ds, "latitude", np.nan)),
                                "longitude": float(get_attr(ds, "longitude", np.nan)),
                                "elevation": float(get_attr(ds, "elevation", np.nan)),
                                "location_available": int(bool(get_attr(ds, "location_available", False))),
                            }


def insert_segment_records(conn, file_id, records, batch_size=5000):
    cur = conn.cursor()

    sql = """
    INSERT OR IGNORE INTO waveform_segments (
        file_id,
        h5_file,
        dataset_path,
        year_id,
        day_id,
        station_id,
        station_key,
        network,
        station,
        location,
        channel,
        starttime,
        endtime,
        start_epoch,
        end_epoch,
        sampling_rate,
        delta,
        npts,
        dtype,
        source_file,
        latitude,
        longitude,
        elevation,
        location_available
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    batch = []

    for rec in records:
        batch.append(
            (
                file_id,
                rec["h5_file"],
                rec["dataset_path"],
                rec["year_id"],
                rec["day_id"],
                rec["station_id"],
                rec["station_key"],
                rec["network"],
                rec["station"],
                rec["location"],
                rec["channel"],
                rec["starttime"],
                rec["endtime"],
                rec["start_epoch"],
                rec["end_epoch"],
                rec["sampling_rate"],
                rec["delta"],
                rec["npts"],
                rec["dtype"],
                rec["source_file"],
                rec["latitude"],
                rec["longitude"],
                rec["elevation"],
                rec["location_available"],
            )
        )

        if len(batch) >= batch_size:
            cur.executemany(sql, batch)
            conn.commit()
            batch.clear()

    if batch:
        cur.executemany(sql, batch)
        conn.commit()


def update_station_table(conn):
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO stations (
            station_key,
            network,
            station,
            latitude,
            longitude,
            elevation,
            location_available
        )
        SELECT
            station_key,
            network,
            station,
            AVG(latitude),
            AVG(longitude),
            AVG(elevation),
            MAX(location_available)
        FROM waveform_segments
        GROUP BY station_key, network, station
        """
    )

    conn.commit()


def build_index(h5_input, db_file, reset=True, default_location=DEFAULT_LOCATION):
    h5_files = resolve_h5_files(h5_input)

    conn = connect_db(db_file)
    init_db(conn, reset=reset)

    print(f"[INFO] HDF5 files: {len(h5_files)}")
    print(f"[INFO] SQLite DB: {db_file}")

    total = 0

    for i, h5_file in enumerate(h5_files, start=1):
        print(f"[INFO] Indexing {i}/{len(h5_files)}: {h5_file}")

        file_id = get_or_insert_file_id(conn, h5_file)

        count = 0

        def record_generator():
            nonlocal count
            for rec in iter_waveform_datasets(h5_file, default_location=default_location):
                count += 1
                yield rec

        insert_segment_records(conn, file_id, record_generator())
        total += count

        print(f"[INFO]   segments indexed: {count}")

    update_station_table(conn)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM waveform_segments")
    n_segments = cur.fetchone()["n"]

    cur.execute("SELECT COUNT(*) AS n FROM stations")
    n_stations = cur.fetchone()["n"]

    conn.close()

    print(f"[OK] Indexed segments: {n_segments}")
    print(f"[OK] Indexed stations: {n_stations}")


# -----------------------------
# Querying
# -----------------------------

def query_segments(
    db_file,
    network=None,
    station=None,
    station_key=None,
    location=None,
    channels=None,
    starttime=None,
    endtime=None,
    limit=None,
):
    start_epoch = parse_time_to_epoch(starttime) if starttime else None
    end_epoch = parse_time_to_epoch(endtime) if endtime else None

    if start_epoch is None:
        start_epoch = -1.0e30
    if end_epoch is None:
        end_epoch = 1.0e30

    channels = comma_list(channels) if isinstance(channels, str) else channels

    clauses = [
        "end_epoch >= ?",
        "start_epoch <= ?",
    ]
    params = [start_epoch, end_epoch]

    if station_key:
        clauses.append("station_key = ?")
        params.append(station_key)
    else:
        if network:
            clauses.append("network = ?")
            params.append(network)
        if station:
            clauses.append("station = ?")
            params.append(station)

    if location:
        clauses.append("location = ?")
        params.append(location)

    if channels:
        placeholders = ",".join(["?"] * len(channels))
        clauses.append(f"channel IN ({placeholders})")
        params.extend(channels)

    sql = f"""
    SELECT *
    FROM waveform_segments
    WHERE {' AND '.join(clauses)}
    ORDER BY network, station, location, channel, start_epoch
    """

    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))

    conn = connect_db(db_file)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    return rows


def print_query_results(rows, as_json=False):
    if as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    print(f"[OK] Matched segments: {len(rows)}")

    for row in rows:
        print(
            f"{row['network']}.{row['station']}.{row['location']} "
            f"{row['channel']} "
            f"{row['starttime']} -> {row['endtime']} "
            f"npts={row['npts']} sr={row['sampling_rate']} "
            f"path={row['dataset_path']} "
            f"h5={row['h5_file']}"
        )


def read_segment(row):
    with h5py.File(row["h5_file"], "r") as h5:
        ds = h5[row["dataset_path"]]
        data = ds[()]
    return data


def read_query_results(rows):
    out = []

    for row in rows:
        data = read_segment(row)
        item = dict(row)
        item["data_shape"] = tuple(data.shape)
        item["data"] = data
        out.append(item)

    return out


def trim_array_by_time(data, row, query_starttime=None, query_endtime=None):
    """
    Optional trimming by query time.

    This assumes regular sampling and uses segment-level starttime/sampling_rate.
    """
    if query_starttime is None and query_endtime is None:
        return data

    sr = float(row["sampling_rate"])
    seg_start = float(row["start_epoch"])
    seg_end = float(row["end_epoch"])

    q0 = parse_time_to_epoch(query_starttime) if query_starttime else seg_start
    q1 = parse_time_to_epoch(query_endtime) if query_endtime else seg_end

    q0 = max(q0, seg_start)
    q1 = min(q1, seg_end)

    if q1 < q0:
        return data[:0]

    i0 = int(round((q0 - seg_start) * sr))
    i1 = int(round((q1 - seg_start) * sr)) + 1

    i0 = max(i0, 0)
    i1 = min(i1, len(data))

    return data[i0:i1]


def read_query_results_trimmed(rows, query_starttime=None, query_endtime=None):
    out = []

    for row in rows:
        data = read_segment(row)
        data = trim_array_by_time(
            data,
            row,
            query_starttime=query_starttime,
            query_endtime=query_endtime,
        )

        item = dict(row)
        item["data_shape"] = tuple(data.shape)
        item["data"] = data
        out.append(item)

    return out


# -----------------------------
# CLI
# -----------------------------

def cmd_build(args):
    build_index(
        h5_input=args.h5,
        db_file=args.db,
        reset=not args.no_reset,
        default_location=args.default_location,
    )


def cmd_query(args):
    rows = query_segments(
        db_file=args.db,
        network=args.network,
        station=args.station,
        station_key=args.station_key,
        location=args.location,
        channels=args.channels,
        starttime=args.starttime,
        endtime=args.endtime,
        limit=args.limit,
    )
    print_query_results(rows, as_json=args.json)


def cmd_read(args):
    rows = query_segments(
        db_file=args.db,
        network=args.network,
        station=args.station,
        station_key=args.station_key,
        location=args.location,
        channels=args.channels,
        starttime=args.starttime,
        endtime=args.endtime,
        limit=args.limit,
    )

    items = read_query_results_trimmed(
        rows,
        query_starttime=args.starttime,
        query_endtime=args.endtime,
    )

    print(f"[OK] Loaded segments: {len(items)}")
    for item in items:
        print(
            f"{item['network']}.{item['station']}.{item['location']} "
            f"{item['channel']} "
            f"{item['starttime']} -> {item['endtime']} "
            f"trimmed_shape={item['data_shape']} "
            f"path={item['dataset_path']}"
        )

    if args.output_npz:
        arrays = {}
        meta = []

        for i, item in enumerate(items):
            key = f"arr_{i:06d}"
            arrays[key] = item["data"]

            meta_item = dict(item)
            meta_item.pop("data", None)
            meta.append(meta_item)

        arrays["metadata_json"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez(args.output_npz, **arrays)
        print(f"[OK] Saved NPZ: {args.output_npz}")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="SQLite index tool for hierarchical HDF5 continuous waveform datasets."
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build SQLite index from HDF5 files.")
    p_build.add_argument(
        "--h5",
        required=True,
        help="HDF5 input: single file, directory, glob pattern, or file path.",
    )
    p_build.add_argument(
        "--db",
        required=True,
        help="Output SQLite database file.",
    )
    p_build.add_argument(
        "--no_reset",
        action="store_true",
        help="Do not reset existing tables; append new records with INSERT OR IGNORE.",
    )
    p_build.add_argument(
        "--default_location",
        default=DEFAULT_LOCATION,
        help='Default location code. Default: "--".',
    )
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="Query indexed waveform segments.")
    p_query.add_argument("--db", required=True, help="SQLite database file.")
    p_query.add_argument("--network", default=None, help="Network code, e.g. BK.")
    p_query.add_argument("--station", default=None, help="Station code, e.g. BDM.")
    p_query.add_argument("--station_key", default=None, help="Station key, e.g. BK.BDM.")
    p_query.add_argument("--location", default=None, help="Location code. Optional.")
    p_query.add_argument("--channels", default=None, help="Comma-separated channels, e.g. BHE,BHN,BHZ.")
    p_query.add_argument("--starttime", default=None, help="Query start time.")
    p_query.add_argument("--endtime", default=None, help="Query end time.")
    p_query.add_argument("--limit", type=int, default=None, help="Limit number of results.")
    p_query.add_argument("--json", action="store_true", help="Print results as JSON.")
    p_query.set_defaults(func=cmd_query)

    p_read = sub.add_parser("read", help="Query and read waveform data.")
    p_read.add_argument("--db", required=True, help="SQLite database file.")
    p_read.add_argument("--network", default=None, help="Network code, e.g. BK.")
    p_read.add_argument("--station", default=None, help="Station code, e.g. BDM.")
    p_read.add_argument("--station_key", default=None, help="Station key, e.g. BK.BDM.")
    p_read.add_argument("--location", default=None, help="Location code. Optional.")
    p_read.add_argument("--channels", default=None, help="Comma-separated channels, e.g. BHE,BHN,BHZ.")
    p_read.add_argument("--starttime", default=None, help="Query start time.")
    p_read.add_argument("--endtime", default=None, help="Query end time.")
    p_read.add_argument("--limit", type=int, default=None, help="Limit number of results.")
    p_read.add_argument("--output_npz", default=None, help="Optional output NPZ file.")
    p_read.set_defaults(func=cmd_read)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()