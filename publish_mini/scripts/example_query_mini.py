#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query a few waveform segments from the mini SQLite index."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--starttime", default="2019-07-06T04:00:00")
    parser.add_argument("--endtime", default="2019-07-06T04:05:00")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    db_path = args.root / "data" / "index" / "waveform_index.sqlite"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT h5_file, dataset_path, network, station, location, channel,
               starttime, endtime, sampling_rate, npts
        FROM waveform_segments
        WHERE endtime >= ? AND starttime <= ?
        ORDER BY network, station, location, channel
        LIMIT ?
        """,
        (args.starttime, args.endtime, args.limit),
    ).fetchall()
    conn.close()

    for row in rows:
        item = dict(row)
        print(
            f"{item['network']}.{item['station']}.{item['location']} "
            f"{item['channel']} {item['starttime']} -> {item['endtime']} "
            f"sr={item['sampling_rate']} npts={item['npts']} "
            f"path={item['dataset_path']}"
        )


if __name__ == "__main__":
    main()
