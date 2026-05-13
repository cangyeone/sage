#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python example_query_waveform.py \
  --db data/index/waveform_index.sqlite \
  --network BK \
  --station BDM \
  --channel BHZ \
  --starttime 2019-07-06T04:00:00 \
  --endtime 2019-07-06T04:01:00
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.waveform_index_api import query_and_merge, save_merged_to_npz


def main():
    parser = argparse.ArgumentParser(
        description="Example API client for querying and merging waveform data."
    )

    parser.add_argument(
        "--db",
        required=True,
        help="SQLite waveform index database, e.g. data/index/waveform_index.sqlite",
    )

    parser.add_argument(
        "--network",
        default="*",
        help='Network code or wildcard, e.g. "BK", "C*", "*".',
    )

    parser.add_argument(
        "--station",
        default="*",
        help='Station code or wildcard, e.g. "BDM", "BD*", "*".',
    )

    parser.add_argument(
        "--location",
        default="*",
        help='Location code or wildcard, e.g. "00", "--", "*".',
    )

    parser.add_argument(
        "--channel",
        default="*",
        help='Channel code or wildcard, e.g. "BHE", "BH*", "*Z", "*".',
    )

    parser.add_argument(
        "--starttime",
        required=True,
        help="Query start time, e.g. 2019-07-01T00:00:00",
    )

    parser.add_argument(
        "--endtime",
        required=True,
        help="Query end time, e.g. 2019-07-01T01:00:00",
    )

    parser.add_argument(
        "--fill_value",
        type=float,
        default=0.0,
        help="Fill value for gaps.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for matched segments.",
    )

    parser.add_argument(
        "--output_npz",
        default=None,
        help="Optional output NPZ file.",
    )

    parser.add_argument(
        "--show_segments",
        action="store_true",
        help="Print segment details.",
    )

    args = parser.parse_args()

    merged, rows = query_and_merge(
        db_file=args.db,
        network=args.network,
        station=args.station,
        location=args.location,
        channel=args.channel,
        starttime=args.starttime,
        endtime=args.endtime,
        fill_value=args.fill_value,
        dtype=np.float32,
        limit=args.limit,
    )

    print("=" * 80)
    print("Query")
    print("=" * 80)
    print(f"network   : {args.network}")
    print(f"station   : {args.station}")
    print(f"location  : {args.location}")
    print(f"channel   : {args.channel}")
    print(f"starttime : {args.starttime}")
    print(f"endtime   : {args.endtime}")

    print()
    print("=" * 80)
    print("Matched segments")
    print("=" * 80)
    print(f"Segment count: {len(rows)}")

    if args.show_segments:
        for row in rows:
            print(
                f"{row['network']}.{row['station']}.{row['location']} "
                f"{row['channel']} "
                f"{row['starttime']} -> {row['endtime']} "
                f"sr={row['sampling_rate']} "
                f"npts={row['npts']} "
                f"path={row['dataset_path']} "
                f"h5={row['h5_file']}"
            )

    print()
    print("=" * 80)
    print("Merged waveforms")
    print("=" * 80)

    if len(merged) == 0:
        print("No waveform found.")
        return

    for key, item in merged.items():
        print(
            f"{key}: "
            f"shape={item['data'].shape}, "
            f"sr={item['sampling_rate']}, "
            f"filled_ratio={item['filled_ratio']:.4f}, "
            f"segments={len(item['segments'])}"
        )

    if args.output_npz:
        save_merged_to_npz(merged, args.output_npz)
        print()
        print(f"[OK] Saved merged waveform NPZ: {args.output_npz}")


if __name__ == "__main__":
    main()
