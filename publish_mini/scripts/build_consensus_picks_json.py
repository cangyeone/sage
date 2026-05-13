#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build consensus neural-network phase picks from multiple JSONL pick files.

功能
----
1. 不修改原人工标注 JSON 和原模型 JSONL。
2. 可一次性把多个模型 JSONL 建成一个 SQLite 数据库，方便检索。
3. 按 P/S 归一化后，在 is_phase 秒窗口内合并多个模型拾取。
4. 至少 min_models 个不同模型检测到，才保留为 consensus phase。
5. 输出 JSON 结构：years -> days -> stations -> neural_picks。
6. 保留：
   - consensus mean/std
   - model 文件名索引
   - 每个模型原始震相类型
   - 每个模型相对于 consensus mean 的 residual
   - phase_prob/snr/polarity 等字段
   - 匹配到的人工标注 manual/automatic
   - event_id

Example
-------
python build_consensus_picks_json.py \
  --auto-jsonl \
    data/picks/phasenet.jsonl \
    data/picks/eqtransformer.jsonl \
    data/picks/skynet.jsonl \
  --label-json data/label/annotations_for_continuous_hdf5.json \
  --index-db data/picks/consensus_models.sqlite \
  --out-json data/label/consensus_nn_picks.json \
  --build-index \
  --drop-existing \
  --is-phase 1.5 \
  --min-model-fraction 0.5 \
  --human-match 1.5 \
  --phase-map "P:Pg,P,Pn;S:Sg,S,Sn"
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# Basic helpers
# ============================================================

def parse_utc_to_epoch_seconds(value: str) -> float:
    if value is None:
        raise ValueError("time value is None")
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.timestamp()


def epoch_to_utc_string(t: float) -> str:
    return (
        datetime.fromtimestamp(float(t), tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def norm_location(loc: Optional[str]) -> str:
    if loc is None or loc == "":
        return "--"
    return str(loc)


def norm_station_id(
    station_id: Optional[str],
    network: Optional[str] = None,
    station: Optional[str] = None,
    location: Optional[str] = None,
) -> str:
    if station_id:
        parts = str(station_id).split(".")
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}.{norm_location(parts[2])}"
        return str(station_id)
    return f"{network}.{station}.{norm_location(location)}"


def split_year_day(t_epoch: float) -> Tuple[str, str]:
    dt = datetime.fromtimestamp(float(t_epoch), tz=timezone.utc)
    return f"{dt.year:04d}", dt.strftime("%Y-%m-%d")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip bad JSONL line {path}:{line_no}: {exc}", file=sys.stderr)


def parse_phase_map(text: Optional[str]) -> Dict[str, str]:
    """
    Input:
        "P:Pg,P,Pn;S:Sg,S,Sn"

    Output:
        {
          "Pg": "P", "P": "P", "Pn": "P",
          "Sg": "S", "S": "S", "Sn": "S"
        }
    """
    if not text:
        text = "P:Pg,P,Pn;S:Sg,S,Sn"

    out: Dict[str, str] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        canonical, aliases = item.split(":", 1)
        canonical = canonical.strip()
        out[canonical] = canonical
        for ph in aliases.split(","):
            ph = ph.strip()
            if ph:
                out[ph] = canonical
    return out


# ============================================================
# SQLite database for multiple model JSONL files
# ============================================================

def connect_db(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn


def init_multi_model_db(conn: sqlite3.Connection, drop_existing: bool = False) -> None:
    cur = conn.cursor()

    if drop_existing:
        cur.execute("DROP TABLE IF EXISTS picks")
        cur.execute("DROP TABLE IF EXISTS models")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS models (
            model_index INTEGER PRIMARY KEY,
            model_name TEXT NOT NULL,
            model_path TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_index INTEGER NOT NULL,
            station_id TEXT NOT NULL,
            network TEXT,
            station TEXT,
            location TEXT,
            phase_original TEXT NOT NULL,
            phase_canonical TEXT NOT NULL,
            time_epoch REAL NOT NULL,
            sec_key INTEGER NOT NULL,
            phase_prob REAL,
            polarity TEXT,
            polarity_prob REAL,
            snr REAL,
            amplitude REAL,
            h5_file TEXT,
            raw_json TEXT,
            FOREIGN KEY(model_index) REFERENCES models(model_index)
        )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_station_phase_time ON picks(station_id, phase_canonical, time_epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_station_phase_sec ON picks(station_id, phase_canonical, sec_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_time ON picks(time_epoch)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_picks_model ON picks(model_index)")
    conn.commit()


def build_multi_model_index(
    auto_jsonl_files: Sequence[Path],
    db_path: Path,
    phase_alias_to_canonical: Dict[str, str],
    batch_size: int = 50000,
    drop_existing: bool = False,
    keep_raw_json: bool = False,
    min_prob: Optional[float] = None,
    progress_every: int = 200000,
) -> None:
    conn = connect_db(db_path)
    init_multi_model_db(conn, drop_existing=drop_existing)
    cur = conn.cursor()

    insert_model_sql = """
        INSERT OR REPLACE INTO models(model_index, model_name, model_path)
        VALUES (?, ?, ?)
    """

    insert_pick_sql = """
        INSERT INTO picks (
            model_index, station_id, network, station, location,
            phase_original, phase_canonical, time_epoch, sec_key,
            phase_prob, polarity, polarity_prob, snr, amplitude, h5_file, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for model_i, path0 in enumerate(auto_jsonl_files):
        path = Path(path0).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"auto JSONL not found: {path}")

        cur.execute(insert_model_sql, (model_i, path.name, str(path)))
        conn.commit()

        print(f"[INDEX] model {model_i}: {path}")
        batch = []
        inserted = 0
        skipped = 0
        scanned = 0

        for rec in iter_jsonl(path):
            scanned += 1
            if rec.get("record_type") != "phase_pick":
                continue

            try:
                phase_original = str(rec.get("phase_name"))
                if phase_original not in phase_alias_to_canonical:
                    skipped += 1
                    continue

                phase_prob = rec.get("phase_prob")
                if min_prob is not None and phase_prob is not None:
                    if float(phase_prob) < float(min_prob):
                        skipped += 1
                        continue

                phase_canonical = phase_alias_to_canonical[phase_original]

                station_info = rec.get("station_info") or {}
                station_id = norm_station_id(
                    station_info.get("station_id"),
                    station_info.get("network"),
                    station_info.get("station"),
                    station_info.get("location"),
                )

                t = parse_utc_to_epoch_seconds(rec.get("phase_time"))
                sec_key = int(math.floor(t))
                raw_json = json.dumps(rec, ensure_ascii=False) if keep_raw_json else None

                batch.append((
                    model_i,
                    station_id,
                    station_info.get("network"),
                    station_info.get("station"),
                    norm_location(station_info.get("location")),
                    phase_original,
                    phase_canonical,
                    t,
                    sec_key,
                    phase_prob,
                    rec.get("polarity"),
                    rec.get("polarity_prob"),
                    rec.get("snr"),
                    rec.get("amplitude"),
                    rec.get("h5_file"),
                    raw_json,
                ))

            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"[WARN] skip record in {path.name}: {exc}", file=sys.stderr)

            if len(batch) >= batch_size:
                cur.executemany(insert_pick_sql, batch)
                conn.commit()
                inserted += len(batch)
                batch.clear()
                if inserted % progress_every < batch_size:
                    print(f"        inserted={inserted:,}, skipped={skipped:,}, scanned={scanned:,}")

        if batch:
            cur.executemany(insert_pick_sql, batch)
            conn.commit()
            inserted += len(batch)

        print(f"        done inserted={inserted:,}, skipped={skipped:,}, scanned={scanned:,}")

    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    print(f"[INDEX] SQLite saved: {db_path}")


# ============================================================
# Label index
# ============================================================

@dataclass
class LabelPick:
    label_phase_original: str
    label_phase_canonical: str
    label_time_epoch: float
    station_id: str
    status: str
    event_id: Optional[str]
    distance_km: Optional[float]
    raw: Dict[str, Any]


def iter_label_picks(
    label_json: Path,
    phase_alias_to_canonical: Dict[str, str],
) -> Iterable[LabelPick]:
    with Path(label_json).open("r", encoding="utf-8") as f:
        data = json.load(f)

    years = data.get("years", {})
    for _year_id, year_obj in years.items():
        for _day_id, day_obj in (year_obj.get("days") or {}).items():
            for event_id, event_obj in (day_obj.get("events") or {}).items():
                for station_id0, sta_obj in (event_obj.get("stations") or {}).items():
                    for p in (sta_obj.get("picks") or []):
                        try:
                            ph0 = str(p.get("phase"))
                            if ph0 not in phase_alias_to_canonical:
                                continue

                            station_id = norm_station_id(
                                p.get("station_id") or station_id0,
                                p.get("network"),
                                p.get("station"),
                                p.get("location"),
                            )

                            yield LabelPick(
                                label_phase_original=ph0,
                                label_phase_canonical=phase_alias_to_canonical[ph0],
                                label_time_epoch=parse_utc_to_epoch_seconds(p.get("time")),
                                station_id=station_id,
                                status=str(p.get("status", "unknown")),
                                event_id=p.get("event_id") or event_id,
                                distance_km=p.get("distance_km"),
                                raw=p,
                            )
                        except Exception as exc:
                            print(
                                f"[WARN] skip label pick event={event_id}, station={station_id0}: {exc}",
                                file=sys.stderr,
                            )


def load_label_index(
    label_json: Path,
    phase_alias_to_canonical: Dict[str, str],
) -> Dict[Tuple[str, str], List[LabelPick]]:
    index: Dict[Tuple[str, str], List[LabelPick]] = defaultdict(list)

    n = 0
    for lab in iter_label_picks(label_json, phase_alias_to_canonical):
        index[(lab.station_id, lab.label_phase_canonical)].append(lab)
        n += 1

    for key in index:
        index[key].sort(key=lambda x: x.label_time_epoch)

    print(f"[LABEL] loaded {n:,} label picks, station-phase groups={len(index):,}")
    return index


def find_matching_labels(
    label_index: Dict[Tuple[str, str], List[LabelPick]],
    station_id: str,
    phase_canonical: str,
    mean_epoch: float,
    human_match_s: float,
) -> List[Dict[str, Any]]:
    labs = label_index.get((station_id, phase_canonical), [])
    if not labs:
        return []

    matched = []
    lo = mean_epoch - human_match_s
    hi = mean_epoch + human_match_s

    # 简单线性扫描足够；如果标签极大，可再改 bisect。
    for lab in labs:
        if lab.label_time_epoch < lo:
            continue
        if lab.label_time_epoch > hi:
            break

        residual = lab.label_time_epoch - mean_epoch
        item = {
            "phase": lab.label_phase_canonical,
            "phase_original": lab.label_phase_original,
            "time": epoch_to_utc_string(lab.label_time_epoch),
            "time_epoch": lab.label_time_epoch,
            "residual_to_consensus_s": residual,
            "status": lab.status,
            "event_id": lab.event_id,
            "distance_km": lab.distance_km,
        }
        matched.append(item)

    matched.sort(key=lambda x: abs(x["residual_to_consensus_s"]))
    return matched


# ============================================================
# Consensus construction
# ============================================================

def get_model_index(conn: sqlite3.Connection) -> Dict[str, str]:
    rows = conn.execute(
        "SELECT model_index, model_name FROM models ORDER BY model_index"
    ).fetchall()
    return {str(i): name for i, name in rows}


def iter_station_phase_groups(conn: sqlite3.Connection) -> Iterable[Tuple[str, str]]:
    sql = """
        SELECT DISTINCT station_id, phase_canonical
        FROM picks
        ORDER BY station_id, phase_canonical
    """
    for station_id, phase in conn.execute(sql):
        yield str(station_id), str(phase)


def load_group_picks(
    conn: sqlite3.Connection,
    station_id: str,
    phase_canonical: str,
) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            id, model_index, station_id,
            phase_original, phase_canonical,
            time_epoch, phase_prob,
            polarity, polarity_prob, snr, amplitude, h5_file
        FROM picks
        WHERE station_id = ?
          AND phase_canonical = ?
        ORDER BY time_epoch
    """
    rows = conn.execute(sql, (station_id, phase_canonical)).fetchall()

    out = []
    for r in rows:
        out.append({
            "id": int(r[0]),
            "model_index": int(r[1]),
            "station_id": str(r[2]),
            "phase_original": str(r[3]),
            "phase_canonical": str(r[4]),
            "time_epoch": float(r[5]),
            "phase_prob": r[6],
            "polarity": r[7],
            "polarity_prob": r[8],
            "snr": r[9],
            "amplitude": r[10],
            "h5_file": r[11],
        })
    return out


def choose_one_pick_per_model(
    picks: List[Dict[str, Any]],
    center_epoch: float,
) -> List[Dict[str, Any]]:
    """
    同一个模型在同一窗口内可能有多个 pick。
    consensus 统计时每个模型只保留离当前中心最近的一个，避免单模型重复刷票。
    """
    best: Dict[int, Dict[str, Any]] = {}
    for p in picks:
        mi = int(p["model_index"])
        if mi not in best:
            best[mi] = p
        else:
            old = best[mi]
            if abs(p["time_epoch"] - center_epoch) < abs(old["time_epoch"] - center_epoch):
                best[mi] = p
    return sorted(best.values(), key=lambda x: x["time_epoch"])


def build_consensus_for_group(
    picks: List[Dict[str, Any]],
    is_phase_s: float,
    min_models: int,
) -> List[Dict[str, Any]]:
    """
    在单个 station + P/S phase 内做时间聚类。

    算法：
    - picks 已按 time_epoch 排序。
    - 滑动窗口收集 is_phase_s 内的 pick。
    - 每个模型只投一票。
    - 如果不同模型数 >= min_models，则形成一个 consensus。
    - 为避免重复，形成 consensus 后跳过这个窗口覆盖的 picks。
    """
    if not picks:
        return []

    n = len(picks)
    out = []
    i = 0

    while i < n:
        t0 = picks[i]["time_epoch"]

        j = i
        window = []
        while j < n and picks[j]["time_epoch"] - t0 <= is_phase_s:
            window.append(picks[j])
            j += 1

        # 初始中心用窗口均值，再每个模型取离中心最近的一个
        center0 = sum(p["time_epoch"] for p in window) / max(len(window), 1)
        unique = choose_one_pick_per_model(window, center0)

        if len({p["model_index"] for p in unique}) >= min_models:
            times = [p["time_epoch"] for p in unique]
            mean_epoch = sum(times) / len(times)
            if len(times) >= 2:
                std_s = math.sqrt(sum((x - mean_epoch) ** 2 for x in times) / (len(times) - 1))
            else:
                std_s = 0.0

            # 用更新后的 mean 再筛一遍，保证所有参与模型都在 is_phase 内
            unique2 = [
                p for p in unique
                if abs(p["time_epoch"] - mean_epoch) <= is_phase_s
            ]

            if len({p["model_index"] for p in unique2}) >= min_models:
                times = [p["time_epoch"] for p in unique2]
                mean_epoch = sum(times) / len(times)
                std_s = (
                    math.sqrt(sum((x - mean_epoch) ** 2 for x in times) / (len(times) - 1))
                    if len(times) >= 2 else 0.0
                )

                model_picks = []
                for p in unique2:
                    model_picks.append({
                        "model_index": int(p["model_index"]),
                        "phase_original": p["phase_original"],
                        "phase": p["phase_canonical"],
                        "time": epoch_to_utc_string(p["time_epoch"]),
                        "time_epoch": p["time_epoch"],
                        "residual_to_mean_s": p["time_epoch"] - mean_epoch,
                        "phase_prob": p["phase_prob"],
                        "polarity": p["polarity"],
                        "polarity_prob": p["polarity_prob"],
                        "snr": p["snr"],
                        "amplitude": p["amplitude"],
                        "h5_file": p["h5_file"],
                        "pick_db_id": p["id"],
                    })

                out.append({
                    "phase": unique2[0]["phase_canonical"],
                    "mean_time": epoch_to_utc_string(mean_epoch),
                    "mean_epoch": mean_epoch,
                    "std_s": std_s,
                    "n_models": len({p["model_index"] for p in unique2}),
                    "model_indices": sorted({int(p["model_index"]) for p in unique2}),
                    "model_picks": model_picks,
                })

                # 跳过已经被该 consensus 覆盖的 picks，避免重复输出同一震相
                end_time = mean_epoch + is_phase_s
                while i < n and picks[i]["time_epoch"] <= end_time:
                    i += 1
                continue

        i += 1

    return out


def insert_nested_output(
    output: Dict[str, Any],
    station_id: str,
    pick: Dict[str, Any],
) -> None:
    year, day = split_year_day(pick["mean_epoch"])

    years = output.setdefault("years", {})
    year_obj = years.setdefault(year, {"days": {}})
    day_obj = year_obj["days"].setdefault(day, {"stations": {}})
    sta_obj = day_obj["stations"].setdefault(station_id, {"neural_picks": []})

    sta_obj["neural_picks"].append(pick)


def build_consensus_json(
    db_path: Path,
    label_json: Optional[Path],
    out_json: Path,
    phase_alias_to_canonical: Dict[str, str],
    is_phase_s: float = 1.5,
    min_models: int = 3,
    min_model_fraction: Optional[float] = None,
    human_match_s: float = 1.5,
) -> Dict[str, Any]:
    conn = connect_db(db_path)

    model_index = get_model_index(conn)
    if not model_index:
        raise RuntimeError("No models found in SQLite. Run with --build-index first.")

    if min_model_fraction is not None:
        if not (0 < float(min_model_fraction) <= 1):
            raise ValueError("--min-model-fraction must be in the interval (0, 1].")
        min_models = max(1, math.ceil(float(min_model_fraction) * len(model_index)))

    label_index = {}
    if label_json is not None:
        label_index = load_label_index(label_json, phase_alias_to_canonical)

    output: Dict[str, Any] = {
        "format": "consensus_neural_phase_picks_v1",
        "model_index": model_index,
        "params": {
            "is_phase_s": is_phase_s,
            "min_models": min_models,
            "min_model_fraction": min_model_fraction,
            "human_match_s": human_match_s,
            "phase_map": phase_alias_to_canonical,
        },
        "years": {},
    }

    total_consensus = 0

    for k, (station_id, phase) in enumerate(iter_station_phase_groups(conn), 1):
        if k % 1000 == 0:
            print(f"[CONSENSUS] processed station-phase groups: {k:,}, consensus={total_consensus:,}")

        picks = load_group_picks(conn, station_id, phase)
        consensus_items = build_consensus_for_group(
            picks=picks,
            is_phase_s=is_phase_s,
            min_models=min_models,
        )

        for item in consensus_items:
            human_matches = []
            event_ids = []

            if label_index:
                human_matches = find_matching_labels(
                    label_index=label_index,
                    station_id=station_id,
                    phase_canonical=item["phase"],
                    mean_epoch=item["mean_epoch"],
                    human_match_s=human_match_s,
                )
                event_ids = sorted({
                    str(x["event_id"])
                    for x in human_matches
                    if x.get("event_id") is not None
                })

            item["human_matches"] = human_matches
            item["event_ids"] = event_ids
            item["matched_human"] = bool(human_matches)

            insert_nested_output(output, station_id, item)
            total_consensus += 1

    conn.close()

    output["summary"] = {
        "n_consensus_picks": total_consensus,
        "n_models": len(model_index),
    }

    out_json = Path(out_json).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[DONE] consensus picks={total_consensus:,}")
    print(f"[DONE] saved JSON: {out_json}")
    return output


def print_db_info(db_path: Path) -> None:
    conn = connect_db(db_path)

    n_model = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    n_pick = conn.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
    print(f"models: {n_model:,}")
    print(f"picks:  {n_pick:,}")

    print("\nmodel index:")
    for idx, name, path in conn.execute("SELECT model_index, model_name, model_path FROM models ORDER BY model_index"):
        print(f"  {idx}: {name}  ({path})")

    print("\nphase counts:")
    for phase_canonical, phase_original, c in conn.execute("""
        SELECT phase_canonical, phase_original, COUNT(*)
        FROM picks
        GROUP BY phase_canonical, phase_original
        ORDER BY phase_canonical, COUNT(*) DESC
    """):
        print(f"  {phase_canonical:2s} / {phase_original:4s}: {c:,}")

    conn.close()


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build consensus neural-network phase-pick JSON from multiple model JSONL files."
    )

    parser.add_argument(
        "--auto-jsonl",
        type=Path,
        nargs="+",
        required=True,
        help="Multiple model JSONL pick files.",
    )
    parser.add_argument(
        "--label-json",
        type=Path,
        default=None,
        help="Original annotation JSON. Used only for matching human labels; not modified.",
    )
    parser.add_argument(
        "--index-db",
        type=Path,
        default=Path("data/picks/consensus_models.sqlite"),
        help="SQLite database for all model picks.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/label/consensus_nn_picks.json"),
        help="Output consensus JSON.",
    )

    parser.add_argument("--build-index", action="store_true", help="Build SQLite index from all JSONL files.")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing SQLite tables before building.")
    parser.add_argument("--keep-raw-json", action="store_true", help="Store raw JSON in SQLite. Not recommended for very large files.")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--min-prob", type=float, default=None, help="Optional minimum phase_prob for model picks.")

    parser.add_argument("--is-phase", type=float, default=1.5, help="Consensus time window in seconds.")
    parser.add_argument("--min-models", type=int, default=3, help="Minimum number of different models required.")
    parser.add_argument(
        "--min-model-fraction",
        type=float,
        default=None,
        help="Optional fraction of indexed models required for consensus; overrides --min-models after rounding up.",
    )
    parser.add_argument("--human-match", type=float, default=1.5, help="Human-label matching window around consensus mean.")
    parser.add_argument(
        "--phase-map",
        type=str,
        default="P:Pg,P,Pn;S:Sg,S,Sn",
        help='Canonical phase mapping. Example: "P:Pg,P,Pn;S:Sg,S,Sn".',
    )

    parser.add_argument("--db-info", action="store_true", help="Print SQLite DB info and exit.")
    args = parser.parse_args()

    phase_alias_to_canonical = parse_phase_map(args.phase_map)

    if args.build_index:
        build_multi_model_index(
            auto_jsonl_files=args.auto_jsonl,
            db_path=args.index_db,
            phase_alias_to_canonical=phase_alias_to_canonical,
            batch_size=args.batch_size,
            drop_existing=args.drop_existing,
            keep_raw_json=args.keep_raw_json,
            min_prob=args.min_prob,
        )

    if args.db_info:
        print_db_info(args.index_db)
        return

    build_consensus_json(
        db_path=args.index_db,
        label_json=args.label_json,
        out_json=args.out_json,
        phase_alias_to_canonical=phase_alias_to_canonical,
        is_phase_s=args.is_phase,
        min_models=args.min_models,
        min_model_fraction=args.min_model_fraction,
        human_match_s=args.human_match,
    )


if __name__ == "__main__":
    main()
