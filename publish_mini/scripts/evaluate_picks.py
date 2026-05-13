#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate PhaseNet/AI picks against continuous-HDF5 annotation JSON.

Main features
-------------
1. Stream-read huge JSONL auto-pick files, e.g. >40 GB.
2. Build a fast per-station / per-phase / per-second SQLite index for auto picks.
3. Compare human/label picks with auto picks using a configurable TP tolerance.
4. Report P/S recall where labels P/S map to auto Pg/Sg by default.
5. Report travel-time residual distribution within a wider error window.
6. Separate statistics for manual labels only and all labels.
7. Count automatic picks in the SQLite index.
8. Fit Gaussian and Student-t residual models.
9. Plot residual histograms with fitted Gaussian and Student-t PDFs.

Typical usage
-------------
python evaluate_phasenet_picks.py \
  --auto-jsonl data/picks/phasenet.pick.jsonl \
  --label-json data/label/annotations_for_continuous_hdf5.json \
  --index-db data/picks/phasenet.pick.index.sqlite \
  --outdir eval_phasenet \
  --build-index \
  --tp-tol 1.5 \
  --err-window 5.0

If the SQLite index already exists, omit --build-index.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import bisect

import numpy as np

# matplotlib is only needed for --plot. Keep import lazy in plot_results().

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


# -----------------------------
# Time and phase normalization
# -----------------------------

def parse_utc_to_epoch_seconds(value: str) -> float:
    """Parse ISO UTC string to epoch seconds.

    Supports strings with or without trailing Z. Naive timestamps are treated as UTC.
    """
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


def norm_location(loc: Optional[str]) -> str:
    if loc is None or loc == "":
        return "--"
    return str(loc)


def norm_station_id(station_id: Optional[str], network: Optional[str] = None,
                    station: Optional[str] = None, location: Optional[str] = None) -> str:
    """Normalize station id to network.station.location, using -- for empty location."""
    if station_id:
        parts = str(station_id).split(".")
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}.{norm_location(parts[2])}"
        return str(station_id)
    return f"{network}.{station}.{norm_location(location)}"


DEFAULT_PHASE_MAP = {
    "P": ["Pg"],
    "S": ["Sg"],
    "Pg": ["Pg"],
    "Sg": ["Sg"],
    "Pn": ["Pg", "Pn", "P"],
    "Sn": ["Sg", "Sn", "S"],
}


def parse_phase_map(text: Optional[str]) -> Dict[str, List[str]]:
    """Parse phase map string like 'P:Pg,Pn;S:Sg,Sn'."""
    if not text:
        return dict(DEFAULT_PHASE_MAP)
    phase_map: Dict[str, List[str]] = {}
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        left, right = item.split(":", 1)
        phase_map[left.strip()] = [x.strip() for x in right.split(",") if x.strip()]
    return phase_map


# -----------------------------
# SQLite index for huge JSONL
# -----------------------------

def connect_db(db_path: Path) -> sqlite3.Connection:
    """Open SQLite DB and automatically create its parent directory."""
    db_path = Path(db_path).expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")  # about 200 MB
    return conn


def init_pick_index(conn: sqlite3.Connection, drop_existing: bool = False) -> None:
    cur = conn.cursor()
    if drop_existing:
        cur.execute("DROP TABLE IF EXISTS auto_picks")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS auto_picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT NOT NULL,
            network TEXT,
            station TEXT,
            location TEXT,
            phase_name TEXT NOT NULL,
            time_epoch REAL NOT NULL,
            sec_key INTEGER NOT NULL,
            phase_prob REAL,
            polarity TEXT,
            polarity_prob REAL,
            snr REAL,
            amplitude REAL,
            h5_file TEXT,
            raw_json TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_station_phase_sec ON auto_picks(station_id, phase_name, sec_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_station_sec ON auto_picks(station_id, sec_key)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_time ON auto_picks(time_epoch)")
    conn.commit()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip bad JSONL line {line_no}: {exc}", file=sys.stderr)


def build_auto_pick_index(
    auto_jsonl: Path,
    db_path: Path,
    batch_size: int = 50000,
    drop_existing: bool = False,
    keep_raw_json: bool = False,
    progress_every: int = 200000,
) -> None:
    auto_jsonl = Path(auto_jsonl).expanduser().resolve()
    db_path = Path(db_path).expanduser().resolve()
    if not auto_jsonl.exists():
        raise FileNotFoundError(f"auto JSONL not found: {auto_jsonl}")

    conn = connect_db(db_path)
    init_pick_index(conn, drop_existing=drop_existing)
    cur = conn.cursor()

    insert_sql = (
        "INSERT INTO auto_picks "
        "(station_id, network, station, location, phase_name, time_epoch, sec_key, "
        "phase_prob, polarity, polarity_prob, snr, amplitude, h5_file, raw_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    batch = []
    n = 0
    skipped = 0
    for rec in iter_jsonl(auto_jsonl):
        if rec.get("record_type") != "phase_pick":
            continue
        try:
            station_info = rec.get("station_info") or {}
            station_id = norm_station_id(
                station_info.get("station_id"),
                station_info.get("network"),
                station_info.get("station"),
                station_info.get("location"),
            )
            phase_name = str(rec.get("phase_name"))
            t = parse_utc_to_epoch_seconds(rec.get("phase_time"))
            sec_key = int(math.floor(t))
            raw_json = json.dumps(rec, ensure_ascii=False) if keep_raw_json else None
            batch.append((
                station_id,
                station_info.get("network"),
                station_info.get("station"),
                norm_location(station_info.get("location")),
                phase_name,
                t,
                sec_key,
                rec.get("phase_prob"),
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
                print(f"[WARN] skip record: {exc}", file=sys.stderr)
            continue

        if len(batch) >= batch_size:
            cur.executemany(insert_sql, batch)
            conn.commit()
            n += len(batch)
            batch.clear()
            if n % progress_every < batch_size:
                print(f"[INDEX] inserted {n:,} picks, skipped {skipped:,}")

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        n += len(batch)

    conn.execute("ANALYZE")
    conn.commit()
    conn.close()
    print(f"[INDEX] done. inserted {n:,} picks, skipped {skipped:,}. db={db_path}")


# -----------------------------
# Label loading
# -----------------------------

@dataclass
class LabelPick:
    label_phase: str
    label_time_epoch: float
    station_id: str
    status: str
    event_id: Optional[str]
    distance_km: Optional[float]
    raw: Dict[str, Any]


def iter_label_picks(label_json: Path) -> Iterable[LabelPick]:
    with label_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    years = data.get("years", {})
    for _year_id, year_obj in years.items():
        for _day_id, day_obj in (year_obj.get("days") or {}).items():
            for event_id, event_obj in (day_obj.get("events") or {}).items():
                for station_id0, sta_obj in (event_obj.get("stations") or {}).items():
                    for p in (sta_obj.get("picks") or []):
                        try:
                            station_id = norm_station_id(
                                p.get("station_id") or station_id0,
                                p.get("network"),
                                p.get("station"),
                                p.get("location"),
                            )
                            yield LabelPick(
                                label_phase=str(p.get("phase")),
                                label_time_epoch=parse_utc_to_epoch_seconds(p.get("time")),
                                station_id=station_id,
                                status=str(p.get("status", "unknown")),
                                event_id=p.get("event_id") or event_id,
                                distance_km=p.get("distance_km"),
                                raw=p,
                            )
                        except Exception as exc:
                            print(f"[WARN] skip label pick in event={event_id}, station={station_id0}: {exc}", file=sys.stderr)


# -----------------------------
# Matching and statistics
# -----------------------------

# -----------------------------
# Waveform coverage index
# -----------------------------

class WaveformCoverageIndex:
    """In-memory coverage index built from the waveform SQLite index.

    For each station_key (network.station), stores a sorted list of
    (start_epoch, end_epoch) tuples.  Coverage queries use bisect for
    O(log N) lookup, so the 290 k label-pick loop stays fast.

    Station matching ignores the location code, because the waveform index
    uses 2-part keys (e.g. 'CI.AVM') while label station_ids are 3-part
    (e.g. 'CI.AVM.--').
    """

    def __init__(self, waveform_db: Path) -> None:
        waveform_db = Path(waveform_db).expanduser().resolve()
        if not waveform_db.exists():
            raise FileNotFoundError(f"Waveform index DB not found: {waveform_db}")
        self._index: Dict[str, List[Tuple[float, float]]] = {}
        self._load(waveform_db)

    def _load(self, db_path: Path) -> None:
        """Load all waveform segments into memory, merging by station_key."""
        conn = sqlite3.connect(str(db_path))
        # Load one row per (station_key, start_epoch, end_epoch); channel/location
        # don't matter for coverage — any channel present means the station was processed.
        sql = """
            SELECT station_key, MIN(start_epoch) AS t0, MAX(end_epoch) AS t1
            FROM waveform_segments
            GROUP BY station_key, CAST(start_epoch / 86400 AS INTEGER)
            ORDER BY station_key, t0
        """
        raw: Dict[str, List[Tuple[float, float]]] = {}
        for row in conn.execute(sql):
            key, t0, t1 = str(row[0]), float(row[1]), float(row[2])
            raw.setdefault(key, []).append((t0, t1))
        conn.close()

        # Merge overlapping/adjacent intervals per station
        for key, segs in raw.items():
            segs.sort()
            merged: List[Tuple[float, float]] = []
            for t0, t1 in segs:
                if merged and t0 <= merged[-1][1] + 1.0:   # 1 s gap tolerance
                    merged[-1] = (merged[-1][0], max(merged[-1][1], t1))
                else:
                    merged.append((t0, t1))
            self._index[key] = merged

        n_sta = len(self._index)
        n_seg = sum(len(v) for v in self._index.values())
        print(f"[COVERAGE] loaded {n_sta} stations, {n_seg} merged segments from {db_path.name}")

    @staticmethod
    def _station_key_from_id(station_id: str) -> str:
        """Extract 'network.station' from a 3-part 'network.station.location' id."""
        parts = station_id.split(".")
        if len(parts) >= 2:
            return f"{parts[0]}.{parts[1]}"
        return station_id

    def has_coverage(self, station_id: str, time_epoch: float) -> bool:
        """Return True if any waveform segment covers *time_epoch* for this station."""
        key = self._station_key_from_id(station_id)
        segs = self._index.get(key)
        if not segs:
            return False
        # Find the last segment whose start <= time_epoch
        starts = [s[0] for s in segs]
        idx = bisect.bisect_right(starts, time_epoch) - 1
        if idx < 0:
            return False
        return segs[idx][1] >= time_epoch

    def covered_networks(self) -> List[str]:
        return sorted({k.split(".")[0] for k in self._index})


@dataclass
class MatchResult:
    subset: str
    label_phase: str
    station_id: str
    event_id: Optional[str]
    label_time_epoch: float
    matched: bool
    auto_phase: Optional[str] = None
    auto_time_epoch: Optional[float] = None
    residual_s: Optional[float] = None  # auto - label
    phase_prob: Optional[float] = None
    snr: Optional[float] = None
    distance_km: Optional[float] = None
    has_waveform: Optional[bool] = None   # True/False when waveform_db used; None otherwise


def query_nearest_auto_pick(
    conn: sqlite3.Connection,
    station_id: str,
    auto_phases: Sequence[str],
    label_time_epoch: float,
    search_window_s: float,
    min_prob: Optional[float] = None,
) -> Optional[Tuple[str, float, float, Optional[float], Optional[float]]]:
    """Return nearest auto pick as (phase, time_epoch, residual, prob, snr)."""
    sec0 = int(math.floor(label_time_epoch - search_window_s))
    sec1 = int(math.floor(label_time_epoch + search_window_s))
    placeholders = ",".join("?" for _ in auto_phases)
    params: List[Any] = [station_id, *auto_phases, sec0, sec1]
    prob_clause = ""
    if min_prob is not None:
        prob_clause = " AND phase_prob >= ?"
        params.append(float(min_prob))

    sql = f"""
        SELECT phase_name, time_epoch, phase_prob, snr
        FROM auto_picks
        WHERE station_id = ?
          AND phase_name IN ({placeholders})
          AND sec_key BETWEEN ? AND ?
          {prob_clause}
        ORDER BY ABS(time_epoch - ?) ASC
        LIMIT 1
    """
    params.append(float(label_time_epoch))
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    phase, t, prob, snr = row
    residual = float(t) - float(label_time_epoch)
    if abs(residual) > search_window_s:
        return None
    return phase, float(t), residual, prob, snr


def evaluate(
    label_json: Path,
    db_path: Path,
    outdir: Path,
    phase_map: Dict[str, List[str]],
    tp_tol: float = 1.5,
    err_window: float = 5.0,
    min_prob: Optional[float] = None,
    waveform_db: Optional[Path] = None,
) -> Tuple[List[MatchResult], Dict[str, Any]]:
    outdir = Path(outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(
            f"SQLite index DB not found: {db_path}. "
            "Run once with --build-index, or check --index-db."
        )
    conn = connect_db(db_path)

    # Optional waveform coverage filter
    cov_index: Optional[WaveformCoverageIndex] = None
    if waveform_db is not None:
        cov_index = WaveformCoverageIndex(Path(waveform_db))

    results: List[MatchResult] = []
    label_counter = Counter()
    label_status_counter = Counter()

    for lab in iter_label_picks(label_json):
        if lab.label_phase not in phase_map:
            continue
        label_counter[lab.label_phase] += 1
        label_status_counter[(lab.status, lab.label_phase)] += 1

        # Waveform coverage check
        has_waveform: Optional[bool] = None
        if cov_index is not None:
            has_waveform = cov_index.has_coverage(lab.station_id, lab.label_time_epoch)

        auto_phases = phase_map[lab.label_phase]
        # Search by wider window for residual distribution. TP uses tp_tol later.
        nearest = query_nearest_auto_pick(
            conn, lab.station_id, auto_phases, lab.label_time_epoch,
            search_window_s=err_window, min_prob=min_prob,
        )

        for subset in ("all", "manual", "automatic"):
            if subset == "manual" and lab.status != "manual":
                continue
            if subset == "automatic" and lab.status != "automatic":
                continue
            if nearest is None:
                results.append(MatchResult(
                    subset=subset, label_phase=lab.label_phase, station_id=lab.station_id,
                    event_id=lab.event_id, label_time_epoch=lab.label_time_epoch,
                    matched=False, distance_km=lab.distance_km,
                    has_waveform=has_waveform,
                ))
            else:
                auto_phase, auto_time, residual, prob, snr = nearest
                results.append(MatchResult(
                    subset=subset, label_phase=lab.label_phase, station_id=lab.station_id,
                    event_id=lab.event_id, label_time_epoch=lab.label_time_epoch,
                    matched=abs(residual) <= tp_tol,
                    auto_phase=auto_phase, auto_time_epoch=auto_time,
                    residual_s=residual, phase_prob=prob, snr=snr,
                    distance_km=lab.distance_km, has_waveform=has_waveform,
                ))

    conn.close()

    summary = summarize_results(results, label_counter, label_status_counter, tp_tol, err_window)
    summary["auto_pick_count"] = get_auto_counts(db_path, min_prob=min_prob, phase_map=phase_map)
    write_outputs(results, summary, outdir)
    return results, summary


def fit_student_t(residuals: np.ndarray) -> Dict[str, Optional[float]]:
    if residuals.size < 3 or scipy_stats is None:
        return {"df": None, "loc": None, "scale": None}
    try:
        df, loc, scale = scipy_stats.t.fit(residuals)
        return {"df": float(df), "loc": float(loc), "scale": float(scale)}
    except Exception:
        return {"df": None, "loc": None, "scale": None}


def fit_gaussian(residuals: np.ndarray) -> Dict[str, Optional[float]]:
    """Maximum-likelihood Gaussian fit for residuals."""
    if residuals.size < 2:
        return {"mean": None, "std_mle": None, "std_unbiased": None}
    return {
        "mean": float(np.mean(residuals)),
        "std_mle": float(np.std(residuals, ddof=0)),
        "std_unbiased": float(np.std(residuals, ddof=1)),
    }


def get_auto_counts(
    db_path: Path,
    min_prob: Optional[float] = None,
    phase_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Count automatic picks in the SQLite index.

    Returns original automatic phase counts, e.g. Pg/Sg, plus optional counts
    mapped to label phases, e.g. P->Pg and S->Sg. If min_prob is set, the
    counts are computed after applying phase_prob >= min_prob.
    """
    conn = connect_db(db_path)

    where = ""
    params: List[Any] = []
    if min_prob is not None:
        where = "WHERE phase_prob >= ?"
        params.append(float(min_prob))

    rows = conn.execute(
        f"SELECT phase_name, COUNT(*) FROM auto_picks {where} GROUP BY phase_name ORDER BY phase_name",
        params,
    ).fetchall()
    by_auto_phase = {str(ph): int(c) for ph, c in rows}
    total = int(sum(by_auto_phase.values()))

    mapped: Dict[str, int] = {}
    if phase_map:
        for label_phase, auto_phases in phase_map.items():
            mapped[label_phase] = int(sum(by_auto_phase.get(ap, 0) for ap in auto_phases))

    conn.close()
    return {
        "filter": {"min_prob": min_prob},
        "total": total,
        "by_auto_phase": by_auto_phase,
        "mapped_to_label_phase": mapped,
    }


def summarize_results(
    results: List[MatchResult],
    label_counter: Counter,
    label_status_counter: Counter,
    tp_tol: float,
    err_window: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "tp_tolerance_s": tp_tol,
        "residual_window_s": err_window,
        "label_phase_count_all_status": dict(label_counter),
        "label_phase_count_by_status": {f"{k[0]}:{k[1]}": v for k, v in label_status_counter.items()},
        "subsets": {},
    }

    for subset in sorted({r.subset for r in results}):
        subset_results = [r for r in results if r.subset == subset]
        phases = sorted({r.label_phase for r in subset_results})
        subset_summary: Dict[str, Any] = {}
        for ph in phases:
            ph_results = [r for r in subset_results if r.label_phase == ph]
            n_label = len(ph_results)
            n_tp = sum(1 for r in ph_results if r.matched)
            residuals = np.array([r.residual_s for r in ph_results if r.residual_s is not None and abs(r.residual_s) <= err_window], dtype=float)
            tp_residuals = np.array([r.residual_s for r in ph_results if r.matched and r.residual_s is not None], dtype=float)

            # Waveform-coverage-corrected recall: denominator is only labels that
            # have waveform data available (has_waveform=True). Labels where
            # has_waveform is None (waveform_db not supplied) are excluded from
            # the covered subset so the field stays None rather than 0/1.
            cov_results = [r for r in ph_results if r.has_waveform is True]
            n_label_cov = len(cov_results)
            n_tp_cov = sum(1 for r in cov_results if r.matched)
            # Distinguish "waveform_db not used" (all None) from "used but 0 covered"
            use_cov = any(r.has_waveform is not None for r in ph_results)

            phase_summary = {
                "n_label": int(n_label),
                "n_matched_within_tp_tol": int(n_tp),
                "recall": float(n_tp / n_label) if n_label else None,
                # Coverage-corrected metrics (only present when --waveform-db is used)
                "n_label_with_waveform": int(n_label_cov) if use_cov else None,
                "n_tp_with_waveform": int(n_tp_cov) if use_cov else None,
                "recall_covered": float(n_tp_cov / n_label_cov) if (use_cov and n_label_cov) else None,
                "n_residual_within_err_window": int(residuals.size),
                "residual_mean_s": float(np.mean(residuals)) if residuals.size else None,
                "residual_std_s": float(np.std(residuals, ddof=1)) if residuals.size > 1 else None,
                "residual_median_s": float(np.median(residuals)) if residuals.size else None,
                "residual_abs_p90_s": float(np.percentile(np.abs(residuals), 90)) if residuals.size else None,
                "residual_abs_p95_s": float(np.percentile(np.abs(residuals), 95)) if residuals.size else None,
                "tp_residual_std_s": float(np.std(tp_residuals, ddof=1)) if tp_residuals.size > 1 else None,
                "gaussian_fit_all_within_err_window": fit_gaussian(residuals),
                "student_t_fit_all_within_err_window": fit_student_t(residuals),
            }
            subset_summary[ph] = phase_summary

        # Combined P+S / Pg+Sg style summary
        n_label_all = len(subset_results)
        n_tp_all = sum(1 for r in subset_results if r.matched)
        cov_all = [r for r in subset_results if r.has_waveform is True]
        n_label_cov_all = len(cov_all)
        n_tp_cov_all = sum(1 for r in cov_all if r.matched)
        use_cov_all = any(r.has_waveform is not None for r in subset_results)
        subset_summary["P_S_combined"] = {
            "n_label": int(n_label_all),
            "n_matched_within_tp_tol": int(n_tp_all),
            "recall": float(n_tp_all / n_label_all) if n_label_all else None,
            "n_label_with_waveform": int(n_label_cov_all) if use_cov_all else None,
            "n_tp_with_waveform": int(n_tp_cov_all) if use_cov_all else None,
            "recall_covered": float(n_tp_cov_all / n_label_cov_all) if (use_cov_all and n_label_cov_all) else None,
        }
        summary["subsets"][subset] = subset_summary
    return summary


def write_outputs(results: List[MatchResult], summary: Dict[str, Any], outdir: Path) -> None:
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (outdir / "matches.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    with (outdir / "summary.tsv").open("w", encoding="utf-8") as f:
        f.write("subset\tphase\tn_label\tn_tp\trecall\tn_label_with_waveform\tn_tp_with_waveform\trecall_covered\tn_residual\tmean_s\tstd_s\tmedian_s\tabs_p90_s\tabs_p95_s\tgauss_mean\tgauss_std_mle\tgauss_std_unbiased\tt_df\tt_loc\tt_scale\n")
        for subset, ss in summary["subsets"].items():
            for phase, d in ss.items():
                if phase == "P_S_combined":
                    continue
                gfit = d.get("gaussian_fit_all_within_err_window", {}) or {}
                tfit = d.get("student_t_fit_all_within_err_window", {}) or {}
                f.write("\t".join([
                    subset, phase,
                    str(d.get("n_label")), str(d.get("n_matched_within_tp_tol")), str(d.get("recall")),
                    str(d.get("n_label_with_waveform")), str(d.get("n_tp_with_waveform")), str(d.get("recall_covered")),
                    str(d.get("n_residual_within_err_window")), str(d.get("residual_mean_s")),
                    str(d.get("residual_std_s")), str(d.get("residual_median_s")),
                    str(d.get("residual_abs_p90_s")), str(d.get("residual_abs_p95_s")),
                    str(gfit.get("mean")), str(gfit.get("std_mle")), str(gfit.get("std_unbiased")),
                    str(tfit.get("df")), str(tfit.get("loc")), str(tfit.get("scale")),
                ]) + "\n")


def load_matches(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_results(outdir: Path) -> None:
    import matplotlib.pyplot as plt

    matches_path = outdir / "matches.jsonl"
    if not matches_path.exists():
        print(f"[PLOT] missing {matches_path}", file=sys.stderr)
        return
    rows = load_matches(matches_path)
    plot_dir = outdir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Residual distribution by subset and phase
    for subset in sorted({r["subset"] for r in rows}):
        for phase in sorted({r["label_phase"] for r in rows if r["subset"] == subset}):
            vals = np.array([
                r["residual_s"] for r in rows
                if r["subset"] == subset and r["label_phase"] == phase and r.get("residual_s") is not None
            ], dtype=float)
            if vals.size == 0:
                continue
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            fig = plt.figure(figsize=(7.5, 4.8))
            plt.hist(vals, bins=100, density=True, alpha=0.55, label=f"Residuals (n={vals.size:,})")

            x_min, x_max = float(np.min(vals)), float(np.max(vals))
            if x_min == x_max:
                x_min -= 1.0
                x_max += 1.0
            x = np.linspace(x_min, x_max, 800)

            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=0)) if vals.size > 1 else 0.0
            if scipy_stats is not None and sigma > 0:
                gauss_pdf = scipy_stats.norm.pdf(x, loc=mu, scale=sigma)
                plt.plot(x, gauss_pdf, linewidth=2, label=f"Gaussian μ={mu:.3f}, σ={sigma:.3f}")

                if vals.size >= 3:
                    try:
                        df, loc, scale = scipy_stats.t.fit(vals)
                        if scale > 0:
                            t_pdf = scipy_stats.t.pdf(x, df, loc=loc, scale=scale)
                            plt.plot(x, t_pdf, linewidth=2, label=f"Student-t df={df:.2f}, loc={loc:.3f}, scale={scale:.3f}")
                    except Exception as exc:
                        print(f"[PLOT] Student-t fit failed for {subset}/{phase}: {exc}", file=sys.stderr)
            else:
                plt.text(
                    0.02, 0.95,
                    "Install scipy to overlay Gaussian/Student-t PDFs",
                    transform=plt.gca().transAxes,
                    va="top",
                )

            plt.axvline(0.0, linestyle="--", linewidth=1)
            plt.xlabel("Residual time: automatic - label (s)")
            plt.ylabel("Probability density")
            plt.title(f"Residual distribution with fits: {subset}, {phase}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            fig.savefig(plot_dir / f"residual_fit_{subset}_{phase}.png", dpi=220)
            plt.close(fig)

    # Recall bar chart
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    labels, vals = [], []
    for subset, ss in summary["subsets"].items():
        for phase, d in ss.items():
            if phase == "P_S_combined":
                continue
            if d.get("recall") is not None:
                labels.append(f"{subset}-{phase}")
                vals.append(d["recall"])
    if labels:
        fig = plt.figure(figsize=(max(7, 0.8 * len(labels)), 4.5))
        plt.bar(labels, vals)
        plt.ylim(0, 1)
        plt.ylabel("Recall")
        plt.title("Phase-pick recall")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plot_dir / "recall_bar.png", dpi=200)
        plt.close(fig)


    # Automatic pick counts by original automatic phase
    auto_counts = summary.get("auto_pick_count", {}).get("by_auto_phase", {})
    if auto_counts:
        phases = list(auto_counts.keys())
        counts = [auto_counts[p] for p in phases]
        fig = plt.figure(figsize=(max(7, 0.8 * len(phases)), 4.5))
        plt.bar(phases, counts)
        plt.ylabel("Number of automatic picks")
        plt.title("Automatic pick counts by phase")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(plot_dir / "auto_pick_count_bar.png", dpi=220)
        plt.close(fig)
    print(f"[PLOT] saved figures to {plot_dir}")


# -----------------------------
# Optional utilities
# -----------------------------

def print_db_info(db_path: Path) -> None:
    db_path = Path(db_path).expanduser().resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"SQLite index DB not found: {db_path}")
    conn = connect_db(db_path)
    n = conn.execute("SELECT COUNT(*) FROM auto_picks").fetchone()[0]
    print(f"auto_picks: {n:,}")
    print("phase counts:")
    for ph, c in conn.execute("SELECT phase_name, COUNT(*) FROM auto_picks GROUP BY phase_name ORDER BY COUNT(*) DESC"):
        print(f"  {ph}: {c:,}")
    conn.close()


# -----------------------------
# CLI
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate automatic phase picks against continuous-HDF5 annotation JSON.")
    parser.add_argument("--auto-jsonl", type=Path, default=Path("data/picks/skynet.phase.jsonl"))
    parser.add_argument("--label-json", type=Path, default=Path("data/label/annotations_for_continuous_hdf5.json"))
    parser.add_argument("--index-db", type=Path, default=Path("~/skynet.pick.index.sqlite"))
    parser.add_argument("--outdir", type=Path, default=Path("eval_skynet"))
    parser.add_argument("--build-index", action="store_true", help="Build or update SQLite index from auto JSONL.")
    parser.add_argument("--drop-existing", action="store_true", help="Drop existing index table before rebuilding.")
    parser.add_argument("--keep-raw-json", action="store_true", help="Store raw JSON in SQLite. Not recommended for 40GB files.")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--tp-tol", type=float, default=1.5, help="TP tolerance in seconds.")
    parser.add_argument("--err-window", type=float, default=5.0, help="Window for residual distribution in seconds.")
    parser.add_argument("--min-prob", type=float, default=None, help="Optional minimum automatic pick probability.")
    parser.add_argument("--phase-map", type=str, default=None, help="Example: 'P:Pg;S:Sg' or 'P:Pg,Pn;S:Sg,Sn'.")
    parser.add_argument("--waveform-db", type=Path, default=None,
                        help="Waveform coverage SQLite index (built by hdf5_waveform_index.py). "
                             "When supplied, each label pick is checked for waveform availability. "
                             "recall_covered is computed over the subset that has waveform data, "
                             "so labels from un-processed stations/days are excluded from the denominator.")
    parser.add_argument("--plot", action="store_true", help="Generate figures after evaluation.")
    parser.add_argument("--db-info", action="store_true", help="Only print index database info.")
    args = parser.parse_args()

    if args.build_index:
        build_auto_pick_index(
            args.auto_jsonl, args.index_db,
            batch_size=args.batch_size,
            drop_existing=args.drop_existing,
            keep_raw_json=args.keep_raw_json,
        )

    if args.db_info:
        print_db_info(args.index_db)
        return

    phase_map = parse_phase_map(args.phase_map)
    results, summary = evaluate(
        label_json=args.label_json,
        db_path=args.index_db,
        outdir=args.outdir,
        phase_map=phase_map,
        tp_tol=args.tp_tol,
        err_window=args.err_window,
        min_prob=args.min_prob,
        waveform_db=args.waveform_db,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.plot:
        plot_results(args.outdir)


if __name__ == "__main__":
    main()
 