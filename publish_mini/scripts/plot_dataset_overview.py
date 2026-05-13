#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Overview Figure
=======================
A comprehensive visualization of the continuous seismic waveform dataset,
contrasting the 2019 Ridgecrest earthquake sequence with a quiet 2021 period.

Usage
-----
python plot_dataset_overview.py \
    --label-json  data/label/annotations_for_continuous_hdf5.json \
    --waveform-db data/index/waveform_index.sqlite \
    --h5-dir      data/hdf5 \
    --out         figures/dataset_overview.pdf

Dependencies
------------
  numpy, matplotlib, h5py, scipy (optional, for envelope)
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────────────────────────

RIDGECREST_COLOR  = "#C0392B"   # deep red  – 2019 Ridgecrest sequence
QUIET_COLOR       = "#2471A3"   # steel blue – 2021 quiet period
NET_COLORS = {"CI": "#1F618D", "BK": "#196F3D", "NC": "#B7770D"}
NET_LABELS = {"CI": "CI – Southern California", "BK": "BK – Berkeley", "NC": "NC – Northern California"}
P_COLOR = "#1A5276"
S_COLOR = "#922B21"
ACCENT   = "#F39C12"

FONT_TITLE  = dict(fontsize=14, fontweight="bold", color="#1C2833")
FONT_LABEL  = dict(fontsize=12, color="#2C3E50")
FONT_ANNOT  = dict(fontsize=10, color="#555555")


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_stations(db_path: Path) -> List[Dict]:
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT station_key, network, station,
               AVG(latitude) as lat, AVG(longitude) as lon,
               COUNT(DISTINCT DATE(datetime(start_epoch,'unixepoch'))) as n_days
        FROM waveform_segments
        WHERE latitude IS NOT NULL AND ABS(latitude) > 0.1
        GROUP BY station_key
    """).fetchall()
    conn.close()
    return [{"key": r[0], "net": r[1], "sta": r[2],
             "lat": r[3], "lon": r[4], "n_days": r[5]} for r in rows]


def load_events(label_json: Path) -> List[Dict]:
    with open(label_json, encoding="utf-8") as f:
        data = json.load(f)
    events = []
    for yr in data.get("years", {}).values():
        for day_obj in yr.get("days", {}).values():
            for ev in day_obj.get("events", {}).values():
                evd = ev.get("event", {})
                t   = evd.get("event_time", "")
                mag = evd.get("magnitude")
                if not t or mag is None:
                    continue
                events.append({
                    "time": t, "day": t[:10],
                    "mag":  float(mag),
                    "lat":  evd.get("latitude"),
                    "lon":  evd.get("longitude"),
                    "dep":  evd.get("depth_km"),
                    "picks": ev.get("counts", {}).get("pick_count", 0),
                })
    return events


def load_station_picks(label_json: Path, station_id: str, date_str: str) -> List[Dict]:
    with open(label_json, encoding="utf-8") as f:
        data = json.load(f)
    picks = []
    for yr in data.get("years", {}).values():
        for day_obj in yr.get("days", {}).values():
            for ev in day_obj.get("events", {}).values():
                for sid0, sobj in ev.get("stations", {}).items():
                    for p in sobj.get("picks", []):
                        sid = p.get("station_id") or sid0
                        t   = p.get("time", "")
                        if sid == station_id and t.startswith(date_str):
                            picks.append({"time": t, "phase": p.get("phase", "?"),
                                          "status": p.get("status", "")})
    return picks

def select_best_station_day(
    label_json: Path,
    waveform_db: Path,
    year_prefix: str = "2019",
    min_picks: int = 50,
    max_picks: int = 300,
    preferred_channels: Tuple[str, ...] = ("HHZ", "BHZ", "EHZ", "HNZ"),
) -> Optional[Tuple[str, str, str, int]]:
    """
    Select station-day with the largest number of reference picks
    and available waveform in the database.

    Returns
    -------
    (station_id, channel, date_str, n_picks)
    """
    with open(label_json, encoding="utf-8") as f:
        data = json.load(f)

    counter = defaultdict(int)

    for yr in data.get("years", {}).values():
        for day_obj in yr.get("days", {}).values():
            for ev in day_obj.get("events", {}).values():
                for sid0, sobj in ev.get("stations", {}).items():
                    for p in sobj.get("picks", []):
                        sid = p.get("station_id") or sid0
                        t = p.get("time", "")
                        if sid and t.startswith(year_prefix):
                            counter[(sid, t[:10])] += 1

    if not counter:
        return None

    conn = sqlite3.connect(str(waveform_db))

    candidates = [
        ((sid, date_str), n_picks)
        for (sid, date_str), n_picks in counter.items()
        if min_picks <= n_picks <= max_picks
    ]

    candidates = sorted(candidates, key=lambda kv: kv[1], reverse=True)
    for (sid, date_str), n_picks in candidates:
        for ch in preferred_channels:
            row = conn.execute("""
                SELECT channel
                FROM waveform_segments
                WHERE station_id=?
                  AND channel=?
                  AND DATE(datetime(start_epoch,'unixepoch'))=?
                ORDER BY npts DESC
                LIMIT 1
            """, (sid, ch, date_str)).fetchone()

            if row is not None:
                conn.close()
                return sid, ch, date_str, n_picks

    conn.close()
    return None

def query_waveform(db_path: Path, station_id: str, channel: str, date_str: str) -> Optional[Dict]:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("""
        SELECT dataset_path, h5_file, npts, sampling_rate, start_epoch, latitude, longitude
        FROM waveform_segments
        WHERE station_id=? AND channel=?
          AND DATE(datetime(start_epoch,'unixepoch'))=?
        ORDER BY npts DESC LIMIT 1
    """, (station_id, channel, date_str)).fetchone()
    conn.close()
    if row is None:
        return None
    return {"path": row[0], "h5_file": row[1], "npts": row[2],
            "sr": row[3], "t0": row[4],
            "lat": row[5], "lon": row[6]}


def read_waveform_downsampled(info: Dict, h5_dir: Optional[Path] = None,
                               target_hz: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Read waveform and downsample to target_hz via RMS in each window."""
    h5_path = info["h5_file"]
    if h5_dir is not None:
        h5_path = str(h5_dir / Path(h5_path).name)
    with h5py.File(h5_path, "r") as h5:
        raw = h5[info["path"]][:]
    raw = raw.astype(np.float32)

    sr      = float(info["sr"])
    win     = max(1, int(sr / target_hz))
    n_wins  = len(raw) // win
    data    = raw[: n_wins * win].reshape(n_wins, win)

    # RMS envelope for display
    envelope = np.sqrt(np.mean(data ** 2, axis=1))

    t0   = float(info["t0"])
    times = np.arange(n_wins) / target_hz  # seconds from midnight
    return times, envelope


def iso_to_epoch(s: str) -> float:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


# ──────────────────────────────────────────────────────────────────────────────
# Individual panel drawers
# ──────────────────────────────────────────────────────────────────────────────

def draw_station_map(ax: plt.Axes, stations: List[Dict], events: List[Dict]) -> None:
    """Panel A – station map with network colours and Ridgecrest epicentres."""

    # Background colour
    ax.set_facecolor("#EBF5FB")

    # Plot stations per network
    for net, color in NET_COLORS.items():
        sub = [s for s in stations if s["net"] == net]
        lons = [s["lon"] for s in sub]
        lats = [s["lat"] for s in sub]
        ax.scatter(lons, lats, c=color, s=22, alpha=0.75, linewidths=0,
                   label=f"{net}  ({len(sub)} stations)", zorder=3)

    # Ridgecrest main shocks
    ridgecrest = [
        (35.705, -117.504, "M6.4\n2019-07-04"),
        (35.770, -117.599, "M7.1\n2019-07-06"),
    ]
    for lat, lon, lbl in ridgecrest:
        ax.scatter(lon, lat, marker="*", c=RIDGECREST_COLOR, s=420,
                   zorder=6, edgecolors="white", linewidths=0.7)
        ax.annotate(lbl, (lon, lat), xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=RIDGECREST_COLOR, fontweight="bold", zorder=7)

    # Mark the two showcase stations
    showcase = [
        ("CI.ADO.--", 34.550, -117.434, "CI.ADO\n(Ridgecrest\ncase)", "left"),
        ("CI.CSH.--", 33.644, -116.596, "CI.CSH\n(Quiet\ncase)",     "right"),
    ]
    for sid, lat, lon, lbl, ha in showcase:
        ax.scatter(lon, lat, marker="^", c=ACCENT, s=120, zorder=5,
                   edgecolors="white", linewidths=0.8)
        xoff = 6 if ha == "left" else -6
        ax.annotate(lbl, (lon, lat), xytext=(xoff, 7), textcoords="offset points",
                    ha=ha, fontsize=9.5, color="#6E2F1A", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0),
                    zorder=7)

    # Legend entries for epicentres and stations
    leg_extra = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=RIDGECREST_COLOR,
               markersize=7, label="Ridgecrest epicentre"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=ACCENT,
               markersize=5, label="Showcase station"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + leg_extra, labels + [h.get_label() for h in leg_extra],
              fontsize=7, loc="upper left",
              framealpha=0.88, edgecolor="#AAAAAA", labelspacing=0.3,
              handlelength=1.2, handletextpad=0.4, borderpad=0.4)

    # Map extent & labels
    ax.set_xlim(-124.6, -113.8)
    ax.set_ylim(32.2, 43.2)
    ax.set_xlabel("Longitude", **FONT_LABEL)
    ax.set_ylabel("Latitude", **FONT_LABEL)
    ax.tick_params(labelsize=10)

    # Simple graticule
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, lw=0.4, color="white", alpha=0.7)

    # Panel label
    ax.set_title("A · Seismic Network Coverage", loc="left", **FONT_TITLE, pad=6)

    # Dataset overview inset (text box)
    n_ci = sum(1 for s in stations if s["net"] == "CI")
    n_bk = sum(1 for s in stations if s["net"] == "BK")
    n_nc = sum(1 for s in stations if s["net"] == "NC")
    total_picks = sum(e["picks"] for e in events)
    summary = (
        f"Networks: CI·BK·NC\n"
        f"Stations:  {len(stations):,}  total\n"
        f"Days:  14  (7 + 7)\n"
        f"Events:  {len(events):,}\n"
        f"Total P/S picks:  {total_picks:,}"
    )
    ax.text(0.975, 0.97, summary,
            transform=ax.transAxes, fontsize=8.5,
            va="top", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", fc="white", alpha=0.88,
                      ec="#AAAAAA", lw=0.8))


def draw_activity_timeline(ax: plt.Axes, events: List[Dict]) -> None:
    """Panel B – events & picks per day with broken x-axis feel."""

    daily_ev  = defaultdict(int)
    daily_pk  = defaultdict(int)
    for e in events:
        daily_ev[e["day"]] += 1
        daily_pk[e["day"]] += e["picks"]

    days_2019 = sorted(d for d in daily_ev if d.startswith("2019"))
    days_2021 = sorted(d for d in daily_ev if d.startswith("2021"))
    all_days  = days_2019 + ["gap"] + days_2021

    # X positions with a visual gap
    pos = {}
    x = 0
    for d in days_2019:
        pos[d] = x; x += 1
    x += 1.2          # gap
    for d in days_2021:
        pos[d] = x; x += 1

    # Bars: events (left y), picks (right y)
    ax2 = ax.twinx()

    bar_w = 0.38
    for d in days_2019 + days_2021:
        col = RIDGECREST_COLOR if d.startswith("2019") else QUIET_COLOR
        x_  = pos[d]
        ax.bar(x_ - bar_w/2, max(daily_ev[d], 1), width=bar_w,
               color=col, alpha=0.90, zorder=3)
        ax2.bar(x_ + bar_w/2, max(daily_pk[d], 1), width=bar_w,
                color=col, alpha=0.45, zorder=2)

    # X tick labels
    xticks = [pos[d] for d in days_2019 + days_2021]
    xlbls  = [d[5:] for d in days_2019 + days_2021]   # MM-DD
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlbls, rotation=40, ha="right", fontsize=10)

    # Log scale on both y-axes
    ax.set_yscale("log")
    ax2.set_yscale("log")
    ax.set_ylim(bottom=0.7)
    ax2.set_ylim(bottom=0.7)

    # Gap annotation – place at a fixed log-scale-friendly y
    gap_x = (pos[days_2019[-1]] + pos[days_2021[0]]) / 2
    ax.text(gap_x, 2.5, "╌╌  ~2 yrs  ╌╌",
            ha="center", va="bottom", fontsize=9.5, color="#888888")

    # Axes styling – use compact 10^n tick labels
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax2.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.set_ylabel("Events / day", **FONT_LABEL)
    ax2.set_ylabel("Picks / day", labelpad=0, fontsize=12, color="#666666")
    ax.tick_params(axis="y", labelsize=10)
    ax2.tick_params(axis="y", labelsize=10, labelcolor="#888888")
    ax.set_xlim(-0.7, pos[days_2021[-1]] + 0.7)

    # Legend
    handles = [
        mpatches.Patch(color=RIDGECREST_COLOR, alpha=0.9, label="Events (2019)"),
        mpatches.Patch(color=QUIET_COLOR,       alpha=0.9, label="Events (2021)"),
        mpatches.Patch(color=RIDGECREST_COLOR,  alpha=0.4, label="Picks (2019)"),
        mpatches.Patch(color=QUIET_COLOR,        alpha=0.4, label="Picks (2021)"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="upper left",
              framealpha=0.85, edgecolor="#AAAAAA", ncol=2, labelspacing=0.4)
    ax.set_title("B · Daily Seismic Activity", loc="left", **FONT_TITLE, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top", "left"]].set_visible(False)
    ax.grid(axis="y", lw=0.4, alpha=0.5, zorder=0)
    box = ax.get_position()
    dx = 0.02
    ax.set_position([box.x0 - dx, box.y0, box.width, box.height])


def draw_magnitude_distribution(ax: plt.Axes, events: List[Dict]) -> None:
    """Panel C – cumulative magnitude-frequency plot for both periods."""

    ev_2019 = sorted([e["mag"] for e in events if e["day"].startswith("2019")])
    ev_2021 = sorted([e["mag"] for e in events if e["day"].startswith("2021")])

    def cdf(mags):
        m  = np.array(sorted(mags))
        n  = np.arange(len(m), 0, -1)  # cumulative from right
        return m, n

    m19, n19 = cdf(ev_2019)
    m21, n21 = cdf(ev_2021)

    ax.semilogy(m19, n19, color=RIDGECREST_COLOR, lw=1.8,
                label=f"2019  (N={len(ev_2019):,})")
    ax.semilogy(m21, n21, color=QUIET_COLOR, lw=1.8,
                label=f"2021  (N={len(ev_2021):,})")

    ax.fill_betweenx(n19, m19, alpha=0.10, color=RIDGECREST_COLOR)
    ax.fill_betweenx(n21, m21, alpha=0.10, color=QUIET_COLOR)

    # Mark main shocks
    for idx, mag, lbl in [(0, 7.1, "M7.1"), (1, 6.4, "M6.4")]:
        ax.axvline(mag, lw=1.2, ls="--", color=RIDGECREST_COLOR, alpha=0.7)
        ax.text(mag + 0.07, n19.max() * (idx*2+4)*0.1, lbl,
                fontsize=9.5, color=RIDGECREST_COLOR, va="top", fontweight="bold")

    ax.set_xlabel("Magnitude", **FONT_LABEL)
    ax.set_ylabel("Cumul. # events ≥ M", labelpad=0, **FONT_LABEL)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.85, edgecolor="#AAAAAA")
    ax.set_title("C · Magnitude–Frequency", loc="left", **FONT_TITLE, pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(lw=0.4, alpha=0.4)


def draw_waveform(ax: plt.Axes,
                  times: np.ndarray,
                  envelope: np.ndarray,
                  picks: List[Dict],
                  t0_epoch: float,
                  date_str: str,
                  station: str,
                  channel: str,
                  period_label: str,
                  color: str) -> None:
    """Panel D / E – single-day waveform envelope with pick markers."""

    # Normalise envelope for display
    env_norm = envelope / (np.percentile(envelope, 99) + 1e-9)
    env_norm = np.clip(env_norm, 0, 8)

    # Shade fill
    ax.fill_between(times / 3600, env_norm, alpha=0.35, color=color, lw=0)
    ax.plot(times / 3600, env_norm, lw=0.5, color=color, alpha=0.8)

    # Pick markers
    p_times = [p for p in picks if p["phase"] == "P"]
    s_times = [p for p in picks if p["phase"] == "S"]
    for group, col, yoff, lbl in [
        (p_times, P_COLOR, 0.92, "P"),
        (s_times, S_COLOR, 0.62, "S"),
    ]:
        for p in group:
            try:
                t_epoch = iso_to_epoch(p["time"])
                t_sec   = t_epoch - t0_epoch
                t_hr    = t_sec / 3600
                ax.axvline(t_hr, lw=0.7, color=col, alpha=0.65, zorder=4)
            except Exception:
                continue
        if group:
            ax.text(
                0.99, yoff,
                f"{lbl}  ({len(group)})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                color=col,
                fontweight="bold",
                zorder=20,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="#CCCCCC",
                    linewidth=0.4,
                    alpha=0.85,
                ),
            )

    # Axes
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)], fontsize=10)
    ax.set_ylabel("Norm. amplitude", **FONT_LABEL)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Title
    title = (f"{period_label} · {station}  {channel} · {date_str}")
    ax.set_title(title, loc="left", **FONT_TITLE, pad=5)

    # Annotation box
    n_picks = len(picks)

    ax.text(
        0.01, 0.97,                      # 更靠上
        f"Reference picks: {n_picks}",
        transform=ax.transAxes,
        fontsize=10,
        color="#333333",
        va="top",
        zorder=10,                       # 🔴 关键：压到最上层
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",           # 比 fc 更规范
            edgecolor="none",
            alpha=0.85                   # 稍微更实一点
        )
    )


def draw_stats_banner(ax: plt.Axes, stations: List[Dict], events: List[Dict]) -> None:
    """Panel F – horizontal key-numbers summary."""
    ax.axis("off")

    n_ev_2019  = sum(1 for e in events if e["day"].startswith("2019"))
    n_ev_2021  = sum(1 for e in events if e["day"].startswith("2021"))
    n_pk_2019  = sum(e["picks"] for e in events if e["day"].startswith("2019"))
    n_pk_2021  = sum(e["picks"] for e in events if e["day"].startswith("2021"))
    max_mag    = max(e["mag"] for e in events)

    items = [
        ("979",                  "Seismic\nstations",          "#1F618D"),
        ("CI·BK·NC",             "Networks",                   "#196F3D"),
        ("14 days",              "Continuous\ncoverage",        "#7D6608"),
        (f"{n_ev_2019:,}",       "2019 sequence\nevents",      RIDGECREST_COLOR),
        (f"{n_pk_2019:,}",       "2019 sequence\nP/S picks",   RIDGECREST_COLOR),
        (f"M{max_mag:.1f}",      "Largest\nevent",             RIDGECREST_COLOR),
        (f"{n_ev_2021:,}",       "2021 quiet\nevents",         QUIET_COLOR),
        (f"{n_pk_2021:,}",       "2021 quiet\nP/S picks",      QUIET_COLOR),
        (f"×{n_ev_2019//max(n_ev_2021,1):.0f}",
                                 "More events\nin 2019",       "#7B241C"),
    ]

    ncols = len(items)
    for i, (val, lbl, col) in enumerate(items):
        x = (i + 0.5) / ncols
        ax.text(x, 0.70, val, ha="center", va="center",
                fontsize=15, fontweight="bold", color=col,
                transform=ax.transAxes)
        ax.text(x, 0.18, lbl, ha="center", va="center",
                fontsize=7.2, color="#555555", linespacing=1.05,
                transform=ax.transAxes)

    # Divider lines
    for i in range(1, ncols):
        x = i / ncols
        ax.axvline(x, lw=0.5, color="#CCCCCC", ymin=0.05, ymax=0.95)

    ax.set_title("F · Dataset Overview and Scale", loc="left", **FONT_TITLE, pad=4)


# ──────────────────────────────────────────────────────────────────────────────
# Main figure assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_figure(label_json: Path, waveform_db: Path, h5_dir: Path,
                 out_path: Path) -> None:

    print("[1/6] Loading stations …")
    stations = load_stations(waveform_db)

    print("[2/6] Loading events & picks …")
    events = load_events(label_json)

    print("[3/6] Loading waveforms …")
    best_2019 = select_best_station_day(
    label_json=label_json,
    waveform_db=waveform_db,
    year_prefix="2019",
    max_picks=500,
    preferred_channels=("HHZ", "BHZ", "EHZ", "HNZ"),
    )
    
    
    if best_2019 is None:
        print("  [WARN] No valid 2019 station-day found. Fall back to CI.ADO.--")
        ridge_sid, ridge_ch, ridge_date = "CI.ADO.--", "HHZ", "2019-07-05"
    else:
        ridge_sid, ridge_ch, ridge_date, ridge_npicks = best_2019
        print(
            f"  Best 2019 station-day: {ridge_sid} {ridge_ch} "
            f"{ridge_date} with {ridge_npicks} reference picks"
        )


    wf_cfg = [
        ("ridgecrest", ridge_sid, ridge_ch, ridge_date, RIDGECREST_COLOR),
        ("quiet",      "CI.CSH.--", "HHZ", "2021-11-14", QUIET_COLOR),
    ]
    waveforms = {}
    picks_wf  = {}
    for label, sid, ch, date_str, col in wf_cfg:
        info = query_waveform(waveform_db, sid, ch, date_str)
        if info is None:
            print(f"  [WARN] waveform not found: {sid} {ch} {date_str}")
            continue
        print(f"  Reading {sid} {ch} {date_str}  npts={info['npts']:,} …")
        t, env = read_waveform_downsampled(info, h5_dir=h5_dir, target_hz=1.0)
        waveforms[label] = (t, env, info, col, sid, ch, date_str)
        picks_wf[label]  = load_station_picks(label_json, sid, date_str)
        print(f"    {len(picks_wf[label])} picks found for {sid} on {date_str}")

    # ── Layout ────────────────────────────────────────────────────────────────
    print("[4/6] Building figure …")
    fig = plt.figure(figsize=(12, 8), dpi=150)
    fig.patch.set_facecolor("white")

    gs_outer = gridspec.GridSpec(
        3, 1,
        hspace=0.52,
        #wspace=0.55, 
        height_ratios=[4.2, 1.55, 1.55],
        left=0.07, right=0.97, top=0.97, bottom=0.13,
    )

    # Row 0: map + timeline + magnitude
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[0],
        width_ratios=[1.35, 1.50, 1.15], wspace=0.30,
    )
    ax_map  = fig.add_subplot(gs_top[0])
    ax_time = fig.add_subplot(gs_top[1])
    ax_mag  = fig.add_subplot(gs_top[2])

    # Rows 1–2: waveforms
    ax_wf = {}
    for row_i, key in enumerate(["ridgecrest", "quiet"]):
        ax_wf[key] = fig.add_subplot(gs_outer[row_i + 1])

    # Stats banner: inset below the quiet waveform panel
    ax_banner = ax_wf["quiet"].inset_axes([0, -0.92, 1.0, 0.50])

    # ── Draw panels ───────────────────────────────────────────────────────────
    print("[5/6] Drawing panels …")

    draw_station_map(ax_map, stations, events)
    draw_activity_timeline(ax_time, events)
    draw_magnitude_distribution(ax_mag, events)

    for label, col, period_lbl in [
        ("ridgecrest", RIDGECREST_COLOR,
         "D · Dense Ridgecrest Aftershock Sequence"),
        ("quiet", QUIET_COLOR,
         "E · Quiet Period"),
    ]:
        ax = ax_wf[label]
        if label in waveforms:
            t, env, info, c, sid, ch, date_str = waveforms[label]
            draw_waveform(
                ax, t, env,
                picks_wf.get(label, []),
                t0_epoch=info["t0"],
                date_str=date_str,
                station=sid, channel=ch,
                period_label=period_lbl,
                color=c,
            )
        else:
            ax.text(0.5, 0.5, "Waveform not available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_title(period_lbl, loc="left", **FONT_TITLE, pad=5)
        ax.set_xlabel("Time (UTC)", **FONT_LABEL)

    draw_stats_banner(ax_banner, stations, events)

    # ── Title ─────────────────────────────────────────────────────────────────
    # suptitle removed per user request; panel titles (A–F) are retained.

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"[6/6] Saving → {out_path} …")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Done.  {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset overview figure.")
    parser.add_argument("--label-json",  type=Path,
                        default=Path("data/label/annotations_for_continuous_hdf5.json"))
    parser.add_argument("--waveform-db", type=Path,
                        default=Path("data/index/waveform_index.sqlite"))
    parser.add_argument("--h5-dir",      type=Path,
                        default=Path("data/hdf5"))
    parser.add_argument("--out",         type=Path,
                        default=Path("figures/dataset_overview.png"))
    parser.add_argument("--dpi",         type=int, default=200)
    args = parser.parse_args()

    build_figure(
        label_json   = args.label_json,
        waveform_db  = args.waveform_db,
        h5_dir       = args.h5_dir,
        out_path     = args.out,
    )


if __name__ == "__main__":
    main()
