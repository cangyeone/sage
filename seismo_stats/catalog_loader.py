"""
catalog_loader.py — Load earthquake catalog data from various sources.

Supported formats
-----------------
1. SAGE picks .txt  (output of sage_picker / conversational_agent batch picking)
2. CSV catalog      columns: time, magnitude, longitude, latitude, depth
3. JSON catalog     list of dicts with the same keys
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CatalogData:
    """Container for earthquake catalog information."""
    # Core fields (always present after loading from a catalog file)
    times: List[datetime] = field(default_factory=list)       # origin/pick times
    magnitudes: List[float] = field(default_factory=list)     # magnitudes (may be empty)
    longitudes: List[float] = field(default_factory=list)     # degrees E
    latitudes: List[float] = field(default_factory=list)      # degrees N
    depths: List[float] = field(default_factory=list)         # km

    # Additional metadata
    stations: List[str] = field(default_factory=list)         # NET.STA.LOC
    phases: List[str] = field(default_factory=list)           # Pg / Sg / Pn / Sn
    confidences: List[float] = field(default_factory=list)    # model confidence
    snrs: List[float] = field(default_factory=list)           # signal-to-noise ratio
    source_file: str = ""

    def __len__(self):
        return max(len(self.times), len(self.magnitudes))

    @property
    def has_magnitudes(self) -> bool:
        return len(self.magnitudes) > 0

    @property
    def has_locations(self) -> bool:
        return len(self.longitudes) > 0 and len(self.latitudes) > 0

    def summary(self) -> str:
        lines = [f"CatalogData  source: {self.source_file}"]
        lines.append(f"  records   : {len(self)}")
        if self.times:
            lines.append(f"  time range: {min(self.times)}  →  {max(self.times)}")
        if self.has_magnitudes:
            lines.append(f"  magnitude : {min(self.magnitudes):.1f} – {max(self.magnitudes):.1f}")
        if self.has_locations:
            lines.append(f"  longitude : {min(self.longitudes):.3f} – {max(self.longitudes):.3f}")
            lines.append(f"  latitude  : {min(self.latitudes):.3f} – {max(self.latitudes):.3f}")
        if self.phases:
            from collections import Counter
            ph = Counter(self.phases)
            lines.append(f"  phases    : " + ", ".join(f"{k}={v}" for k, v in ph.most_common()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SAGE picks .txt loader
# ---------------------------------------------------------------------------

def load_picks_txt(path: str | Path) -> CatalogData:
    """
    Load one or more SAGE picks .txt files.

    Parameters
    ----------
    path : str or Path
        Path to a single .txt file  **or**  a directory that will be
        searched recursively for ``sage_picks_*.txt`` files.

    Returns
    -------
    CatalogData
        All picks found in the file(s).

    Notes
    -----
    Expected row format (comma-separated, lines starting with # are skipped):
        Phase, RelTimeSec, Confidence, AbsoluteTime, SNR, Amplitude,
        Station, Polarity, PolarityProb
    Example:
        Pg,32640.180,0.937,2021-05-21 09:04:00.185000,23.796,892.73,YN.YSW03.00,N,0.000
    """
    path = Path(path)
    files: List[Path] = []

    if path.is_dir():
        # Collect all sage_picks_*.txt in the directory tree
        files = sorted(path.rglob("sage_picks_*.txt"))
        if not files:
            # Fallback: any .txt file that looks like a picks file
            files = sorted(path.rglob("*.txt"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    if not files:
        raise FileNotFoundError(f"No picks .txt files found under: {path}")

    data = CatalogData(source_file=str(path))

    for fpath in files:
        with open(fpath, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 7:
                    continue
                try:
                    phase = parts[0]
                    confidence = float(parts[2])
                    abs_time = datetime.strptime(parts[3], "%Y-%m-%d %H:%M:%S.%f")
                    snr = float(parts[4])
                    station = parts[6]

                    data.phases.append(phase)
                    data.times.append(abs_time)
                    data.confidences.append(confidence)
                    data.snrs.append(snr)
                    data.stations.append(station)
                except (ValueError, IndexError):
                    continue  # skip malformed rows

    if not data.times:
        raise ValueError(f"No valid picks found in: {path}")

    return data


# ---------------------------------------------------------------------------
# Generic catalog (CSV / JSON) loader
# ---------------------------------------------------------------------------

_TIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
]

_MAGNITUDE_COLS = ["magnitude", "mag", "ml", "mb", "ms", "mw", "m"]
_TIME_COLS      = ["time", "origin_time", "datetime", "date", "origintime"]
_LON_COLS       = ["longitude", "lon", "x"]
_LAT_COLS       = ["latitude",  "lat", "y"]
_DEPTH_COLS     = ["depth", "dep", "z"]


def _parse_time(s: str) -> Optional[datetime]:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def _find_col(headers: List[str], candidates: List[str]) -> Optional[int]:
    """Return index of first matching column header (case-insensitive)."""
    h_lower = [h.lower().strip() for h in headers]
    for name in candidates:
        if name in h_lower:
            return h_lower.index(name)
    return None


def load_catalog_csv(path: str | Path) -> CatalogData:
    """Load an earthquake catalog from a CSV file."""
    path = Path(path)
    data = CatalogData(source_file=str(path))

    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        # Skip comment lines
        lines = [l for l in fh if not l.startswith("#")]

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header row: {path}")

    headers = list(reader.fieldnames)
    i_time  = _find_col(headers, _TIME_COLS)
    i_mag   = _find_col(headers, _MAGNITUDE_COLS)
    i_lon   = _find_col(headers, _LON_COLS)
    i_lat   = _find_col(headers, _LAT_COLS)
    i_dep   = _find_col(headers, _DEPTH_COLS)

    for row in reader:
        vals = list(row.values())
        if i_time is not None:
            t = _parse_time(vals[i_time].strip())
            if t:
                data.times.append(t)
        if i_mag is not None:
            try:
                data.magnitudes.append(float(vals[i_mag]))
            except ValueError:
                pass
        if i_lon is not None:
            try:
                data.longitudes.append(float(vals[i_lon]))
            except ValueError:
                pass
        if i_lat is not None:
            try:
                data.latitudes.append(float(vals[i_lat]))
            except ValueError:
                pass
        if i_dep is not None:
            try:
                data.depths.append(float(vals[i_dep]))
            except ValueError:
                pass

    if not data.times and not data.magnitudes:
        raise ValueError(f"No usable data found in CSV: {path}")
    return data


def load_catalog_json(path: str | Path) -> CatalogData:
    """Load an earthquake catalog from a JSON file (list of dicts)."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)

    if isinstance(records, dict):
        # Try common wrappers: {"events": [...]}
        for key in ("events", "data", "catalog", "earthquakes"):
            if key in records and isinstance(records[key], list):
                records = records[key]
                break

    if not isinstance(records, list):
        raise ValueError(f"JSON root must be a list of dicts: {path}")

    data = CatalogData(source_file=str(path))

    def _get(d: dict, candidates: List[str]):
        for k in candidates:
            for key in d:
                if key.lower().strip() == k:
                    return d[key]
        return None

    for rec in records:
        if not isinstance(rec, dict):
            continue
        t_val = _get(rec, _TIME_COLS)
        if t_val:
            t = _parse_time(str(t_val))
            if t:
                data.times.append(t)
        m_val = _get(rec, _MAGNITUDE_COLS)
        if m_val is not None:
            try:
                data.magnitudes.append(float(m_val))
            except ValueError:
                pass
        lon_val = _get(rec, _LON_COLS)
        if lon_val is not None:
            try:
                data.longitudes.append(float(lon_val))
            except ValueError:
                pass
        lat_val = _get(rec, _LAT_COLS)
        if lat_val is not None:
            try:
                data.latitudes.append(float(lat_val))
            except ValueError:
                pass
        dep_val = _get(rec, _DEPTH_COLS)
        if dep_val is not None:
            try:
                data.depths.append(float(dep_val))
            except ValueError:
                pass

    if not data.times and not data.magnitudes:
        raise ValueError(f"No usable data found in JSON: {path}")
    return data


def load_catalog_file(path: str | Path) -> CatalogData:
    """
    Auto-detect format and load an earthquake catalog file.

    Supports: .csv, .txt (catalog format), .json
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        return load_catalog_json(path)
    elif suffix == ".csv":
        return load_catalog_csv(path)
    elif suffix == ".txt":
        # Decide: SAGE picks format or generic catalog?
        with open(path, encoding="utf-8", errors="replace") as fh:
            head = "".join(fh.readline() for _ in range(5))
        if "震相" in head or "相对时间" in head or re.search(r'^Pg|^Sg|^Pn|^Sn|^P,|^S,', head, re.M):
            return load_picks_txt(path)
        else:
            return load_catalog_csv(path)
    else:
        raise ValueError(f"Unsupported catalog format: {suffix} ({path})")
