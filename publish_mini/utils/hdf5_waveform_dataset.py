#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Disable HDF5 POSIX file locking before h5py initialises.
# Must be set before the first h5py import / H5open() call.
# Prevents h5py.File() from blocking indefinitely on lock acquisition
# (e.g. due to Spotlight, Time Machine, NFS, or a crashed prior run).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from obspy import Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Site, Station
from obspy.core.inventory.response import (
    CoefficientsTypeResponseStage,
    FIRResponseStage,
    InstrumentSensitivity,
    PolesZerosResponseStage,
    PolynomialResponseStage,
    Response,
    ResponseStage,
)


DEFAULT_LOCATION = "--"


def parse_time(t):
    return UTCDateTime(str(t))


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8", errors="ignore")
    return value


def normalize_location(location, default=DEFAULT_LOCATION):
    location = decode_attr(location)
    if location is None:
        return default
    location = str(location).strip()
    return location if location else default


def channel_suffix(channel):
    return str(channel)[-1].upper()


def channel_prefix(channel):
    ch = str(channel).upper()
    if len(ch) >= 3:
        return ch[:2]
    return ch[:-1]


def component_rank(channel):
    order = {
        "E": 0,
        "1": 0,
        "N": 1,
        "2": 1,
        "Z": 2,
        "3": 2,
    }
    return order.get(channel_suffix(channel), 99)


def has_three_components(channels):
    suffixes = {channel_suffix(ch) for ch in channels}
    return {"E", "N", "Z"}.issubset(suffixes) or {"1", "2", "3"}.issubset(suffixes)


def is_z_only_channels(channels):
    suffixes = {channel_suffix(ch) for ch in channels}
    return len(channels) == 1 and suffixes == {"Z"}


def get_attr(obj, name, default=None):
    if name in obj.attrs:
        return decode_attr(obj.attrs[name])
    return default


def get_float_attr(obj, name, default=np.nan):
    try:
        return float(get_attr(obj, name, default))
    except Exception:
        return float(default)


def get_bool_attr(obj, name, default=False):
    value = get_attr(obj, name, default)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes"]

    return bool(value)


def resolve_h5_files(h5_input):
    if isinstance(h5_input, (list, tuple)):
        files = []
        for item in h5_input:
            files.extend(resolve_h5_files(item))
        return sorted(set(files))

    h5_input = str(h5_input)
    p = Path(h5_input)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        files = sorted(str(x) for x in p.glob("*.h5"))
        files += sorted(str(x) for x in p.glob("*.hdf5"))
        return files

    files = sorted(glob.glob(h5_input))
    if files:
        return files

    raise FileNotFoundError(f"No HDF5 files found from input: {h5_input}")


def make_sample_key_from_index_item(item):
    """Build a stable sample key from index metadata only."""
    channels = item.get("channels", [])
    if not channels and item.get("channel", ""):
        channels = [item.get("channel", "")]
    channels = ",".join(str(x) for x in channels)

    return "|".join([
        str(item.get("h5_file", "")),
        str(item.get("year_id", "")),
        str(item.get("day_id", "")),
        str(item.get("station_id", "")),
        str(item.get("channel_family", item.get("channel", ""))),
        channels,
    ])


def make_sample_key_from_record(record):
    """Build the same sample key from one output JSONL record.

    Must produce a key identical to make_sample_key_from_index_item so that
    the resume scanner can match written records back to index entries.

    Two known pitfalls:
    1. station_id: the pick record stores it inside station_info (and now also
       at the top level after the station_id fix).  We try both places.
    2. Z-only replicated channels: _getitem_three stores ["EHZ","EHZ","EHZ"]
       in channels_out when replicate_z_only=True, but the index entry stores
       the raw ["EHZ"] list.  Deduplicate before joining to match the index key.
    """
    channels = record.get("channels") or []  # handle None / missing
    # Deduplicate for Z-only replicated samples so the key matches the index
    # entry, which was built from the raw (non-replicated) channel list.
    if record.get("z_only_replicated", False):
        seen: set = set()
        channels = [ch for ch in channels if not (ch in seen or seen.add(ch))]
    channels = ",".join(str(x) for x in channels)

    station_info = record.get("station_info", {}) or {}
    station_id = station_info.get("station_id", record.get("station_id", ""))

    return "|".join([
        str(record.get("h5_file", "")),
        str(record.get("year_id", "")),
        str(record.get("day_id", "")),
        str(station_id),
        str(record.get("channel_family", record.get("channel", ""))),
        channels,
    ])


def _parse_jsonl_chunk(path, start_byte, end_byte, record_type):
    """Parse one byte-range chunk of a JSONL file and return finished sample keys.

    Called from load_finished_sample_keys via ThreadPoolExecutor. Python's
    built-in json.loads (C extension) releases the GIL during parsing, so
    multiple threads genuinely run in parallel for large files.

    Both ``phase_pick`` and ``error`` records are treated as "finished" so that
    samples which errored on a previous run are skipped on resume rather than
    retried indefinitely.  Error records store the pre-built key in the
    ``sample_key`` field; phase_pick records reconstruct the key from individual
    fields via make_sample_key_from_record.
    """
    keys = set()
    total = 0
    bad = 0
    with open(path, "rb") as f:
        f.seek(start_byte)
        if start_byte > 0:
            f.readline()          # discard partial line at chunk boundary
        while f.tell() < end_byte:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except Exception:
                bad += 1
                continue
            rt = record.get("record_type", "")
            # Accept phase_pick records (primary output), error records
            # (inference failures), and no_pick records (successful runs
            # with zero detections above min_confidence).  All three mean
            # the station-day was fully processed and should be skipped on
            # resume.  error and no_pick records carry a pre-built
            # sample_key field; phase_pick records are reconstructed below.
            if rt in ("error", "no_pick"):
                key = record.get("sample_key", "")
                if key and key.strip("|"):
                    keys.add(key)
                continue
            if record_type and rt != record_type:
                continue
            try:
                key = make_sample_key_from_record(record)
                if key.strip("|"):
                    keys.add(key)
            except Exception:
                bad += 1
                continue
    return keys, total, bad


def load_finished_sample_keys(jsonl_file, record_type="phase_pick", num_threads=4):
    """Return (finished_keys_set, total_lines, bad_lines) for resume filtering.

    The JSONL file is split into ``num_threads`` byte-range chunks and parsed
    concurrently via ThreadPoolExecutor.  Python's json.loads (C extension)
    releases the GIL during parsing, so multiple threads genuinely overlap.

    Sample keys are reconstructed directly from JSONL record fields (h5_file,
    year_id, day_id, station_id, channel_family, channels) — no companion
    index file is needed.
    """
    jsonl_file = Path(jsonl_file)
    finished = set()

    if not jsonl_file.exists():
        return finished, 0, 0

    file_size = jsonl_file.stat().st_size
    if file_size == 0:
        return finished, 0, 0

    # Don't spin up more threads than make sense for small files.
    n = max(1, min(num_threads, file_size // (256 * 1024)))
    chunk_size = file_size // n
    chunks = [
        (i * chunk_size, (i + 1) * chunk_size if i < n - 1 else file_size)
        for i in range(n)
    ]

    def _parse(args):
        return _parse_jsonl_chunk(str(jsonl_file), args[0], args[1], record_type)

    if n == 1:
        results = [_parse(chunks[0])]
    else:
        with ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(_parse, chunks))

    total_lines = 0
    bad_lines = 0
    for keys, count, bad in results:
        finished.update(keys)
        total_lines += count
        bad_lines += bad

    return finished, total_lines, bad_lines


def fill_segments_to_array(
    segments,
    fill_value=0.0,
    dtype=np.float32,
    use_overlap_mask=True,
    max_duration_sec=90000.0,
):
    if len(segments) == 0:
        return None, None, None, None

    segments = sorted(segments, key=lambda x: x["starttime"])

    sampling_rate = float(segments[0]["sampling_rate"])
    global_start = min(s["starttime"] for s in segments)
    global_end = max(s["endtime"] for s in segments)

    duration_sec = float(global_end - global_start)
    npts = int(round(duration_sec * sampling_rate)) + 1

    if (not np.isfinite(sampling_rate)) or sampling_rate <= 0:
        raise ValueError(f"Invalid sampling_rate={sampling_rate}")
    if npts <= 0:
        raise ValueError(
            f"Invalid npts={npts}, start={global_start}, end={global_end}, sr={sampling_rate}"
        )
    if max_duration_sec and max_duration_sec > 0 and duration_sec > float(max_duration_sec):
        raise ValueError(
            f"Abnormal segment time span: duration={duration_sec:.3f}s, "
            f"npts={npts}, sr={sampling_rate}, start={global_start}, end={global_end}"
        )

    data = np.full(npts, fill_value, dtype=dtype)
    filled = np.zeros(npts, dtype=bool) if use_overlap_mask else None

    for seg in segments:
        seg_data = seg["data"].astype(dtype, copy=False)

        i0 = int(round((seg["starttime"] - global_start) * sampling_rate))
        i1 = i0 + len(seg_data)

        if i0 < 0:
            seg_data = seg_data[-i0:]
            i0 = 0

        if i1 > npts:
            seg_data = seg_data[: npts - i0]
            i1 = npts

        if i0 >= i1:
            continue

        target = slice(i0, i1)
        seg_data = seg_data[: i1 - i0]

        if use_overlap_mask:
            mask = ~filled[target]
            data_view = data[target]
            data_view[mask] = seg_data[mask]
            filled[target][mask] = True
        else:
            data[target] = seg_data

    return data, global_start, global_end, sampling_rate


def resample_1d_array(x, original_sr, target_sr=None, dtype=np.float32):
    """
    线性插值重采样，不依赖 scipy。

    x: [T]
    original_sr: 原始采样率
    target_sr: 目标采样率；None 表示不重采样
    """
    if x is None:
        return x, original_sr

    x = np.asarray(x, dtype=dtype)

    if target_sr is None:
        return x, float(original_sr)

    original_sr = float(original_sr)
    target_sr = float(target_sr)

    if not np.isfinite(original_sr) or original_sr <= 0:
        return x, original_sr

    if abs(original_sr - target_sr) < 1e-6:
        return x, original_sr

    if len(x) <= 1:
        return x.astype(dtype, copy=False), target_sr

    duration = (len(x) - 1) / original_sr
    new_npts = int(round(duration * target_sr)) + 1

    old_t = np.arange(len(x), dtype=np.float64) / original_sr
    new_t = np.arange(new_npts, dtype=np.float64) / target_sr

    y = np.interp(new_t, old_t, x).astype(dtype)

    return y, target_sr


def resample_2d_array(x, original_sr, target_sr=None, dtype=np.float32):
    """
    x: [T, C]
    """
    x = np.asarray(x, dtype=dtype)

    if target_sr is None:
        return x, float(original_sr)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [T, C], got shape={x.shape}")

    ys = []
    current_sr = original_sr

    for i in range(x.shape[1]):
        y, current_sr = resample_1d_array(
            x[:, i],
            original_sr=original_sr,
            target_sr=target_sr,
            dtype=dtype,
        )
        ys.append(y)

    min_len = min(len(y) for y in ys)
    ys = [y[:min_len] for y in ys]

    return np.stack(ys, axis=1).astype(dtype), current_sr


def utc_or_none(value):
    if value is None:
        return None
    value = decode_attr(value)
    value = "" if value is None else str(value).strip()
    if not value or value.lower() in {"none", "null", "nan"}:
        return None
    try:
        return UTCDateTime(value)
    except Exception:
        return None


def finite_float_or_none(value):
    try:
        value = float(value)
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def finite_float(value, default=0.0):
    out = finite_float_or_none(value)
    return float(default) if out is None else float(out)


def int_or_none(value):
    try:
        return int(value)
    except Exception:
        return None


def complex_list_from_json(items):
    out = []
    for item in items or []:
        if isinstance(item, dict):
            out.append(complex(float(item.get("real", 0.0)), float(item.get("imag", 0.0))))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append(complex(float(item[0]), float(item[1])))
        else:
            out.append(complex(item))
    return out


def _stage_common_kwargs(stage):
    return {
        "stage_sequence_number": int(stage.get("stage_sequence_number", 0)),
        "stage_gain": finite_float(stage.get("stage_gain", 1.0), 1.0),
        "stage_gain_frequency": finite_float(stage.get("stage_gain_frequency", 0.0), 0.0),
        "input_units": stage.get("input_units") or "",
        "output_units": stage.get("output_units") or "",
        "input_units_description": stage.get("input_units_description"),
        "output_units_description": stage.get("output_units_description"),
        "decimation_input_sample_rate": finite_float_or_none(
            stage.get("decimation_input_sample_rate")
        ),
        "decimation_factor": int_or_none(stage.get("decimation_factor")),
        "decimation_offset": int_or_none(stage.get("decimation_offset")),
        "decimation_delay": finite_float_or_none(stage.get("decimation_delay")),
        "decimation_correction": finite_float_or_none(stage.get("decimation_correction")),
    }


def response_stage_from_json(stage):
    stage_type = stage.get("type", "ResponseStage")
    common = _stage_common_kwargs(stage)

    if stage_type == "PolesZerosResponseStage":
        return PolesZerosResponseStage(
            **common,
            pz_transfer_function_type=stage.get(
                "pz_transfer_function_type",
                "LAPLACE (RADIANS/SECOND)",
            ),
            normalization_frequency=finite_float(
                stage.get("normalization_frequency", common["stage_gain_frequency"]),
                common["stage_gain_frequency"],
            ),
            normalization_factor=finite_float(stage.get("normalization_factor", 1.0), 1.0),
            zeros=complex_list_from_json(stage.get("zeros", [])),
            poles=complex_list_from_json(stage.get("poles", [])),
        )

    if stage_type == "CoefficientsTypeResponseStage":
        return CoefficientsTypeResponseStage(
            **common,
            cf_transfer_function_type=stage.get("cf_transfer_function_type", "DIGITAL"),
            numerator=list(stage.get("numerator", stage.get("numerators", [])) or []),
            denominator=list(stage.get("denominator", stage.get("denominators", [])) or []),
        )

    if stage_type == "FIRResponseStage":
        return FIRResponseStage(
            **common,
            symmetry=stage.get("symmetry", "NONE"),
            coefficients=list(stage.get("coefficients", []) or []),
        )

    if stage_type == "PolynomialResponseStage":
        return PolynomialResponseStage(
            **common,
            frequency_lower_bound=finite_float(stage.get("frequency_lower_bound", 0.0), 0.0),
            frequency_upper_bound=finite_float(stage.get("frequency_upper_bound", 0.0), 0.0),
            approximation_lower_bound=finite_float(
                stage.get("approximation_lower_bound", 0.0),
                0.0,
            ),
            approximation_upper_bound=finite_float(
                stage.get("approximation_upper_bound", 0.0),
                0.0,
            ),
            maximum_error=finite_float(stage.get("maximum_error", 0.0), 0.0),
            coefficients=list(stage.get("coefficients", []) or []),
            approximation_type=stage.get("approximation_type", "MACLAURIN"),
        )

    return ResponseStage(**common)


def response_from_json_record(record):
    sensitivity = None
    sens = record.get("instrument_sensitivity") or {}
    if sens:
        sensitivity = InstrumentSensitivity(
            value=finite_float(sens.get("value", 1.0), 1.0),
            frequency=finite_float(sens.get("frequency", 0.0), 0.0),
            input_units=sens.get("input_units") or "",
            output_units=sens.get("output_units") or "",
            input_units_description=sens.get("input_units_description"),
            output_units_description=sens.get("output_units_description"),
        )

    stages = [response_stage_from_json(stage) for stage in record.get("stages", []) or []]
    return Response(
        resource_id=record.get("response_id"),
        instrument_sensitivity=sensitivity,
        response_stages=stages,
    )


def inventory_from_response_record(record, response):
    start_date = utc_or_none(record.get("epoch_start"))
    end_date = utc_or_none(record.get("epoch_end"))
    latitude = finite_float(record.get("latitude", 0.0), 0.0)
    longitude = finite_float(record.get("longitude", 0.0), 0.0)
    elevation = finite_float(record.get("elevation_m", 0.0), 0.0)

    channel = Channel(
        code=str(record.get("channel", "")),
        location_code=normalize_location(record.get("location", DEFAULT_LOCATION), DEFAULT_LOCATION),
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        depth=finite_float(record.get("depth_m", 0.0), 0.0),
        azimuth=finite_float_or_none(record.get("azimuth")),
        dip=finite_float_or_none(record.get("dip")),
        sample_rate=finite_float_or_none(record.get("sample_rate")),
        start_date=start_date,
        end_date=end_date,
        response=response,
    )
    station = Station(
        code=str(record.get("station", "")),
        latitude=latitude,
        longitude=longitude,
        elevation=elevation,
        site=Site(name=str(record.get("station", ""))),
        channels=[channel],
        start_date=start_date,
        end_date=end_date,
    )
    network = Network(code=str(record.get("network", "")), stations=[station])
    return Inventory(networks=[network], source="SeismicX-Cont response JSON")


def load_response_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    by_key = {}
    by_id = {}
    responses = obj.get("responses", [])

    for record in responses:
        key = (
            str(record.get("network", "")),
            str(record.get("station", "")),
            normalize_location(record.get("location", DEFAULT_LOCATION), DEFAULT_LOCATION),
            str(record.get("channel", "")),
        )
        by_key.setdefault(key, []).append(record)
        response_id = record.get("response_id")
        if response_id:
            by_id[str(response_id)] = record

    for records in by_key.values():
        records.sort(key=lambda item: str(item.get("epoch_start", "")))

    return {
        "path": str(path),
        "schema": obj.get("schema", ""),
        "source": obj.get("source", {}),
        "summary": obj.get("summary", {}),
        "responses_by_key": by_key,
        "responses_by_id": by_id,
        "response_count": len(responses),
    }


def response_record_matches_time(record, starttime, endtime=None):
    epoch_start = utc_or_none(record.get("epoch_start"))
    epoch_end = utc_or_none(record.get("epoch_end"))

    if starttime is None:
        return True
    if epoch_start is not None and starttime < epoch_start:
        return False
    if epoch_end is not None and starttime > epoch_end:
        return False
    return True


def next_pow_2(n):
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def apply_response_spectrum(data, sampling_rate, response, output="VEL"):
    data = np.asarray(data, dtype=np.float64)
    npts = int(data.shape[0])
    if npts <= 1:
        return data

    nfft = next_pow_2(npts)
    delta = 1.0 / float(sampling_rate)
    spectrum = np.fft.rfft(data, n=nfft)
    resp, _freq = response.get_evalresp_response(
        t_samp=delta,
        nfft=nfft,
        output=output,
    )
    spectrum *= resp
    return np.fft.irfft(spectrum, n=nfft)[:npts]


def get_position_from_segments(segments):
    for seg in segments:
        if seg.get("location_available", False):
            return {
                "longitude": seg.get("longitude", np.nan),
                "latitude": seg.get("latitude", np.nan),
                "elevation": seg.get("elevation", np.nan),
                "location_available": True,
                "location_source": seg.get("location_source", ""),
                "position_match_mode": seg.get("position_match_mode", ""),
                "position_is_fallback": seg.get("position_is_fallback", False),
                "station_position_starttime": seg.get("station_position_starttime", ""),
                "station_position_endtime": seg.get("station_position_endtime", ""),
            }

    return {
        "longitude": np.nan,
        "latitude": np.nan,
        "elevation": np.nan,
        "location_available": False,
        "location_source": "default_nan_no_station_record",
        "position_match_mode": "default_nan_no_station_record",
        "position_is_fallback": False,
        "station_position_starttime": "",
        "station_position_endtime": "",
    }


class HDF5WaveformDataset(Dataset):
    """
    HDF5 连续波形 dataloader。

    默认行为：
        1. mode="three"
        2. 默认保留 HH/BH/EH/HN 三分量
        3. 默认保留 EHZ 单通道
        4. 默认将单通道 Z 复制为 [Z, Z, Z]
        5. 可选重采样到 target_sampling_rate

    mode:
        single:
            每个 channel 一个样本，返回 waveform: [T]

        three:
            每个通道族一个样本，返回 waveform: [T, 3]
            分量顺序为 E/N/Z 或 1/2/3

        multi:
            每个通道族一个样本，返回 waveform: [T, C]
    """

    def __init__(
        self,
        h5_file,
        mode="three",
        fill_value=0.0,
        dtype=np.float32,
        default_location=DEFAULT_LOCATION,
        allowed_families=("HH", "BH", "EH", "HN"),
        allowed_z_only_channels=("EHZ",),
        allow_z_only=True,
        replicate_z_only=True,
        target_sampling_rate=None,
        skip_sample_keys=None,
        skip_jsonl=None,
        skip_record_type="phase_pick",
        keep_h5_open=True,
        include_segments_metadata=True,
        use_overlap_mask=True,
        h5_rdcc_nbytes=8 * 1024 * 1024,
        max_duration_sec=90000.0,
        instrument_response_json=None,
        remove_instrument_response=False,
        response_output="VEL",
        response_pre_filt=None,
        response_water_level=60,
        response_zero_mean=True,
        response_taper=True,
        response_taper_fraction=0.05,
        response_error_behavior="raise",
        simulate_instrument_response=False,
        simulation_response_json=None,
        simulation_response_id=None,
        simulation_response_selector=None,
        simulation_paz=None,
        simulation_output=None,
        simulation_sensitivity=True,
    ):
        assert mode in ["single", "three", "multi"]
        assert response_error_behavior in ["raise", "warn", "skip"]

        self.h5_files = resolve_h5_files(h5_file)
        self.mode = mode
        self.fill_value = fill_value
        self.dtype = dtype
        self.default_location = default_location

        self.allowed_families = tuple(x.upper() for x in allowed_families)
        self.allowed_z_only_channels = tuple(x.upper() for x in allowed_z_only_channels)
        self.allow_z_only = bool(allow_z_only)
        self.replicate_z_only = bool(replicate_z_only)
        self.target_sampling_rate = target_sampling_rate
        self.keep_h5_open = bool(keep_h5_open)
        self.include_segments_metadata = bool(include_segments_metadata)
        self.use_overlap_mask = bool(use_overlap_mask)
        # HDF5 raw-data chunk cache per open file handle.  h5py default is 1 MB.
        # 8 MB is a good balance for single-pass inference: large enough to avoid
        # re-reading chunks within one channel, small enough not to waste RAM.
        # (Training / repeated-access workloads can benefit from a larger value.)
        self.h5_rdcc_nbytes = max(0, int(h5_rdcc_nbytes))
        # Hard safety limit for one consolidated channel waveform.
        # Default 90000 s = 25 h, enough for one UTC day with small tolerance.
        self.max_duration_sec = float(max_duration_sec) if max_duration_sec is not None else 0.0
        self.instrument_response_json = (
            str(instrument_response_json) if instrument_response_json is not None else None
        )
        self.remove_instrument_response = bool(remove_instrument_response)
        self.response_output = str(response_output).upper() if response_output is not None else "VEL"
        self.response_pre_filt = (
            tuple(float(x) for x in response_pre_filt)
            if response_pre_filt is not None else None
        )
        self.response_water_level = response_water_level
        self.response_zero_mean = bool(response_zero_mean)
        self.response_taper = bool(response_taper)
        self.response_taper_fraction = float(response_taper_fraction)
        self.response_error_behavior = response_error_behavior
        self.simulate_instrument_response = bool(simulate_instrument_response)
        self.simulation_response_json = (
            str(simulation_response_json) if simulation_response_json is not None else None
        )
        self.simulation_response_id = (
            str(simulation_response_id) if simulation_response_id is not None else None
        )
        self.simulation_response_selector = dict(simulation_response_selector or {})
        self.simulation_paz = simulation_paz
        self.simulation_output = (
            str(simulation_output).upper()
            if simulation_output is not None else self.response_output
        )
        self.simulation_sensitivity = bool(simulation_sensitivity)

        if self.remove_instrument_response and self.instrument_response_json is None:
            raise ValueError(
                "instrument_response_json is required when remove_instrument_response is enabled."
            )
        if self.simulate_instrument_response and (
            self.simulation_paz is None
            and self.simulation_response_id is None
            and not self.simulation_response_selector
            and self.simulation_response_json is None
        ):
            raise ValueError(
                "simulate_instrument_response=True requires simulation_paz, "
                "simulation_response_id, simulation_response_selector, or "
                "simulation_response_json."
            )
        if self.simulate_instrument_response and (
            self.simulation_response_json is None
            and self.simulation_paz is None
            and self.instrument_response_json is None
        ):
            raise ValueError(
                "simulation_response_json is required when simulation selects a "
                "response record and instrument_response_json is not set."
            )

        # Per-process HDF5 handle cache. DataLoader workers get their own cache.
        self._h5_cache = {}
        self._response_store = None
        self._simulation_response_store = None
        self._response_object_cache = {}
        self._inventory_cache = {}
        self._simulation_response_record = None
        self._simulation_response_object = None

        self.index = []
        self._build_index()

        # Resume filtering happens on metadata-only index entries. Removed samples
        # will never enter __getitem__, so their waveform data are not read.
        self.finished_sample_keys = set(skip_sample_keys or [])
        self.skip_jsonl_stats = None
        if skip_jsonl is not None:
            keys, total_lines, bad_lines = load_finished_sample_keys(
                skip_jsonl,
                record_type=skip_record_type,
            )
            self.finished_sample_keys.update(keys)
            self.skip_jsonl_stats = {
                "jsonl_file": str(skip_jsonl),
                "total_lines": total_lines,
                "bad_lines": bad_lines,
                "finished_keys": len(keys),
            }

        self.original_index_size = len(self.index)
        self.filtered_index_size = 0
        if self.finished_sample_keys:
            self.filter_index_by_finished_keys(self.finished_sample_keys)

    def __getstate__(self):
        """Do not pickle open h5py handles into DataLoader workers."""
        state = self.__dict__.copy()
        state["_h5_cache"] = {}
        state["_response_store"] = None
        state["_simulation_response_store"] = None
        state["_response_object_cache"] = {}
        state["_inventory_cache"] = {}
        state["_simulation_response_record"] = None
        state["_simulation_response_object"] = None
        return state

    def __setstate__(self, state):
        """Restore state in a worker process with a fresh, empty handle cache.

        Called by pickle when DataLoader workers deserialize the dataset. Ensures
        _h5_cache is always empty in the new process regardless of start method
        (spawn, forkserver, or fork), so each worker opens its own HDF5 handles
        lazily on the first __getitem__ call.
        """
        self.__dict__.update(state)
        self._h5_cache = {}  # always start clean in every process
        self._response_object_cache = {}
        self._inventory_cache = {}

    def close(self):
        """Close all cached HDF5 handles and free their chunk + metadata caches."""
        cache = getattr(self, "_h5_cache", {})
        for h5 in list(cache.values()):
            try:
                h5.close()
            except Exception:
                pass
        cache.clear()

    def flush_h5_cache(self):
        """Close the current HDF5 handle and clear _h5_cache.

        Call this periodically (e.g. every N samples) to flush the per-file
        HDF5 metadata cache that the C library accumulates as it visits
        dataset groups.  The handle is reopened lazily on the next
        __getitem__ call, so this is safe to call at any time between samples.
        """
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _get_h5_handle(self, h5_file):
        rdcc = getattr(self, "h5_rdcc_nbytes", 8 * 1024 * 1024)

        if not self.keep_h5_open:
            return h5py.File(h5_file, "r", rdcc_nbytes=rdcc)

        if not hasattr(self, "_h5_cache"):
            self._h5_cache = {}

        h5 = self._h5_cache.get(h5_file)
        if h5 is None:
            # Inference is a single pass over sorted files: when a new file is
            # requested the previous file will not be accessed again.  Close all
            # stale handles immediately so their HDF5 chunk caches (rdcc_nbytes
            # each) are released rather than accumulating for the life of the
            # process.  Keeping only the current file's handle preserves the
            # within-file I/O benefit while bounding cache memory to O(1) files.
            for _f, _h in list(self._h5_cache.items()):
                if _f != h5_file:
                    try:
                        _h.close()
                    except Exception:
                        pass
                    self._h5_cache.pop(_f, None)
            h5 = h5py.File(h5_file, "r", rdcc_nbytes=rdcc)
            self._h5_cache[h5_file] = h5
        return h5

    def sample_key(self, idx_or_item):
        if isinstance(idx_or_item, int):
            item = self.index[idx_or_item]
        else:
            item = idx_or_item
        return make_sample_key_from_index_item(item)

    def filter_index_by_finished_keys(self, finished_keys):
        """Remove finished samples from self.index before waveform reading."""
        finished_keys = set(finished_keys or [])
        if not finished_keys:
            return 0

        old_n = len(self.index)
        self.index = [
            item for item in self.index
            if make_sample_key_from_index_item(item) not in finished_keys
        ]
        removed = old_n - len(self.index)
        self.filtered_index_size += removed
        return removed

    def filter_index_by_jsonl(self, jsonl_file, record_type="phase_pick"):
        keys, total_lines, bad_lines = load_finished_sample_keys(
            jsonl_file,
            record_type=record_type,
        )
        removed = self.filter_index_by_finished_keys(keys)
        self.skip_jsonl_stats = {
            "jsonl_file": str(jsonl_file),
            "total_lines": total_lines,
            "bad_lines": bad_lines,
            "finished_keys": len(keys),
            "removed_from_index": removed,
        }
        return removed, keys, total_lines, bad_lines

    def _ensure_response_store(self):
        if self._response_store is None:
            if self.instrument_response_json is None:
                raise ValueError("instrument_response_json is not configured")
            self._response_store = load_response_json(self.instrument_response_json)
        return self._response_store

    def _ensure_simulation_response_store(self):
        if self._simulation_response_store is None:
            path = self.simulation_response_json or self.instrument_response_json
            if path is None:
                raise ValueError("No simulation response JSON is configured")
            self._simulation_response_store = load_response_json(path)
        return self._simulation_response_store

    def _get_response_object(self, record):
        response_id = str(record.get("response_id", ""))
        cache_key = response_id or id(record)
        if cache_key not in self._response_object_cache:
            self._response_object_cache[cache_key] = response_from_json_record(record)
        return self._response_object_cache[cache_key]

    def _get_inventory(self, record):
        response_id = str(record.get("response_id", ""))
        cache_key = response_id or id(record)
        if cache_key not in self._inventory_cache:
            response = self._get_response_object(record)
            self._inventory_cache[cache_key] = inventory_from_response_record(record, response)
        return self._inventory_cache[cache_key]

    def _find_response_record(self, network, station, location, channel, starttime, endtime=None):
        store = self._ensure_response_store()
        key = (
            str(network),
            str(station),
            normalize_location(location, self.default_location),
            str(channel),
        )
        candidates = store["responses_by_key"].get(key, [])
        for record in candidates:
            if response_record_matches_time(record, starttime, endtime):
                return record
        return None

    def _select_simulation_response_record(self):
        if self._simulation_response_record is not None:
            return self._simulation_response_record

        store = self._ensure_simulation_response_store()
        record = None

        if self.simulation_response_id:
            record = store["responses_by_id"].get(self.simulation_response_id)
            if record is None:
                raise KeyError(
                    f"simulation_response_id not found: {self.simulation_response_id}"
                )
        elif self.simulation_response_selector:
            sel = self.simulation_response_selector
            key = (
                str(sel.get("network", "")),
                str(sel.get("station", "")),
                normalize_location(sel.get("location", self.default_location), self.default_location),
                str(sel.get("channel", "")),
            )
            starttime = utc_or_none(sel.get("time")) or utc_or_none(sel.get("starttime"))
            candidates = store["responses_by_key"].get(key, [])
            for item in candidates:
                if response_record_matches_time(item, starttime):
                    record = item
                    break
            if record is None:
                raise KeyError(f"simulation_response_selector did not match any response: {sel}")
        else:
            records_by_id = store["responses_by_id"]
            if len(records_by_id) != 1:
                raise ValueError(
                    "simulation_response_json must contain exactly one response unless "
                    "simulation_response_id or simulation_response_selector is provided."
                )
            record = next(iter(records_by_id.values()))

        self._simulation_response_record = record
        self._simulation_response_object = self._get_response_object(record)
        return record

    def _handle_response_error(self, message):
        if self.response_error_behavior == "raise":
            raise RuntimeError(message)
        if self.response_error_behavior == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return None

    def _apply_instrument_processing(
        self,
        waveform,
        segments,
        channel,
        starttime,
        endtime,
        sampling_rate,
    ):
        metadata = {
            "remove_instrument_response": self.remove_instrument_response,
            "simulate_instrument_response": self.simulate_instrument_response,
            "response_output": self.response_output,
            "simulation_output": self.simulation_output,
            "response_id": "",
            "response_epoch_start": "",
            "response_epoch_end": "",
            "simulation_response_id": "",
            "error": "",
            "processed": False,
        }

        if waveform is None or len(waveform) == 0:
            return waveform, metadata
        if not self.remove_instrument_response and not self.simulate_instrument_response:
            return waveform, metadata

        first_segment = segments[0] if segments else {}
        network = first_segment.get("network", "")
        station = first_segment.get("station", "")
        location = first_segment.get("location", self.default_location)
        channel = first_segment.get("channel", channel)

        try:
            trace = Trace(
                data=np.asarray(waveform, dtype=np.float64),
                header={
                    "network": str(network),
                    "station": str(station),
                    "location": normalize_location(location, self.default_location),
                    "channel": str(channel),
                    "starttime": starttime,
                    "sampling_rate": float(sampling_rate),
                },
            )

            if self.remove_instrument_response:
                record = self._find_response_record(
                    network,
                    station,
                    location,
                    channel,
                    starttime,
                    endtime=endtime,
                )
                if record is None:
                    key = ".".join([
                        str(network),
                        str(station),
                        normalize_location(location, self.default_location),
                        str(channel),
                    ])
                    raise KeyError(f"No response found for {key} at {starttime}")

                metadata.update(
                    {
                        "response_id": record.get("response_id", ""),
                        "response_epoch_start": record.get("epoch_start", ""),
                        "response_epoch_end": record.get("epoch_end", ""),
                    }
                )
                trace.remove_response(
                    inventory=self._get_inventory(record),
                    output=self.response_output,
                    water_level=self.response_water_level,
                    pre_filt=self.response_pre_filt,
                    zero_mean=self.response_zero_mean,
                    taper=self.response_taper,
                    taper_fraction=self.response_taper_fraction,
                )

            if self.simulate_instrument_response:
                if self.simulation_paz is not None:
                    trace.simulate(
                        paz_remove=None,
                        paz_simulate=self.simulation_paz,
                        remove_sensitivity=False,
                        simulate_sensitivity=self.simulation_sensitivity,
                    )
                    metadata["simulation_response_id"] = "simulation_paz"
                else:
                    sim_record = self._select_simulation_response_record()
                    sim_response = self._simulation_response_object
                    trace.data = apply_response_spectrum(
                        trace.data,
                        sampling_rate=trace.stats.sampling_rate,
                        response=sim_response,
                        output=self.simulation_output,
                    )
                    metadata["simulation_response_id"] = sim_record.get("response_id", "")

            metadata["processed"] = True
            return np.asarray(trace.data, dtype=self.dtype), metadata
        except Exception as exc:
            metadata["error"] = str(exc)
            self._handle_response_error(str(exc))
            return np.asarray(waveform, dtype=self.dtype), metadata

    def _is_allowed_channel(self, channel):
        ch = str(channel).upper()
        prefix = channel_prefix(ch)

        if prefix in self.allowed_families:
            return True

        if self.allow_z_only and ch in self.allowed_z_only_channels:
            return True

        return False

    def _is_allowed_family_sample(self, prefix, family_channels):
        prefix = str(prefix).upper()
        family_channels = [str(x).upper() for x in family_channels]

        if prefix not in self.allowed_families:
            if not any(ch in self.allowed_z_only_channels for ch in family_channels):
                return False

        if self.mode == "multi":
            return True

        if has_three_components(family_channels):
            return True

        if self.allow_z_only:
            z_only = [
                ch for ch in family_channels
                if ch in self.allowed_z_only_channels
            ]
            return len(z_only) > 0

        return False

    def _build_index(self):
        for h5_file in self.h5_files:
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

                            if "waveform" not in station_grp:
                                continue

                            waveform_grp = station_grp["waveform"]
                            channels = sorted(
                                ch for ch in waveform_grp.keys()
                                if self._is_allowed_channel(ch)
                            )

                            if len(channels) == 0:
                                continue

                            if self.mode == "single":
                                for cha in channels:
                                    self.index.append(
                                        {
                                            "h5_file": h5_file,
                                            "year_id": year_id,
                                            "day_id": day_id,
                                            "station_id": station_id,
                                            "channel": cha,
                                        }
                                    )
                                continue

                            families = {}
                            for cha in channels:
                                prefix = channel_prefix(cha)
                                families.setdefault(prefix, []).append(cha)

                            for prefix, family_channels in families.items():
                                family_channels = sorted(
                                    family_channels,
                                    key=component_rank,
                                )

                                if not self._is_allowed_family_sample(prefix, family_channels):
                                    continue

                                self.index.append(
                                    {
                                        "h5_file": h5_file,
                                        "year_id": year_id,
                                        "day_id": day_id,
                                        "station_id": station_id,
                                        "channel_family": prefix,
                                        "channels": family_channels,
                                    }
                                )

    def __len__(self):
        return len(self.index)

    def _get_station_group(self, h5, year_id, day_id, station_id):
        return h5[year_id][day_id]["stations"][station_id]

    def _read_position_history(self, station_grp):
        if "position_history" not in station_grp:
            return []

        pos_grp = station_grp["position_history"]
        out = []

        for key in sorted(pos_grp.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            item = pos_grp[key]

            out.append(
                {
                    "network": get_attr(item, "network", ""),
                    "station": get_attr(item, "station", ""),
                    "location": normalize_location(
                        get_attr(item, "location", self.default_location),
                        self.default_location,
                    ),
                    "longitude": get_float_attr(item, "longitude", np.nan),
                    "latitude": get_float_attr(item, "latitude", np.nan),
                    "elevation": get_float_attr(item, "elevation", np.nan),
                    "starttime": get_attr(item, "starttime", ""),
                    "endtime": get_attr(item, "endtime", ""),
                }
            )

        return out

    def _read_station_attrs(self, station_grp):
        location = normalize_location(
            get_attr(station_grp, "location", self.default_location),
            self.default_location,
        )

        return {
            "station_id": get_attr(station_grp, "station_id", ""),
            "network": get_attr(station_grp, "network", ""),
            "station": get_attr(station_grp, "station", ""),
            "location": location,
            "location_is_default": get_bool_attr(
                station_grp,
                "location_is_default",
                location == self.default_location,
            ),
            "longitude": get_float_attr(station_grp, "longitude", np.nan),
            "latitude": get_float_attr(station_grp, "latitude", np.nan),
            "elevation": get_float_attr(station_grp, "elevation", np.nan),
            "location_available": get_bool_attr(station_grp, "location_available", False),
            "location_source": get_attr(station_grp, "location_source", ""),
            "position_match_mode": get_attr(station_grp, "position_match_mode", ""),
            "position_is_fallback": get_bool_attr(station_grp, "position_is_fallback", False),
            "station_position_starttime": get_attr(station_grp, "station_position_starttime", ""),
            "station_position_endtime": get_attr(station_grp, "station_position_endtime", ""),
            "instrument_time_range_start": get_attr(station_grp, "instrument_time_range_start", ""),
            "instrument_time_range_end": get_attr(station_grp, "instrument_time_range_end", ""),
            "position_history": self._read_position_history(station_grp),
        }

    def _read_channel_attrs(self, channel_grp):
        return {
            "channel": get_attr(channel_grp, "channel", ""),
            "segment_count": int(get_attr(channel_grp, "segment_count", 0)),
            "starttime": get_attr(channel_grp, "starttime", ""),
            "endtime": get_attr(channel_grp, "endtime", ""),
            "longitude": get_float_attr(channel_grp, "longitude", np.nan),
            "latitude": get_float_attr(channel_grp, "latitude", np.nan),
            "elevation": get_float_attr(channel_grp, "elevation", np.nan),
            "location_available": get_bool_attr(channel_grp, "location_available", False),
            "location_source": get_attr(channel_grp, "location_source", ""),
            "position_match_mode": get_attr(channel_grp, "position_match_mode", ""),
            "position_is_fallback": get_bool_attr(channel_grp, "position_is_fallback", False),
            "station_position_starttime": get_attr(channel_grp, "station_position_starttime", ""),
            "station_position_endtime": get_attr(channel_grp, "station_position_endtime", ""),
        }

    def _read_channel_segments(self, h5, year_id, day_id, station_id, channel):
        station_grp = self._get_station_group(h5, year_id, day_id, station_id)
        channel_grp = station_grp["waveform"][channel]

        segments = []

        for ds_key in sorted(channel_grp.keys(), key=lambda x: int(x)):
            ds = channel_grp[ds_key]

            segments.append(
                {
                    "data": ds[()],
                    "segment_index": int(get_attr(ds, "segment_index", ds_key)),
                    "starttime": parse_time(get_attr(ds, "starttime", "")),
                    "endtime": parse_time(get_attr(ds, "endtime", "")),
                    "sampling_rate": float(get_attr(ds, "sampling_rate", np.nan)),
                    "delta": float(get_attr(ds, "delta", np.nan)),
                    "npts": int(get_attr(ds, "npts", ds.shape[0])),
                    "network": get_attr(ds, "network", ""),
                    "station": get_attr(ds, "station", ""),
                    "location": normalize_location(
                        get_attr(ds, "location", self.default_location),
                        self.default_location,
                    ),
                    "channel": get_attr(ds, "channel", channel),
                    "mseed_source_file": get_attr(ds, "mseed_source_file", ""),
                    "dtype": get_attr(ds, "dtype", str(ds.dtype)),
                    "longitude": get_float_attr(ds, "longitude", np.nan),
                    "latitude": get_float_attr(ds, "latitude", np.nan),
                    "elevation": get_float_attr(ds, "elevation", np.nan),
                    "location_available": get_bool_attr(ds, "location_available", False),
                    "location_source": get_attr(ds, "location_source", ""),
                    "station_position_starttime": get_attr(ds, "station_position_starttime", ""),
                    "station_position_endtime": get_attr(ds, "station_position_endtime", ""),
                    "position_match_mode": get_attr(ds, "position_match_mode", ""),
                    "position_is_fallback": get_bool_attr(ds, "position_is_fallback", False),
                }
            )

        channel_info = self._read_channel_attrs(channel_grp)
        return segments, channel_info

    def __getitem__(self, idx):
        item = self.index[idx]
        h5_file = item["h5_file"]

        h5 = self._get_h5_handle(h5_file)
        should_close = not self.keep_h5_open

        try:
            year_id = item["year_id"]
            day_id = item["day_id"]
            station_id = item["station_id"]

            station_grp = self._get_station_group(h5, year_id, day_id, station_id)
            station_info = self._read_station_attrs(station_grp)

            if self.mode == "single":
                return self._getitem_single(h5, item, station_info)

            if self.mode == "three":
                return self._getitem_three(h5, item, station_info)

            if self.mode == "multi":
                return self._getitem_multi(h5, item, station_info)

            raise ValueError(f"Unsupported mode: {self.mode}")
        finally:
            if should_close:
                try:
                    h5.close()
                except Exception:
                    pass

    def _getitem_single(self, h5, item, station_info):
        year_id = item["year_id"]
        day_id = item["day_id"]
        station_id = item["station_id"]
        channel = item["channel"]

        segments, channel_info = self._read_channel_segments(
            h5, year_id, day_id, station_id, channel
        )

        waveform, starttime, endtime, original_sr = fill_segments_to_array(
            segments,
            fill_value=self.fill_value,
            dtype=self.dtype,
            use_overlap_mask=self.use_overlap_mask,
            max_duration_sec=self.max_duration_sec,
        )
        # Free raw HDF5 data arrays immediately after consolidation.
        # For day-long waveforms each segment["data"] can be tens of MB;
        # holding them until function return causes unbounded RSS growth.
        for _seg in segments:
            _seg.pop("data", None)

        if waveform is None:
            waveform = np.zeros(0, dtype=self.dtype)

        waveform, instrument_processing = self._apply_instrument_processing(
            waveform,
            segments=segments,
            channel=channel,
            starttime=starttime,
            endtime=endtime,
            sampling_rate=original_sr,
        )

        waveform, current_sr = resample_1d_array(
            waveform,
            original_sr=original_sr,
            target_sr=self.target_sampling_rate,
            dtype=self.dtype,
        )

        position_info = get_position_from_segments(segments)
        station_info = dict(station_info)
        station_info.update(position_info)

        return {
            "mode": "single",
            "h5_file": item["h5_file"],
            "year_id": year_id,
            "day_id": day_id,
            "station_id": station_id,
            "station_info": station_info,
            "channel_info": channel_info,
            "channel": channel,
            "channels": [channel],
            "instrument_processing": instrument_processing,
            "waveform": torch.from_numpy(waveform),
            "segments": (
                [
                    {k: v for k, v in seg.items() if k != "data"}
                    for seg in segments
                ]
                if self.include_segments_metadata else []
            ),
            "starttime": str(starttime) if starttime is not None else "",
            "endtime": str(endtime) if endtime is not None else "",
            "original_sampling_rate": original_sr,
            "sampling_rate": current_sr,
            "target_sampling_rate": self.target_sampling_rate,
            "resampled": (
                self.target_sampling_rate is not None
                and np.isfinite(original_sr)
                and abs(float(original_sr) - float(self.target_sampling_rate)) > 1e-6
            ),
            "npts_original_estimated": int(round((endtime - starttime) * original_sr)) + 1
            if starttime is not None and endtime is not None and np.isfinite(original_sr)
            else 0,
            "npts": waveform.shape[0],
        }

    def _select_three_channels(self, candidate_channels):
        candidate_channels = sorted(candidate_channels, key=component_rank)

        selected = {}

        for cha in candidate_channels:
            suf = channel_suffix(cha)

            if suf in ["E", "1"] and 0 not in selected:
                selected[0] = cha
            elif suf in ["N", "2"] and 1 not in selected:
                selected[1] = cha
            elif suf in ["Z", "3"] and 2 not in selected:
                selected[2] = cha

        is_z_only = False
        z_only_replicated = False

        if not has_three_components(candidate_channels):
            z_candidates = [
                ch for ch in candidate_channels
                if str(ch).upper() in self.allowed_z_only_channels
            ]

            if len(z_candidates) > 0:
                zch = z_candidates[0]
                selected = {2: zch}
                is_z_only = True

                if self.replicate_z_only:
                    selected = {0: zch, 1: zch, 2: zch}
                    z_only_replicated = True

        return selected, is_z_only, z_only_replicated

    def _getitem_three(self, h5, item, station_info):
        year_id = item["year_id"]
        day_id = item["day_id"]
        station_id = item["station_id"]
        channel_family = item["channel_family"]
        candidate_channels = item["channels"]

        selected, is_z_only, z_only_replicated = self._select_three_channels(
            candidate_channels
        )

        arrays = {}
        starts = []
        ends = []
        srs = []
        all_segments = []
        channel_infos = {}
        instrument_processing = {}

        unique_channels = sorted(set(selected.values()))
        channel_arrays = {}

        for cha in unique_channels:
            segments, channel_info = self._read_channel_segments(
                h5, year_id, day_id, station_id, cha
            )

            all_segments.extend(segments)
            channel_infos[cha] = channel_info

            arr, st, et, sr = fill_segments_to_array(
                segments,
                fill_value=self.fill_value,
                dtype=self.dtype,
                use_overlap_mask=self.use_overlap_mask,
                max_duration_sec=self.max_duration_sec,
            )
            # Free raw HDF5 data arrays immediately after consolidation.
            # all_segments holds the same dict objects, so popping here
            # also clears the data refs from all_segments — no deep copy needed.
            for _seg in segments:
                _seg.pop("data", None)

            if arr is None:
                continue

            arr, instrument_processing[cha] = self._apply_instrument_processing(
                arr,
                segments=segments,
                channel=cha,
                starttime=st,
                endtime=et,
                sampling_rate=sr,
            )
            channel_arrays[cha] = arr
            starts.append(st)
            ends.append(et)
            srs.append(sr)

        for comp_idx, cha in selected.items():
            if cha in channel_arrays:
                arrays[comp_idx] = channel_arrays[cha]

        _arrays_empty = len(arrays) == 0
        if _arrays_empty:
            waveform = np.zeros((0, 3), dtype=self.dtype)
            starttime = None
            endtime = None
            original_sr = np.nan
        else:
            original_sr = float(srs[0])
            starttime = min(starts)
            endtime = max(ends)

            # Align components by absolute start time.  The previous version
            # blindly wrote each channel from index 0, which silently misaligned
            # E/N/Z if their first segment start times differed.
            channel_starts = {}
            for cha, arr in channel_arrays.items():
                # Find the start time recorded for this channel by matching the
                # selected channel and using the minimum segment start.
                cha_starts = [
                    seg["starttime"] for seg in all_segments
                    if str(seg.get("channel", "")) == str(cha)
                ]
                if cha_starts:
                    channel_starts[cha] = min(cha_starts)
                else:
                    channel_starts[cha] = starttime

            if is_z_only and not self.replicate_z_only:
                comp_count = 1
                max_end_index = 0
                for comp_idx, arr in arrays.items():
                    cha = selected.get(comp_idx, "")
                    offset = int(round((channel_starts.get(cha, starttime) - starttime) * original_sr))
                    max_end_index = max(max_end_index, max(0, offset) + len(arr))
                waveform = np.full((max_end_index, comp_count), self.fill_value, dtype=self.dtype)

                if 2 in arrays:
                    cha = selected.get(2, "")
                    offset = int(round((channel_starts.get(cha, starttime) - starttime) * original_sr))
                    offset = max(0, offset)
                    waveform[offset: offset + len(arrays[2]), 0] = arrays[2]
            else:
                comp_count = 3
                max_end_index = 0
                for comp_idx, arr in arrays.items():
                    cha = selected.get(comp_idx, "")
                    offset = int(round((channel_starts.get(cha, starttime) - starttime) * original_sr))
                    max_end_index = max(max_end_index, max(0, offset) + len(arr))

                waveform = np.full((max_end_index, comp_count), self.fill_value, dtype=self.dtype)

                for comp_idx, arr in arrays.items():
                    cha = selected.get(comp_idx, "")
                    offset = int(round((channel_starts.get(cha, starttime) - starttime) * original_sr))
                    offset = max(0, offset)
                    waveform[offset: offset + len(arr), comp_idx] = arr

            waveform, current_sr = resample_2d_array(
                waveform,
                original_sr=original_sr,
                target_sr=self.target_sampling_rate,
                dtype=self.dtype,
            )
            # Release per-channel intermediate arrays now that waveform is built.
            del arrays, channel_arrays
            _arrays_empty = False

        if _arrays_empty:
            current_sr = np.nan

        position_info = get_position_from_segments(all_segments)
        station_info = dict(station_info)
        station_info.update(position_info)

        if is_z_only and not self.replicate_z_only:
            channels_out = [selected.get(2, "")]
            component_order = "Z only"
        else:
            channels_out = [
                selected.get(0, ""),
                selected.get(1, ""),
                selected.get(2, ""),
            ]
            component_order = "E/N/Z or 1/2/3"

        return {
            "mode": "three",
            "h5_file": item["h5_file"],
            "year_id": year_id,
            "day_id": day_id,
            "station_id": station_id,
            "station_info": station_info,
            "channel_family": channel_family,
            "channel_info": channel_infos,
            "channels": channels_out,
            "component_order": component_order,
            "is_z_only": is_z_only,
            "z_only_replicated": z_only_replicated,
            "instrument_processing": instrument_processing,
            "waveform": torch.from_numpy(waveform),
            "segments": (
                [
                    {k: v for k, v in seg.items() if k != "data"}
                    for seg in all_segments
                ]
                if self.include_segments_metadata else []
            ),
            "starttime": str(starttime) if starttime is not None else "",
            "endtime": str(endtime) if endtime is not None else "",
            "original_sampling_rate": original_sr,
            "sampling_rate": current_sr,
            "target_sampling_rate": self.target_sampling_rate,
            "resampled": (
                self.target_sampling_rate is not None
                and np.isfinite(original_sr)
                and abs(float(original_sr) - float(self.target_sampling_rate)) > 1e-6
            ),
            "npts": waveform.shape[0],
        }

    def _getitem_multi(self, h5, item, station_info):
        year_id = item["year_id"]
        day_id = item["day_id"]
        station_id = item["station_id"]
        channel_family = item["channel_family"]
        channels = item["channels"]

        arrays = []
        used_channels = []
        starts = []
        ends = []
        srs = []
        all_segments = []
        channel_infos = {}
        instrument_processing = {}

        for cha in channels:
            segments, channel_info = self._read_channel_segments(
                h5, year_id, day_id, station_id, cha
            )

            all_segments.extend(segments)
            channel_infos[cha] = channel_info

            arr, st, et, sr = fill_segments_to_array(
                segments,
                fill_value=self.fill_value,
                dtype=self.dtype,
                use_overlap_mask=self.use_overlap_mask,
                max_duration_sec=self.max_duration_sec,
            )
            # Free raw HDF5 data arrays immediately after consolidation.
            for _seg in segments:
                _seg.pop("data", None)

            if arr is None:
                continue

            arr, instrument_processing[cha] = self._apply_instrument_processing(
                arr,
                segments=segments,
                channel=cha,
                starttime=st,
                endtime=et,
                sampling_rate=sr,
            )
            arrays.append(arr)
            used_channels.append(cha)
            starts.append(st)
            ends.append(et)
            srs.append(sr)

        if len(arrays) == 0:
            waveform = np.zeros((0, 0), dtype=self.dtype)
            starttime = None
            endtime = None
            original_sr = np.nan
            current_sr = np.nan
        else:
            max_len = max(len(a) for a in arrays)
            waveform = np.full(
                (max_len, len(arrays)),
                self.fill_value,
                dtype=self.dtype,
            )

            for i, arr in enumerate(arrays):
                waveform[: len(arr), i] = arr

            starttime = min(starts)
            endtime = max(ends)
            original_sr = float(srs[0])

            waveform, current_sr = resample_2d_array(
                waveform,
                original_sr=original_sr,
                target_sr=self.target_sampling_rate,
                dtype=self.dtype,
            )
            # Release per-channel arrays now that waveform is built.
            del arrays

        position_info = get_position_from_segments(all_segments)
        station_info = dict(station_info)
        station_info.update(position_info)

        return {
            "mode": "multi",
            "h5_file": item["h5_file"],
            "year_id": year_id,
            "day_id": day_id,
            "station_id": station_id,
            "station_info": station_info,
            "channel_family": channel_family,
            "channel_info": channel_infos,
            "channels": used_channels,
            "instrument_processing": instrument_processing,
            "waveform": torch.from_numpy(waveform),
            "segments": (
                [
                    {k: v for k, v in seg.items() if k != "data"}
                    for seg in all_segments
                ]
                if self.include_segments_metadata else []
            ),
            "starttime": str(starttime) if starttime is not None else "",
            "endtime": str(endtime) if endtime is not None else "",
            "original_sampling_rate": original_sr,
            "sampling_rate": current_sr,
            "target_sampling_rate": self.target_sampling_rate,
            "resampled": (
                self.target_sampling_rate is not None
                and np.isfinite(original_sr)
                and abs(float(original_sr) - float(self.target_sampling_rate)) > 1e-6
            ),
            "npts": waveform.shape[0],
        }


def waveform_collate_fn(batch):
    return batch


def hdf5_worker_init_fn(worker_id):
    """Worker initializer for DataLoader when using fork-based multiprocessing.

    With 'fork', child processes inherit the parent's open h5py file handles.
    Accessing inherited handles from multiple processes simultaneously causes
    HDF5 library errors or silent data corruption.  This function walks all live
    objects and resets the handle cache of every HDF5WaveformDataset instance it
    finds, forcing each worker to open fresh, independent handles on its first
    __getitem__ call.

    Usage::

        from torch.utils.data import DataLoader
        loader = DataLoader(
            dataset,
            num_workers=4,
            multiprocessing_context='fork',   # only if you must use fork
            worker_init_fn=hdf5_worker_init_fn,
        )

    Note: With 'spawn' or 'forkserver' (the recommended and default choice on
    Linux when num_workers > 0), this function is not needed because the worker
    process starts fresh and HDF5WaveformDataset.__setstate__ already ensures an
    empty _h5_cache.  It is safe to pass it regardless.
    """
    import gc as _gc

    for obj in _gc.get_objects():
        if isinstance(obj, HDF5WaveformDataset):
            try:
                obj.close()
            except Exception:
                pass
            try:
                obj._h5_cache = {}
            except Exception:
                pass


def padded_collate_fn(batch, fill_value=0.0):
    lengths = []
    arrays = []

    max_t = 0
    max_c = 1

    for item in batch:
        x = item["waveform"]

        if x.ndim == 1:
            x = x[:, None]

        t, c = x.shape
        max_t = max(max_t, t)
        max_c = max(max_c, c)

        lengths.append(t)
        arrays.append(x)

    out = torch.full(
        (len(batch), max_t, max_c),
        fill_value=float(fill_value),
        dtype=arrays[0].dtype,
    )

    for i, x in enumerate(arrays):
        t, c = x.shape
        out[i, :t, :c] = x

    meta = []

    for item in batch:
        d = dict(item)
        d.pop("waveform")
        meta.append(d)

    return {
        "waveform": out,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "meta": meta,
    }


if __name__ == "__main__":
    h5_input = "data/continuous_waveform_usa_20190701.h5"
    """
    # 1. single file
    h5_input = "data/continuous_waveform_usa.h5"

    # 2. glob multiple files
    h5_input = "data/continuous_waveform_usa_*.h5"

    # 3. data directory
    h5_input = "data/"

    # 4. explicit file list
    h5_input = [
        "data/continuous_waveform_usa_20190701.h5",
        "data/continuous_waveform_usa_20211108.h5",
    ]
    """
    dataset = HDF5WaveformDataset(
        h5_file=h5_input,
        mode="three",

        # Default: keep commonly used seismic channel families
        allowed_families=("HH", "BH", "EH", "HN"),

        # Additionally allow single-component (Z only) samples such as EHZ
        allowed_z_only_channels=("EHZ",),
        allow_z_only=True,

        # Whether to replicate single Z component to three channels [Z, Z, Z]
        replicate_z_only=True,

        # Target sampling rate (Hz); None means no resampling
        # e.g., 100.0 → resample all waveforms to 100 Hz
        target_sampling_rate=100.0,

        fill_value=0.0,
        dtype=np.float32,
        default_location="--",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=waveform_collate_fn,
    )

    print("HDF5 files:", len(dataset.h5_files))
    for f in dataset.h5_files:
        print("  ", f)

    print("Number of samples:", len(dataset))

    for batch in loader:
        for item in batch:
            print("=" * 80)
            print("h5_file:", item["h5_file"])
            print("station_id:", item["station_id"])
            print("mode:", item["mode"])
            print("channel_family:", item.get("channel_family", ""))
            print("channels:", item["channels"])
            print("is_z_only:", item.get("is_z_only", False))
            print("z_only_replicated:", item.get("z_only_replicated", False))
            print("starttime:", item["starttime"])
            print("endtime:", item["endtime"])
            print("original_sampling_rate:", item["original_sampling_rate"])
            print("sampling_rate:", item["sampling_rate"])
            print("target_sampling_rate:", item["target_sampling_rate"])
            print("resampled:", item["resampled"])
            print("waveform shape:", tuple(item["waveform"].shape))
        break
