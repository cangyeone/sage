#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory-safe phase picking script for large HDF5 continuous waveform datasets.
Supports Mac/MPS, CUDA (single or multi-GPU), and CPU, with optional multi-process
DataLoader for parallel waveform prefetching.

Key design points:
  1. Resume skipping is pushed into HDF5WaveformDataset(skip_jsonl=...) so already
     processed samples are filtered from the metadata index before __getitem__ reads
     waveform arrays from HDF5.
  2. Segment raw data arrays are freed immediately after fill_segments_to_array to
     avoid unbounded RSS growth on day-long waveforms (~100 MB/sample).
  3. Uses torch.inference_mode() and explicit deletion of large temporary tensors.
  4. MPS: torch.mps.empty_cache() called every sample; model reloaded periodically
     (--reload_model_interval) to reset TorchScript allocator state.
  5. CUDA: non_blocking transfers, cuda.empty_cache every N samples; model reload
     disabled by default (not needed for CUDA allocator).
  6. Multi-process DataLoader: uses 'spawn' context on Linux to avoid h5py + fork
     deadlocks; each worker opens its own HDF5 handles lazily.

Recommended — Mac/MPS (restart loop to avoid MPS allocator accumulation):
  # Each run processes --max_samples new samples then exits with code 75.
  # The OS reclaims Metal + HDF5 allocator state on exit. --resume picks up
  # where the previous run left off. When all samples are done the script
  # exits with code 0 and the loop terminates naturally.
  while true; do
    python run_picker_to_jsonl.py \
      --h5_input 'data/hdf5/continuous_waveform_usa_*.h5' \
      --output_jsonl data/picks/output.jsonl \
      --picker_model pickers/pnsn.v1.jit \
      --polar_model pickers/polar.onnx \
      --device mps --max_samples 200
    [ $? -ne 75 ] && break
  done

Recommended — CUDA with multi-process prefetch:
  python run_picker_to_jsonl_mps_safe.py \
    --h5_input 'data/hdf5/continuous_waveform_usa_*.h5' \
    --output_jsonl data/picks/output.jsonl \
    --picker_model pickers/pnsn.v1.jit \
    --polar_model pickers/polar.onnx \
    --device cuda \
    --batch_size 4 \
    --num_workers 4 \
    --prefetch_factor 2 \
    --multiprocessing_context spawn
"""

import argparse
import datetime
import gc
import json
import platform
import signal
import sys
import time
import os
import heapq
from bisect import bisect_left
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Disable HDF5 POSIX file locking before h5py is imported.
# h5py calls H5open() (which reads HDF5_USE_FILE_LOCKING) the moment the
# module is imported.  Without this, h5py.File() can block indefinitely
# waiting to acquire a POSIX fcntl lock held by Spotlight, Time Machine,
# a crashed previous run, or an NFS server — with no timeout.
# Setting it here (read-only access) is safe; allow user override via env.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import onnxruntime as ort
except Exception:
    ort = None

# Prefer the revised resume-aware loader. Keep a fallback name for compatibility.

from utils.hdf5_waveform_dataset import (
    HDF5WaveformDataset,
    waveform_collate_fn,
    hdf5_worker_init_fn,
)

# orjson is 3-10× faster than stdlib json for serialising Python dicts.
# Fall back gracefully if it is not installed.
try:
    import orjson as _orjson

    def _json_dumps(obj):
        return _orjson.dumps(obj).decode("utf-8")

except ImportError:
    _orjson = None

    def _json_dumps(obj):
        return json.dumps(obj, ensure_ascii=False)


PHASE_ID_TO_NAME = {
    0: "Pg",
    1: "Sg",
    2: "Pn",
    3: "Sn",
    4: "P",
    5: "S",
}


def format_seconds(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def format_rate(num, seconds, suffix="/s"):
    seconds = float(seconds)
    if seconds <= 0:
        return "inf" + suffix
    return f"{float(num) / seconds:.2f}{suffix}"


def safe_shape_text(x):
    try:
        if torch.is_tensor(x):
            return "x".join(str(v) for v in tuple(x.shape))
        if hasattr(x, "shape"):
            return "x".join(str(v) for v in tuple(x.shape))
    except Exception:
        pass
    return "?"


def select_torch_device(device_name="auto"):
    device_name = str(device_name).lower()

    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        print("[WARN] CUDA requested but not available. Fallback to CPU.")
        return torch.device("cpu"), "cpu"

    if device_name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        print("[WARN] MPS requested but not available. Fallback to CPU.")
        return torch.device("cpu"), "cpu"

    return torch.device("cpu"), "cpu"


def sync_device(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def empty_device_cache(device_type):
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def force_device_cleanup(device_type, do_gc=True):
    """Conservative cleanup for CUDA/MPS/CPU."""
    sync_device(device_type)
    if do_gc:
        gc.collect()
    empty_device_cache(device_type)


# ── Per-sample timeout (SIGALRM, Unix/macOS only) ───────────────────────────
# On MPS, torch.mps.synchronize() and torch.mps.empty_cache() can hang
# indefinitely if the Metal command queue enters a bad state.  A SIGALRM
# watchdog lets us escape the hang, write an error record, and continue.
# SIGALRM is only available on the main thread on Unix; on Windows it is
# silently disabled.

_SIGALRM_AVAILABLE = hasattr(signal, "SIGALRM")


class _SampleTimeout(Exception):
    """Raised by SIGALRM when a single sample exceeds sample_timeout_sec."""


def _sample_timeout_handler(signum, frame):
    raise _SampleTimeout("sample timed out (Metal / ONNX hang?)")


class _SampleTimer:
    """Context manager that arms/disarms SIGALRM around one sample.

    Usage::

        with _SampleTimer(timeout_sec):
            ... process one sample ...

    On timeout, _SampleTimeout is raised in the SIGALRM handler, which
    propagates through the with-block.  The caller should catch it, write an
    error record, and continue to the next sample.

    If SIGALRM is not available (Windows) or timeout_sec <= 0, this is a no-op.
    """

    __slots__ = ("_timeout",)

    def __init__(self, timeout_sec: int):
        self._timeout = timeout_sec if _SIGALRM_AVAILABLE and timeout_sec > 0 else 0

    def __enter__(self):
        if self._timeout > 0:
            signal.signal(signal.SIGALRM, _sample_timeout_handler)
            signal.alarm(self._timeout)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timeout > 0:
            signal.alarm(0)          # cancel any pending alarm
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
        return False                 # do not suppress exceptions


def get_process_rss_mb():
    """Return current process RSS in MB if psutil is available."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def get_device_memory_text(device_type):
    """Return a short memory stats string for progress logging.

    For CUDA: shows allocated / reserved MB from torch.cuda.
    For MPS: falls back to process RSS (MPS uses unified memory, no per-device API).
    For CPU: shows process RSS only.
    """
    parts = []

    if device_type == "cuda":
        try:
            alloc = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            parts.append(f"cuda_alloc={alloc:.0f}MB reserved={reserved:.0f}MB")
        except Exception:
            pass

    rss = get_process_rss_mb()
    if rss is not None:
        parts.append(f"rss={rss:.0f}MB")

    return " | " + " | ".join(parts) if parts else ""


def get_dataloader_multiprocessing_context(num_workers, device_type, requested="auto"):
    """Choose a safe multiprocessing start method for the DataLoader.

    h5py is not fork-safe: accessing inherited file handles from multiple child
    processes causes HDF5 library errors or silent corruption.  On Linux the
    default start method is 'fork', so we override it to 'spawn' automatically.
    macOS defaults to 'spawn' (Python ≥ 3.8), so no override is needed there.

    Args:
        num_workers: DataLoader num_workers value.
        device_type: 'cuda', 'mps', or 'cpu'.
        requested: 'auto' lets this function decide; any other string is used as-is.

    Returns:
        A multiprocessing context string, or None (use PyTorch default).
    """
    if num_workers == 0:
        return None  # single-process mode; no context needed

    if requested != "auto":
        return requested  # user override

    system = platform.system()
    if system == "Linux":
        # Linux default is 'fork' which is not safe with h5py.
        return "spawn"

    if system == "Darwin":
        # macOS Python ≥ 3.8 defaults to 'spawn', but be explicit so the
        # log output and MPS+num_workers guidance are unambiguous.
        return "spawn"

    # Windows: platform default ('spawn') is already safe.
    return None


def load_picker_model(picker_model, device, device_type):
    picker = torch.jit.load(str(picker_model), map_location=device)
    picker.eval()
    picker.to(device)
    sync_device(device_type)
    return picker



def get_onnx_providers(device_type="cpu", requested="auto"):
    """Select ONNX Runtime providers for picker / polarity inference.

    device_type is the normalized runtime name from select_torch_device():
    "cuda", "mps", or "cpu".  ONNX Runtime does not have an MPS provider;
    on Apple Silicon / macOS acceleration is exposed through CoreMLExecutionProvider.
    """
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX picker inference.")

    available = list(ort.get_available_providers())

    if requested and requested != "auto":
        providers = [x.strip() for x in str(requested).split(",") if x.strip()]
        providers = [p for p in providers if p in available]
        if "CPUExecutionProvider" not in providers and "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = ["CPUExecutionProvider"]
        return providers

    providers = []
    if device_type == "cuda" and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    elif device_type == "mps":
        # ONNX Runtime acceleration on macOS is CoreML, not a literal MPS EP.
        if "CoreMLExecutionProvider" in available:
            providers.append("CoreMLExecutionProvider")

    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    return providers or available or ["CPUExecutionProvider"]


def load_onnx_picker_model(picker_model, device_type="cpu", providers="auto"):
    """Load ONNX picker model.

    Expected model interface:
      input:  "wave" with shape [T, 3], float32
      output: "prob" with shape [N, C], "time" with shape [N]
    """
    if ort is None:
        raise ImportError("onnxruntime is required for ONNX picker inference.")

    sess_options = ort.SessionOptions()
    try:
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except Exception:
        pass

    selected = get_onnx_providers(device_type=device_type, requested=providers)
    print(f"[INFO] ONNX Runtime providers for picker model: {selected}")
    return ort.InferenceSession(str(picker_model), sess_options=sess_options, providers=selected)


def infer_picker_backend_from_suffix(picker_model):
    """Infer picker backend from model filename suffix.

    .onnx -> ONNX Runtime + external heap-NMS
    .jit/.torchscript/.pt/.pth -> TorchScript by default
    """
    suffix = Path(str(picker_model)).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in (".jit", ".torchscript", ".pt", ".pth"):
        return "torchscript"
    # Keep historical behavior for unknown suffixes.
    return "torchscript"


def postprocess_picker_heap_nms(prob, time_values, prob_thresh=0.1, nms_win=200):
    """Heap-based NMS for ONNX picker outputs.

    prob: [N, C], where channel 0 is usually background and channels 1..C-1 are phases.
    time_values: [N], sample indices corresponding to prob rows.

    Returns ndarray [K, 3]: phase_id, sample_index, confidence.
    phase_id follows the TorchScript wrapper convention: channel 1 -> 0, channel 2 -> 1, etc.
    """
    prob = np.asarray(prob, dtype=np.float32)
    time_values = np.asarray(time_values, dtype=np.float32).reshape(-1)

    if prob.ndim != 2:
        raise ValueError(f"Expected ONNX prob with shape [N, C], got {prob.shape}")
    if time_values.ndim != 1:
        raise ValueError(f"Expected ONNX time with shape [N], got {time_values.shape}")
    if prob.shape[0] != time_values.shape[0]:
        raise ValueError(f"ONNX prob/time length mismatch: prob={prob.shape}, time={time_values.shape}")

    output = []
    n, c = prob.shape
    nms_win = float(nms_win)

    for itr in range(c - 1):
        pc = prob[:, itr + 1]
        mask = pc > float(prob_thresh)
        if not np.any(mask):
            continue

        time_sel = time_values[mask]
        score_sel = pc[mask]

        # Heap sorted by descending score using negative score.
        heap = [(-float(s), float(ts), i) for i, (s, ts) in enumerate(zip(score_sel, time_sel))]
        heapq.heapify(heap)

        accepted_times = []  # sorted by time
        accepted_idx = []

        while heap:
            neg_s, ts, i = heapq.heappop(heap)
            pos = bisect_left(accepted_times, ts)

            conflict = False
            if pos > 0 and abs(ts - accepted_times[pos - 1]) <= nms_win:
                conflict = True
            if pos < len(accepted_times) and abs(accepted_times[pos] - ts) <= nms_win:
                conflict = True

            if conflict:
                continue

            accepted_times.insert(pos, ts)
            accepted_idx.append(i)

        if not accepted_idx:
            continue

        p_time = time_sel[accepted_idx].astype(np.float32, copy=False)
        p_prob = score_sel[accepted_idx].astype(np.float32, copy=False)
        p_type = np.full(p_time.shape, itr, dtype=np.float32)
        output.append(np.stack([p_type, p_time, p_prob], axis=1))

    if not output:
        return np.zeros((0, 3), dtype=np.float32)

    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def run_onnx_picker_from_tensor(sess, x_cpu, prob_thresh=0.1, nms_win=200):
    """Run ONNX picker and external heap-NMS postprocessing.

    This replaces the slow TorchScript-internal NMS. The ONNX model should output
    dense probability/time arrays, and this function returns the final [K, 3]
    picks compatible with the rest of the JSONL writer.
    """
    if torch.is_tensor(x_cpu):
        x_np = x_cpu.detach().cpu().numpy()
    else:
        x_np = np.asarray(x_cpu)

    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if x_np.shape[1] == 1:
        x_np = np.repeat(x_np, 3, axis=1)
    elif x_np.shape[1] > 3:
        x_np = x_np[:, :3]
    elif x_np.shape[1] < 3:
        pad = np.zeros((x_np.shape[0], 3 - x_np.shape[1]), dtype=x_np.dtype)
        x_np = np.concatenate([x_np, pad], axis=1)

    x_np = np.ascontiguousarray(x_np.astype(np.float32, copy=False))

    # Prefer named outputs, matching your ONNX example.  If names differ, fall back to positional outputs.
    try:
        prob, time_values = sess.run(["prob", "time"], {"wave": x_np})
    except Exception:
        outputs = sess.run(None, {"wave": x_np})
        if len(outputs) < 2:
            raise ValueError("ONNX picker must return at least two outputs: prob and time")
        prob, time_values = outputs[0], outputs[1]

    picks = postprocess_picker_heap_nms(
        prob,
        time_values,
        prob_thresh=prob_thresh,
        nms_win=nms_win,
    )
    return picks, np.asarray(prob), np.asarray(time_values)


def to_jsonable(obj):
    if torch.is_tensor(obj):
        return {
            "__tensor__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def parse_starttime_to_datetime(starttime):
    if starttime is None or str(starttime).strip() == "":
        return None

    s = str(starttime).strip().replace("Z", "")

    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        pass

    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.datetime.strptime(s, fmt)
        except Exception:
            continue

    return None


def isoformat_z(dt):
    if dt is None:
        return None
    return dt.isoformat(timespec="microseconds") + "Z"


def ensure_waveform_tensor_for_picker(waveform):
    """Return CPU float32 tensor with shape [T, 3] without unnecessary NumPy copies."""
    if torch.is_tensor(waveform):
        x = waveform.detach()
        if x.device.type != "cpu":
            x = x.cpu()
        if x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)
    else:
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32))

    if x.ndim == 1:
        x = x[:, None]

    if x.shape[1] == 1:
        x = x.repeat(1, 3)
    elif x.shape[1] > 3:
        x = x[:, :3]
    elif x.shape[1] < 3:
        pad = torch.zeros((x.shape[0], 3 - x.shape[1]), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)

    if not x.is_contiguous():
        x = x.contiguous()

    return x


def get_z_component_numpy(waveform):
    """Return Z component as a NumPy view/copy with minimal conversion."""
    x = ensure_waveform_tensor_for_picker(waveform)
    z = x[:, 2]
    return z.numpy()


def run_torchscript_picker_from_tensor(sess, x_cpu, device):
    """Run picker using an already prepared CPU tensor [T, 3].

    This avoids constructing the full waveform tensor twice in one sample. The
    output is an ndarray with columns: phase_id, sample_index, confidence.
    """
    xt = None
    y = None
    out_cpu = None
    try:
        with torch.inference_mode():
            # non_blocking=True overlaps CPU→GPU PCIe transfer with other CUDA work.
            # For MPS (unified memory) and CPU it is a no-op but harmless.
            xt = x_cpu.to(device=device, dtype=torch.float32,
                          non_blocking=(device.type == "cuda"))
            y = sess(xt)

        # Move output back to CPU *before* deleting the MPS tensors so the
        # MPS allocator can reclaim the memory on the next empty_cache call.
        if torch.is_tensor(y):
            out_cpu = y.detach().cpu()
            del y
            y = None
            out = out_cpu.numpy().copy()  # copy so the tensor can be freed
            del out_cpu
            out_cpu = None
        else:
            out = np.asarray(y)

        # Release the device-side input tensor now that inference is done.
        del xt
        xt = None

        if out.ndim == 1:
            out = out[None, :]

        if out.shape[1] == 2:
            conf = np.ones((out.shape[0], 1), dtype=np.float32)
            out = np.concatenate([out, conf], axis=1)

        if out.shape[1] < 3:
            raise ValueError(f"Unexpected picker output shape: {out.shape}")

        return out[:, :3].astype(np.float32, copy=False)

    finally:
        del xt
        del y
        del out_cpu


def run_torchscript_picker(sess, waveform, device):
    """Backward-compatible wrapper."""
    x_cpu = ensure_waveform_tensor_for_picker(waveform)
    try:
        return run_torchscript_picker_from_tensor(sess, x_cpu, device)
    finally:
        del x_cpu

def compute_pick_quality(z, sample_index, sr, snr_window_sec=2.0):
    z = np.asarray(z, dtype=np.float32)
    pidx = int(round(sample_index))

    if len(z) == 0 or not np.isfinite(sr) or float(sr) <= 0:
        return {
            "snr": None,
            "amplitude": None,
            "pre_std": None,
            "post_std": None,
            "pre_abs_p95": None,
            "post_abs_p95": None,
        }

    win = max(1, int(round(float(snr_window_sec) * float(sr))))

    b0 = max(0, pidx - win)
    b1 = max(0, pidx)
    a0 = min(len(z), pidx)
    a1 = min(len(z), pidx + win)

    pre = z[b0:b1]
    post = z[a0:a1]

    if len(pre) == 0:
        pre = np.ones(win, dtype=np.float32)
    if len(post) == 0:
        post = np.ones(win, dtype=np.float32)

    pre_centered = pre - np.mean(pre)
    post_centered = post - np.mean(post)

    pre_std = float(np.std(pre_centered))
    post_std = float(np.std(post_centered))
    snr = post_std / (pre_std + 1e-6)

    amp_end = min(len(z), pidx + int(round(1.0 * float(sr))))
    amp_start = max(0, pidx - int(round(0.2 * float(sr))))
    amp_win = z[amp_start:amp_end]

    amplitude = float(np.max(np.abs(amp_win))) if len(amp_win) > 0 else None

    return {
        "snr": float(snr),
        "amplitude": amplitude,
        "pre_std": pre_std,
        "post_std": post_std,
        "pre_abs_p95": float(np.percentile(np.abs(pre_centered), 95)),
        "post_abs_p95": float(np.percentile(np.abs(post_centered), 95)),
    }


def load_polar_model(polar_model, device_name="cpu", providers="auto"):
    if not polar_model:
        return None

    if ort is None:
        raise ImportError("onnxruntime is required for polar model inference.")

    selected = get_onnx_providers(device_type=device_name, requested=providers)
    print(f"[INFO] ONNX Runtime providers for polarity model: {selected}")
    return ort.InferenceSession(str(polar_model), providers=selected)


def run_polar_picker(polar_sess, z, sample_index):
    """
    Input: Z component, 1024 samples around Pg.
    Output:
        polarity: U/D/N
        polarity probability
    """
    if polar_sess is None:
        return "N", 0.0

    z = np.asarray(z, dtype=np.float32)

    if len(z) == 0:
        return "N", 0.0

    pidx = int(round(sample_index))

    if pidx <= 512:
        pidx = 512
    if pidx >= len(z) - 512:
        pidx = len(z) - 512

    if pidx < 0:
        return "N", 0.0

    pdata = z[pidx - 512:pidx + 512]

    if len(pdata) > 1024:
        pdata = pdata[:1024]
    if len(pdata) < 1024:
        pdata = np.pad(pdata, (0, 1024 - len(pdata)))

    pdata = np.ascontiguousarray(pdata.astype(np.float32, copy=False))
    prob, = polar_sess.run(["prob"], {"wave": pdata})
    prob = np.asarray(prob).reshape(-1)

    polar_id = int(np.argmax(prob))
    polar_prob = float(np.max(prob))

    if polar_id == 0:
        return "U", polar_prob
    if polar_id == 1:
        return "D", polar_prob

    return "N", polar_prob


def get_station_info_compact(item):
    info = item.get("station_info", {})

    return {
        "station_id": item.get("station_id", ""),
        "network": info.get("network", ""),
        "station": info.get("station", ""),
        "location": info.get("location", ""),
        "longitude": info.get("longitude", None),
        "latitude": info.get("latitude", None),
        "elevation": info.get("elevation", None),
        "location_available": bool(info.get("location_available", False)),
        "position_in_time_range": (
            str(info.get("position_match_mode", ""))
            == "strict_time_matched_network_station_only"
        ),
        "position_is_fallback": bool(info.get("position_is_fallback", False)),
    }


def make_sample_key_from_item(item):
    """Build the same key that make_sample_key_from_index_item would produce.

    Must stay in sync with make_sample_key_from_index_item in
    utils/hdf5_waveform_dataset.py so that no_pick / error sentinel records
    have keys that match the dataset's resume filter.

    Z-only replicated stations: __getitem__ stores ["EHZ","EHZ","EHZ"] in
    item["channels"] but the index was built from the raw ["EHZ"] list.
    Deduplicate here so the key matches.
    """
    channels = item.get("channels") or []
    # Deduplicate for Z-only replicated stations so key matches the index.
    if item.get("z_only_replicated", False):
        seen: set = set()
        channels = [ch for ch in channels if not (ch in seen or seen.add(ch))]
    channels = ",".join(str(x) for x in channels)

    return "|".join([
        str(item.get("h5_file", "")),
        str(item.get("year_id", "")),
        str(item.get("day_id", "")),
        str(item.get("station_id", "")),
        str(item.get("channel_family", item.get("channel", ""))),
        channels,
    ])


def build_pick_record(
    item,
    z,
    phase_id,
    sample_index,
    confidence,
    polar_sess=None,
    snr_window_sec=2.0,
):
    sr = float(item.get("sampling_rate", np.nan))
    phase_id = int(phase_id)
    sample_index = float(sample_index)
    confidence = float(confidence)

    phase_name = PHASE_ID_TO_NAME.get(phase_id, f"phase_{phase_id}")

    if np.isfinite(sr) and sr > 0:
        phase_relative_time = sample_index / sr
    else:
        phase_relative_time = None

    start_dt = parse_starttime_to_datetime(item.get("starttime", ""))

    if start_dt is not None and phase_relative_time is not None:
        phase_dt = start_dt + datetime.timedelta(seconds=phase_relative_time)
    else:
        phase_dt = None

    quality = compute_pick_quality(
        z,
        sample_index=sample_index,
        sr=sr,
        snr_window_sec=snr_window_sec,
    )

    polarity = "N"
    polarity_prob = 0.0

    if phase_name == "Pg":
        polarity, polarity_prob = run_polar_picker(
            polar_sess,
            z,
            sample_index=sample_index,
        )

    record = {
        "record_type": "phase_pick",
        "phase_id": phase_id,
        "phase_name": phase_name,
        "phase_prob": confidence,
        "phase_sample": sample_index,
        "phase_relative_time": phase_relative_time,
        "phase_time": isoformat_z(phase_dt),
        "polarity": polarity,
        "polarity_prob": polarity_prob,
        "polarity_available": bool(polar_sess is not None and phase_name == "Pg"),
        "snr": quality.get("snr", None),
        "amplitude": quality.get("amplitude", None),
        "pre_std": quality.get("pre_std", None),
        "post_std": quality.get("post_std", None),
        "pre_abs_p95": quality.get("pre_abs_p95", None),
        "post_abs_p95": quality.get("post_abs_p95", None),
        "channels": item.get("channels", []),
        "channel_family": item.get("channel_family", ""),
        "component_order": item.get("component_order", ""),
        "is_z_only": bool(item.get("is_z_only", False)),
        "z_only_replicated": bool(item.get("z_only_replicated", False)),
        "waveform_shape": list(item["waveform"].shape),
        "sampling_rate": item.get("sampling_rate", None),
        "original_sampling_rate": item.get("original_sampling_rate", None),
        "target_sampling_rate": item.get("target_sampling_rate", None),
        "resampled": bool(item.get("resampled", False)),
        "h5_file": item.get("h5_file", ""),
        "year_id": item.get("year_id", ""),
        "day_id": item.get("day_id", ""),
        # station_id MUST be a top-level field: load_finished_sample_keys rebuilds
        # the resume key from h5_file|year_id|day_id|station_id|channel_family|channels
        # using item.get("station_id", "").  If it is only inside station_info the
        # scanner always sees "" and no record is ever marked as finished.
        "station_id": item.get("station_id", ""),
        "station_info": get_station_info_compact(item),
    }

    return to_jsonable(record)


def create_dataset_with_resume_filter(
    h5_input,
    output_jsonl,
    resume,
    allowed_families,
    allowed_z_only_channels,
    allow_z_only,
    replicate_z_only,
    target_sampling_rate,
    include_segments_metadata,
    keep_h5_open,
    use_overlap_mask,
    h5_rdcc_nbytes=8 * 1024 * 1024,
    max_duration_sec=90000.0,
):
    kwargs = dict(
        h5_file=h5_input,
        mode="three",
        allowed_families=allowed_families,
        allowed_z_only_channels=allowed_z_only_channels,
        allow_z_only=allow_z_only,
        replicate_z_only=replicate_z_only,
        target_sampling_rate=target_sampling_rate,
        fill_value=0.0,
        dtype=np.float32,
        default_location="--",
    )

    # These options exist in the revised loader. If the old loader is imported by
    # accident, the TypeError fallback keeps the script usable, but resume filtering
    # will not be memory-safe until utils/hdf5_waveform_dataset_resume.py is used.
    if resume:
        kwargs["skip_jsonl"] = output_jsonl
        kwargs["skip_record_type"] = "phase_pick"
    kwargs["keep_h5_open"] = keep_h5_open
    kwargs["include_segments_metadata"] = include_segments_metadata
    kwargs["use_overlap_mask"] = use_overlap_mask
    kwargs["h5_rdcc_nbytes"] = h5_rdcc_nbytes
    kwargs["max_duration_sec"] = max_duration_sec

    try:
        return HDF5WaveformDataset(**kwargs), True
    except TypeError as e:
        print("[WARN] Resume-aware loader options were rejected by HDF5WaveformDataset.")
        print(f"[WARN] Details: {e}")
        print("[WARN] Falling back to the old loader API; skip will not happen before waveform loading.")
        for key in [
            "skip_jsonl",
            "skip_record_type",
            "keep_h5_open",
            "include_segments_metadata",
            "use_overlap_mask",
            "h5_rdcc_nbytes",
            "max_duration_sec",
        ]:
            kwargs.pop(key, None)
        return HDF5WaveformDataset(**kwargs), False


def run_picker_to_jsonl(
    h5_input,
    output_jsonl,
    picker_model,
    polar_model=None,
    picker_backend="auto",
    onnx_providers="auto",
    onnx_prob_thresh=0.1,
    onnx_nms_win=200,
    device_name="auto",
    batch_size=1,
    num_workers=0,
    allowed_families=("HH", "BH", "EH", "HN"),
    allowed_z_only_channels=("EHZ",),
    allow_z_only=True,
    replicate_z_only=True,
    target_sampling_rate=100.0,
    min_confidence=0.0,
    snr_window_sec=2.0,
    progress_interval=100,
    resume=True,
    flush_interval=1000,
    gc_interval=2000,
    include_segments_metadata=False,
    keep_h5_open=True,
    use_overlap_mask=True,
    mps_empty_cache_interval=1,
    cuda_empty_cache_interval=200,
    reload_model_interval=-1,
    multiprocessing_context="auto",
    prefetch_factor=2,
    h5_rdcc_nbytes=8 * 1024 * 1024,
    max_samples=0,
    canonical_input_length=0,
    auto_restart=True,
    sample_timeout=120,
    max_picks_per_sample=0,
    max_duration_sec=90000.0,
    slow_load_threshold=10.0,
    slow_infer_threshold=10.0,
    slow_post_threshold=10.0,
    slow_total_threshold=30.0,
    flush_no_pick=False,
):
    t_all0 = time.perf_counter()

    device, device_type = select_torch_device(device_name)

    # Resolve reload_model_interval: -1 means auto-select per device.
    # MPS: reloading the model resets the TorchScript allocator state and
    # avoids unbounded RSS growth — but also triggers Metal kernel
    # recompilation on the very first inference after each reload, which
    # can take 30–300 s and blocks the GPU for that entire period.
    #
    # When max_samples > 0 the process is restarted via os.execv after
    # every N samples, which already resets all Metal state cleanly.
    # Within a single max_samples run there is no benefit to mid-run
    # reloading — it only adds compilation overhead.
    if reload_model_interval == -1:
        if device_type == "mps":
            if max_samples > 0:
                # os.execv handles memory reset; no mid-run reload needed.
                effective_reload_interval = 0
            else:
                # Unlimited run: reload periodically to prevent RSS growth,
                # but much less frequently than the old default of 50.
                effective_reload_interval = 500
        else:
            effective_reload_interval = 0
    else:
        effective_reload_interval = reload_model_interval

    # Resolve canonical_input_length.
    # -1 is the sentinel from argparse; treat it the same as 0 (disabled).
    # The feature must be opted in explicitly via --canonical_input_length N
    # because many models (e.g. EQTransformer) use fixed internal windows that
    # break when the input tensor is padded to an unexpected length.
    if canonical_input_length <= 0:
        canonical_input_length = 0

    # Resolve picker backend from suffix unless explicitly specified.
    picker_backend = str(picker_backend).lower().strip()
    if picker_backend == "auto":
        picker_backend = infer_picker_backend_from_suffix(picker_model)
    if picker_backend not in ("torchscript", "onnx"):
        raise ValueError(
            f"Unsupported picker_backend={picker_backend!r}. Use 'auto', 'torchscript', or 'onnx'."
        )

    # Resolve DataLoader multiprocessing context.
    mp_context = get_dataloader_multiprocessing_context(
        num_workers, device_type, multiprocessing_context
    )

    print("=" * 80)
    print("[INFO] Starting phase picking")
    print(f"[INFO] HDF5 input: {h5_input}")
    print(f"[INFO] Output JSONL: {output_jsonl}")
    print(f"[INFO] Picker model: {picker_model}")
    print(f"[INFO] Picker backend: {picker_backend}")
    if str(picker_backend).lower() == "onnx":
        print(f"[INFO] ONNX picker postprocess: prob_thresh={onnx_prob_thresh}, nms_win={onnx_nms_win}")
        print(f"[INFO] ONNX provider request: {onnx_providers}")
    print(f"[INFO] Polarity model: {polar_model if polar_model else 'disabled'}")
    print(f"[INFO] Requested device: {device_name}")
    print(f"[INFO] Using torch device: {device}")
    print(f"[INFO] Target sampling rate: {target_sampling_rate}")
    print(f"[INFO] Resume: {resume}")
    print(f"[INFO] batch_size={batch_size}, num_workers={num_workers}")
    if num_workers > 0:
        print(f"[INFO] multiprocessing_context={mp_context!r}  prefetch_factor={prefetch_factor}")
    print(f"[INFO] mps_empty_cache_interval={mps_empty_cache_interval}")
    print(f"[INFO] cuda_empty_cache_interval={cuda_empty_cache_interval}")
    print(f"[INFO] reload_model_interval={effective_reload_interval}"
          f"  (requested={reload_model_interval})")
    print(f"[INFO] h5_rdcc_nbytes={h5_rdcc_nbytes // (1024*1024)} MB per file handle (only current file kept open)")
    print(f"[INFO] max_duration_sec={max_duration_sec}")
    print(
        f"[INFO] slow thresholds: load>{slow_load_threshold}s, infer>{slow_infer_threshold}s, "
        f"post>{slow_post_threshold}s, total>{slow_total_threshold}s"
    )
    _mps_str = str(max_picks_per_sample) if max_picks_per_sample > 0 else "0 (no limit)"
    print(f"[INFO] max_picks_per_sample={_mps_str}  flush_no_pick={flush_no_pick}")
    if canonical_input_length > 0:
        print(f"[INFO] canonical_input_length={canonical_input_length} samples "
              f"(all inputs padded/trimmed to this length to fix Metal pipeline cache growth)")
        if target_sampling_rate and target_sampling_rate > 0:
            full_day = int(86400 * target_sampling_rate)
            if canonical_input_length >= full_day:
                print(
                    f"[WARN] canonical_input_length={canonical_input_length} is a full day "
                    f"({full_day} @ {target_sampling_rate:.0f} Hz). Gappy or short station-days "
                    f"will be zero-padded to this length. For sliding-window models such as "
                    f"EQTransformer this multiplies the number of inference windows proportionally "
                    f"and can make per-sample runtime 10–100× slower on days with many gaps. "
                    f"Consider removing --canonical_input_length for EQTransformer."
                )
    else:
        print(f"[INFO] canonical_input_length=disabled")
        if target_sampling_rate and target_sampling_rate > 0:
            suggested = int(86400 * target_sampling_rate)
            # EQTransformer has an internal buffer of size 2*input_length; passing
            # exactly 2*input_length as an index overflows by 1.  Use suggested-2
            # as the safe EQTransformer ceiling.
            suggested_eqt = suggested - 2
            if device_type == "mps":
                print(
                    f"[WARN] MPS + canonical_input_length=disabled: every waveform with "
                    f"a unique sample count triggers a NEW Metal shader compilation. "
                    f"Python SIGALRM cannot interrupt Metal C-layer compilation, so the "
                    f"script will hang indefinitely on each new shape.\n"
                    f"[WARN] Fix: add  --canonical_input_length {suggested_eqt}  to your "
                    f"command. This pads/trims all waveforms to one shape so Metal "
                    f"compiles exactly once. ({suggested_eqt} = 86400 s × "
                    f"{target_sampling_rate:.0f} Hz − 2, safe for EQTransformer.)"
                )
            else:
                print(
                    f"[INFO] Tip: --canonical_input_length {suggested} "
                    f"({int(86400)} s × {target_sampling_rate:.0f} Hz) fixes the Metal "
                    f"kernel to one shape (MPS only). "
                    f"For EQTransformer use {suggested_eqt} instead (avoids internal OOB)."
                )
    if _SIGALRM_AVAILABLE and sample_timeout > 0:
        print(f"[INFO] sample_timeout={sample_timeout}s  "
              f"(SIGALRM watchdog; hangs exceeding this are written as error records)")
    else:
        reason = "disabled by --sample_timeout 0" if sample_timeout <= 0 else "SIGALRM not available on this platform"
        print(f"[INFO] sample_timeout=off  ({reason})")
    print("=" * 80)

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if picker_backend == "onnx":
        picker = load_onnx_picker_model(picker_model, device_type=device_type, providers=onnx_providers)
    else:
        picker = load_picker_model(picker_model, device, device_type)
    print(f"[INFO] Picker model loaded in {format_seconds(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    polar_sess = load_polar_model(polar_model, device_name=device_type, providers=onnx_providers)
    print(f"[INFO] Polarity model loaded in {format_seconds(time.perf_counter() - t0)}")

    t0 = time.perf_counter()
    dataset, resume_filter_is_loader_level = create_dataset_with_resume_filter(
        h5_input=h5_input,
        output_jsonl=output_jsonl,
        resume=resume,
        allowed_families=allowed_families,
        allowed_z_only_channels=allowed_z_only_channels,
        allow_z_only=allow_z_only,
        replicate_z_only=replicate_z_only,
        target_sampling_rate=target_sampling_rate,
        include_segments_metadata=include_segments_metadata,
        keep_h5_open=keep_h5_open,
        use_overlap_mask=use_overlap_mask,
        h5_rdcc_nbytes=h5_rdcc_nbytes,
        max_duration_sec=max_duration_sec,
    )

    total_dataset_samples = len(dataset)

    print(f"[INFO] Dataset indexed in {format_seconds(time.perf_counter() - t0)}")
    print(f"[INFO] Number of HDF5 files: {len(dataset.h5_files)}")
    if hasattr(dataset, "original_index_size"):
        print(f"[INFO] Original samples before resume filtering: {dataset.original_index_size}")
        print(f"[INFO] Samples filtered before waveform loading: {dataset.filtered_index_size}")
    print(f"[INFO] Number of samples to process now: {total_dataset_samples}")
    if max_samples > 0:
        restart_mode = "auto os.execv restart" if auto_restart else "exit code 75 (bash loop)"
        print(f"[INFO] max_samples={max_samples}: {restart_mode} after {max_samples} new samples.")
    elif device_type == "mps":
        print("[WARN] MPS detected with max_samples=0 (unlimited run).")
        print("[WARN] The Metal allocator and CoreML/ONNX Runtime accumulate process-level")
        print("[WARN] state that Python cannot free.  RSS will grow ~20-45 MB/sample early on.")
        print("[WARN] Strongly recommended: add --max_samples 200 (auto_restart is on by default)")
        print("[WARN] so the script restarts itself transparently until all samples are done.")
    if hasattr(dataset, "skip_jsonl_stats") and dataset.skip_jsonl_stats:
        print(f"[INFO] Resume JSONL stats: {dataset.skip_jsonl_stats}")
    print(f"[INFO] Loader-level resume filtering active: {resume_filter_is_loader_level}")

    if device_type == "mps":
        if num_workers == 0:
            print(
                "[INFO] Tip (MPS): --num_workers 1 --multiprocessing_context spawn "
                "enables a background CPU worker to prefetch+decode HDF5 waveforms "
                "while the GPU runs inference, which can significantly improve GPU "
                "utilisation. Workers never touch MPS; 'spawn' is macOS default."
            )
        else:
            # Workers run on CPU only (HDF5 decode). MPS state lives in the main
            # process. This is safe; just confirm spawn context is in use.
            if mp_context not in ("spawn", None):
                print(f"[WARN] MPS+num_workers: multiprocessing_context={mp_context!r}; "
                      f"consider 'spawn' (macOS default) to avoid h5py issues.")
            else:
                print(f"[INFO] MPS+num_workers={num_workers}: workers do CPU-only HDF5 "
                      f"decoding (safe). GPU inference runs in main process.")
    elif device_type == "cuda" and num_workers == 0:
        print("[INFO] Tip: --num_workers 4 (or more) enables parallel waveform "
              "prefetching on CUDA and can significantly improve throughput.")

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=waveform_collate_fn,
        # pin_memory speeds up CPU→GPU transfers on CUDA.
        pin_memory=(device_type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = mp_context
        # hdf5_worker_init_fn resets inherited h5 handles when using 'fork'.
        # It is a no-op with 'spawn'/'forkserver' but safe to always pass.
        loader_kwargs["worker_init_fn"] = hdf5_worker_init_fn

    loader = DataLoader(dataset, **loader_kwargs)

    # ── Metal kernel pre-warmup (MPS only) ──────────────────────────────────
    # On the first run (or after a Metal cache clear), PyTorch/Metal must
    # JIT-compile GPU shaders for every unique input tensor shape encountered.
    # For a model like EQTransformer this can take 5–15 minutes with ZERO
    # console output, making the script appear completely hung.
    #
    # Running a dummy inference here triggers compilation *before* the main
    # loop so the user can see what's happening.  No SIGALRM applies during
    # warmup; the per-sample timeout only guards the main loop.
    # Compiled kernels are cached to ~/Library/Caches/com.apple.metal/ and
    # are re-used automatically on subsequent runs (warmup then takes < 1 s).
    if device_type == "mps" and picker_backend == "torchscript":
        _sr = target_sampling_rate if (target_sampling_rate and target_sampling_rate > 0) else 100.0
        if canonical_input_length > 0:
            _warmup_len = canonical_input_length
        else:
            # canonical_input_length is disabled: warmup uses the full-day shape,
            # but real waveforms will differ → Metal recompiles per unique length.
            # SIGALRM cannot interrupt Metal C-layer compilation, so those will hang.
            # The warning above (at startup) already told the user to add the flag.
            _warmup_len = int(86400 * _sr) - 2  # EQTransformer-safe default
        print(
            f"[INFO] MPS: running Metal kernel pre-warmup "
            f"(dummy inference on a {_warmup_len:,}-sample zero tensor). "
            f"If this is the first run for this model, Metal shader compilation "
            f"may take 5–15 min — this is NORMAL and will only happen once. "
            f"Kernels are cached to ~/Library/Caches/com.apple.metal/.",
            flush=True,
        )
        _t_warmup = time.perf_counter()
        try:
            with torch.inference_mode():
                # Shape must match what run_torchscript_picker_from_tensor passes:
                # ensure_waveform_tensor_for_picker returns [T, 3] (2-D, not batched).
                _dummy_cpu = torch.zeros(_warmup_len, 3, dtype=torch.float32)
                _dummy_gpu = _dummy_cpu.to(device)
                del _dummy_cpu
                _warmup_out = picker(_dummy_gpu)
                del _dummy_gpu, _warmup_out
                torch.mps.synchronize()
            _warmup_elapsed = time.perf_counter() - _t_warmup
            if _warmup_elapsed >= 5.0:
                print(
                    f"[INFO] Metal warmup complete in {format_seconds(_warmup_elapsed)}. "
                    f"Kernels are now cached — main loop inference will be fast.",
                    flush=True,
                )
            else:
                print(
                    f"[INFO] Metal warmup complete in {format_seconds(_warmup_elapsed)} "
                    f"(kernels were already cached).",
                    flush=True,
                )
        except Exception as _warmup_exc:
            print(
                f"[WARN] Metal warmup failed: {_warmup_exc}  "
                f"Continuing — first sample in the main loop may be slow.",
                flush=True,
            )

    total_seen = 0
    total_processed = 0
    total_picks = 0
    total_errors = 0
    total_written_lines = 0

    t_loop0 = time.perf_counter()
    t_last = t_loop0

    open_mode = "a" if resume else "w"
    _max_samples_reached = False

    print("[INFO] Entering main processing loop...", flush=True)
    try:
        with open(output_jsonl, open_mode, encoding="utf-8", buffering=1024 * 1024) as f:
            # ── Manual DataLoader iteration ────────────────────────────────
            # We use iter(loader)+next() instead of `for batch in loader` so
            # that we can arm the SIGALRM watchdog BEFORE the HDF5 read.
            # With num_workers=0, `next(loader_iter)` calls dataset.__getitem__
            # synchronously in the main thread.  A blocked HDF5 syscall IS
            # interruptible by SIGALRM (POSIX signals interrupt IO syscalls),
            # so arming here covers both the waveform-load and inference phases.
            loader_iter = iter(loader)
            _loop_done = False
            while not _loop_done:
                # Declare per-sample state before arming the watchdog so the
                # outer except and finally can always reference these names.
                sample_key = ""
                item = None
                z = picks = x_cpu = waveform = original_length = None
                _hdf5_fetched = False

                _timer = _SampleTimer(sample_timeout)
                _timer.__enter__()
                # Print a loading marker before every HDF5 read so slow reads
                # are never confused with a hung process.
                _load_t0 = time.perf_counter()
                #print(
                #    f"[LOAD] {total_seen + 1}/{total_dataset_samples} "
                #    f"(HDF5 read)...",
                #    flush=True,
                #)
                try:
                    # ── Fetch next batch (HDF5 waveform read) ─────────────
                    # With num_workers=0 this runs in the main thread; the HDF5
                    # C library retries EINTR internally, so SIGALRM cannot
                    # reliably interrupt it.  Use --num_workers 1 to move the
                    # read into a subprocess; the main process then blocks on
                    # queue.get() which IS Python-level and interruptible.
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        _loop_done = True
                        break       # outer finally still runs to disarm timer

                    _load_elapsed = time.perf_counter() - _load_t0
                    _hdf5_fetched = True
                    if _load_elapsed >= slow_load_threshold:
                        print(
                            f"[SLOW_LOAD] next_batch={format_seconds(_load_elapsed)} "
                            f"batch_size={len(batch)} num_workers={num_workers} "
                            f"prefetch_factor={prefetch_factor}",
                            flush=True,
                        )
                    else:
                        #print(
                        #    f"[LOAD_DONE] next_batch={format_seconds(_load_elapsed)} "
                        #    f"batch_size={len(batch)}",
                        #    flush=True,
                        #)
                        pass

                    for item in batch:
                        _sample_t0 = time.perf_counter()
                        _prep_elapsed = 0.0
                        _infer_elapsed = 0.0
                        _post_elapsed = 0.0
                        _pick_count_written = 0
                        _pick_count_model = 0
                        _pick_count_after_filter = 0
                        _post_t0 = None
                        sample_meta = {}
                        total_seen += 1
                        total_processed += 1

                        # Capture the sample key and lightweight metadata now, before
                        # deleting waveform tensors.  This is used for slow-sample logs
                        # and error records.
                        sample_key = make_sample_key_from_item(item)
                        sample_meta = {
                            "station_id": item.get("station_id", ""),
                            "h5_file": item.get("h5_file", ""),
                            "year_id": item.get("year_id", ""),
                            "day_id": item.get("day_id", ""),
                            "channels": item.get("channels", []),
                            "channel_family": item.get("channel_family", ""),
                            "sampling_rate": item.get("sampling_rate", None),
                            "original_sampling_rate": item.get("original_sampling_rate", None),
                            "starttime": item.get("starttime", ""),
                            "endtime": item.get("endtime", ""),
                            "npts": item.get("npts", None),
                        }

                        try:
                            # ── Per-sample inference ───────────────────────
                            # NOTE: _timer is already armed above (before the
                            # HDF5 read); do NOT create or re-arm it here.
                            _prep_t0 = time.perf_counter()
                            waveform = item["waveform"]
                            # Build the full [T, 3] CPU tensor exactly once.
                            # This avoids two large copies for day-long waveforms.
                            x_cpu = ensure_waveform_tensor_for_picker(waveform)
                            original_length = x_cpu.shape[0]
                            _prep_elapsed = time.perf_counter() - _prep_t0

                            # ── Canonical-length padding ───────────────────
                            # On MPS, TorchScript compiles and caches a separate
                            # Metal pipeline for every unique input tensor shape.
                            # Padding every input to one canonical length means
                            # Metal compiles exactly ONE set of kernels and never
                            # allocates another pipeline object.
                            # Picks whose sample_index >= original_length are
                            # filtered out below to suppress padding artefacts.
                            if canonical_input_length > 0 and original_length != canonical_input_length:
                                if original_length < canonical_input_length:
                                    pad = torch.zeros(
                                        canonical_input_length - original_length,
                                        x_cpu.shape[1],
                                        dtype=x_cpu.dtype,
                                    )
                                    x_cpu = torch.cat([x_cpu, pad], dim=0)
                                else:
                                    x_cpu = x_cpu[:canonical_input_length]

                            z = x_cpu[:original_length, 2].numpy()

                            _t_infer = time.perf_counter()
                            _onnx_prob_rows = 0
                            _onnx_prob_shape = ""
                            if picker_backend == "onnx":
                                picks, _prob_raw, _time_raw = run_onnx_picker_from_tensor(
                                    picker,
                                    x_cpu,
                                    prob_thresh=onnx_prob_thresh,
                                    nms_win=onnx_nms_win,
                                )
                                _onnx_prob_rows = int(_prob_raw.shape[0]) if hasattr(_prob_raw, "shape") else 0
                                _onnx_prob_shape = safe_shape_text(_prob_raw)
                                # Release dense ONNX outputs immediately; they can be large for full-day traces.
                                del _prob_raw, _time_raw
                            else:
                                picks = run_torchscript_picker_from_tensor(
                                    picker,
                                    x_cpu,
                                    device=device,
                                )
                            _infer_elapsed = time.perf_counter() - _t_infer
                            _pick_count_model = int(len(picks)) if picks is not None else 0
                            # Print one line per sample so slow HDF5 reads and slow
                            # inferences are both immediately visible in the log.
                            _extra = ""
                            if picker_backend == "onnx":
                                _extra = f" prob_shape={_onnx_prob_shape} prob_rows={_onnx_prob_rows}"
                            #print(
                            #    f"[INFER] {total_processed}: "
                            #    f"{original_length:,}samp "
                            #    f"backend={picker_backend} "
                            #    f"prep={format_seconds(_prep_elapsed)} "
                            #    f"infer+postnms={format_seconds(_infer_elapsed)} "
                            #    f"model_picks={_pick_count_model}"
                            #    f"{_extra}",
                            #    flush=True,
                            #)

                            # Remove any picks the model placed in the padded tail.
                            if canonical_input_length > 0 and picks is not None and len(picks) > 0:
                                picks = picks[picks[:, 1] < original_length]

                        except Exception as e:
                            _timer.__exit__(type(e), e, None)   # disarm SIGALRM
                            total_errors += 1
                            is_timeout = isinstance(e, _SampleTimeout)
                            err_record = {
                                "record_type": "error",
                                "station_id": item.get("station_id", ""),
                                "h5_file": item.get("h5_file", ""),
                                "year_id": item.get("year_id", ""),
                                "day_id": item.get("day_id", ""),
                                "sample_key": sample_key,
                                "error": str(e),
                            }
                            f.write(_json_dumps(to_jsonable(err_record)) + "\n")
                            f.flush()
                            total_written_lines += 1
                            try:
                                if item is not None and "waveform" in item:
                                    del item["waveform"]
                            except Exception:
                                pass
                            picks = z = x_cpu = waveform = None
                            item = None
                            if is_timeout:
                                # A timeout means a Metal / ONNX operation may still
                                # be in-flight.  Do NOT call synchronize() or
                                # empty_cache() — they can themselves block on the
                                # orphaned operation and cause a cascade hang.
                                print(
                                    f"[WARN] Sample timed out after {sample_timeout}s "
                                    f"(skipping, Metal ops will drain at next restart): "
                                    f"{sample_key}",
                                    flush=True,
                                )
                            else:
                                if device_type == "mps":
                                    force_device_cleanup(device_type, do_gc=False)
                            continue

                        _picks_before = total_picks
                        try:
                            # Cap picks per station-day.
                            _above = picks[picks[:, 2] >= min_confidence] if min_confidence > 0 else picks
                            if max_picks_per_sample > 0 and len(_above) > max_picks_per_sample:
                                _order = _above[:, 2].argsort()[::-1]
                                _above = _above[_order[:max_picks_per_sample]]
                            picks = _above
                            _pick_count_after_filter = int(len(picks)) if picks is not None else 0
                            _post_t0 = time.perf_counter()

                            for row in picks:
                                phase_id = int(row[0])
                                sample_index = float(row[1])
                                confidence = float(row[2])

                                record = build_pick_record(
                                    item=item,
                                    z=z,
                                    phase_id=phase_id,
                                    sample_index=sample_index,
                                    confidence=confidence,
                                    polar_sess=polar_sess,
                                    snr_window_sec=snr_window_sec,
                                )

                                f.write(_json_dumps(record) + "\n")
                                total_picks += 1
                                total_written_lines += 1
                                _pick_count_written += 1

                        finally:
                            if _post_t0 is not None:
                                try:
                                    _post_elapsed = time.perf_counter() - _post_t0
                                except Exception:
                                    _post_elapsed = 0.0
                            # Disarm the SIGALRM watchdog on the normal success path.
                            _timer.__exit__(None, None, None)
                            picks = z = x_cpu = waveform = None
                            try:
                                del item["waveform"]
                            except Exception:
                                pass
                            item = None

                        # ── No-pick sentinel ───────────────────────────────
                        # If this station-day produced zero detections write a
                        # lightweight record so the resume scanner marks it done.
                        if total_picks == _picks_before:
                            _no_pick = {"record_type": "no_pick", "sample_key": sample_key}
                            f.write(_json_dumps(_no_pick) + "\n")
                            if flush_no_pick:
                                f.flush()
                            total_written_lines += 1

                        _sample_elapsed = time.perf_counter() - _sample_t0
                        if (
                            _load_elapsed >= slow_load_threshold
                            or _infer_elapsed >= slow_infer_threshold
                            or _post_elapsed >= slow_post_threshold
                            or _sample_elapsed >= slow_total_threshold
                        ):
                            _meta = dict(sample_meta)
                            if _meta.get("npts") is None:
                                _meta["npts"] = original_length
                            print(
                                "[SLOW_SAMPLE] "
                                f"idx={total_processed} "
                                f"load_batch={format_seconds(_load_elapsed)} "
                                f"prep={format_seconds(_prep_elapsed)} "
                                f"infer={format_seconds(_infer_elapsed)} "
                                f"post_write={format_seconds(_post_elapsed)} "
                                f"sample_total={format_seconds(_sample_elapsed)} "
                                f"model_picks={_pick_count_model} "
                                f"kept_picks={_pick_count_after_filter} "
                                f"written_picks={_pick_count_written} "
                                f"shape={original_length:,}x3 "
                                f"meta={to_jsonable(_meta)}",
                                flush=True,
                            )

                        # ── Per-sample allocator cleanup ───────────────────
                        if device_type == "mps" and mps_empty_cache_interval > 0 and total_processed % mps_empty_cache_interval == 0:
                            force_device_cleanup(device_type, do_gc=True)
                        elif device_type == "cuda" and cuda_empty_cache_interval > 0 and total_processed % cuda_empty_cache_interval == 0:
                            force_device_cleanup(device_type, do_gc=False)

                        # ── Periodic h5py handle flush ─────────────────────
                        if effective_reload_interval > 0 and total_processed % effective_reload_interval == 0:
                            try:
                                dataset.flush_h5_cache()
                            except Exception:
                                pass

                        # ── Periodic model reload ──────────────────────────
                        if effective_reload_interval > 0 and total_processed % effective_reload_interval == 0:
                            sync_device(device_type)
                            del picker
                            if polar_sess is not None:
                                del polar_sess
                                polar_sess = None
                            gc.collect()
                            empty_device_cache(device_type)
                            if picker_backend == "onnx":
                                picker = load_onnx_picker_model(picker_model, device_type=device_type, providers=onnx_providers)
                            else:
                                picker = load_picker_model(picker_model, device, device_type)
                            if polar_model:
                                polar_sess = load_polar_model(polar_model, device_name=device_type, providers=onnx_providers)
                            gc.collect()
                            empty_device_cache(device_type)

                        if flush_interval > 0 and total_processed % flush_interval == 0:
                            f.flush()

                        if gc_interval > 0 and total_processed % gc_interval == 0 and device_type != "mps":
                            gc.collect()
                            empty_device_cache(device_type)

                        if (
                            total_seen % max(1, progress_interval) == 0
                            or total_seen == total_dataset_samples
                        ):
                            sync_device(device_type)
                            now = time.perf_counter()
                            elapsed = now - t_loop0
                            speed = total_seen / max(elapsed, 1e-6)

                            if total_dataset_samples > 0:
                                progress = total_seen / total_dataset_samples
                                remaining = total_dataset_samples - total_seen
                                eta = remaining / max(speed, 1e-6)
                            else:
                                progress = 0.0
                                eta = 0.0

                            recent_elapsed = now - t_last
                            t_last = now

                            mem_text = get_device_memory_text(device_type)
                            print(
                                "[PROGRESS] "
                                f"{total_seen}/{total_dataset_samples} "
                                f"({progress * 100:.2f}%) | "
                                f"processed={total_processed} | "
                                f"picks={total_picks} | "
                                f"errors={total_errors} | "
                                f"speed={speed:.3f} samples/s | "
                                f"elapsed={format_seconds(elapsed)} | "
                                f"eta={format_seconds(eta)} | "
                                f"last_interval={format_seconds(recent_elapsed)}"
                                f"{mem_text}"
                            )

                        # ── max_samples early-exit ─────────────────────────
                        if max_samples > 0 and total_processed >= max_samples:
                            _max_samples_reached = True
                            break   # break for-item loop

                except _SampleTimeout:
                    # This fires only when the timeout occurs in next(loader_iter)
                    # (HDF5 IO).  Timeouts during inference are _SampleTimeout
                    # subclasses of Exception and are caught by the inner except.
                    #
                    # NOTE: with num_workers=0 the HDF5 C library retries EINTR
                    # internally, so SIGALRM may not fire until the read eventually
                    # completes or exceeds sample_timeout at a Python check-point.
                    # With num_workers>=1 the main process blocks in Python-level
                    # queue.get(), which IS interruptible, so timeout is reliable.
                    total_errors += 1
                    err_record = {
                        "record_type": "error",
                        "sample_key": sample_key,   # "" if timeout before item was known
                        "error": f"HDF5 read timed out after {sample_timeout}s",
                    }
                    f.write(_json_dumps(to_jsonable(err_record)) + "\n")
                    f.flush()
                    total_written_lines += 1
                    print(
                        f"[WARN] HDF5 read timed out after {sample_timeout}s "
                        f"(sample will be retried on next run)",
                        flush=True,
                    )
                    # With num_workers > 0 the worker process that hung is still
                    # running in the background.  Recreating the iterator drops the
                    # reference to the old loader state and spawns a fresh worker,
                    # so subsequent samples are not blocked by the stuck process.
                    if num_workers > 0:
                        print(
                            "[INFO] Restarting DataLoader to replace the hung "
                            "worker process.",
                            flush=True,
                        )
                        try:
                            del loader_iter
                        except Exception:
                            pass
                        loader_iter = iter(loader)
                    # Do NOT call synchronize()/empty_cache() after a timeout.

                finally:
                    # Always disarm; calling __exit__ multiple times is safe
                    # (signal.alarm(0) is idempotent).
                    _timer.__exit__(None, None, None)
                    # Release any refs left by an HDF5-timeout path.
                    picks = z = x_cpu = waveform = None
                    if item is not None:
                        try:
                            del item["waveform"]
                        except Exception:
                            pass
                    item = None

                if _max_samples_reached:
                    break   # break while loop

            f.flush()

    finally:
        try:
            dataset.close()
        except Exception:
            pass
        gc.collect()
        empty_device_cache(device_type)

    total_elapsed = time.perf_counter() - t_all0
    loop_elapsed = time.perf_counter() - t_loop0

    print("=" * 80)
    print("[OK] Phase picking finished")
    print(f"[OK] Dataset samples seen by DataLoader: {total_seen}")
    print(f"[OK] Samples newly processed: {total_processed}")
    print(f"[OK] Phase picks written: {total_picks}")
    print(f"[OK] JSONL lines written this run: {total_written_lines}")
    print(f"[OK] Errors: {total_errors}")
    print(f"[OK] Output JSONL: {output_jsonl}")
    print(f"[OK] Processing time: {format_seconds(loop_elapsed)}")
    print(f"[OK] Total wall time: {format_seconds(total_elapsed)}")
    if hasattr(dataset, "filtered_index_size"):
        print(f"[OK] Samples skipped before waveform loading: {dataset.filtered_index_size}")
    if total_seen > 0:
        print(f"[OK] Average speed: {total_seen / max(loop_elapsed, 1e-6):.3f} samples/s")
        print(f"[OK] Average time per sample: {loop_elapsed / total_seen:.3f} s/sample")
    if total_picks > 0:
        print(f"[OK] Average time per pick: {loop_elapsed / total_picks:.3f} s/pick")
    if _max_samples_reached:
        print(f"[OK] Stopped early: max_samples={max_samples} reached.")
        if auto_restart:
            print(f"[OK] auto_restart=True — re-executing this process now to free all OS-level "
                  f"Metal / CoreML allocator state (RSS will reset to baseline).")
        else:
            print(f"[OK] auto_restart=False — exiting with code 75. "
                  f"Re-run with --resume to continue from the next sample.")
    print("=" * 80)

    if _max_samples_reached:
        sys.stdout.flush()
        sys.stderr.flush()
        if auto_restart:
            # os.execv replaces this process image with a fresh Python process running
            # the same script and arguments.  The OS reclaims ALL Metal / CoreML /
            # HDF5 allocator state (things that empty_cache and gc.collect cannot touch).
            # --resume is enabled by default so the new process skips already-written
            # samples at the dataset level without re-reading any waveform data.
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            sys.exit(75)


def parse_tuple_arg(value):
    if value is None or str(value).strip() == "":
        return tuple()
    return tuple(x.strip().upper() for x in str(value).split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Run TorchScript or ONNX seismic picker on HDF5 dataloader samples and write JSONL output."
    )

    parser.add_argument(
        "--h5_input",
        default="data/hdf5/continuous_waveform_usa_*.h5",
        help='HDF5 file, directory, or glob pattern.',
    )
    parser.add_argument(
        "--output_jsonl",
        default="data/picks/pnsn.v1.phase.jsonl",
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--picker_model",
        default="pickers/pnsn.v1.diff.jit",
        help="Picker model path. Suffix .onnx uses ONNX Runtime; .jit/.torchscript uses TorchScript by default.",
    )
    parser.add_argument(
        "--picker_backend",
        default="auto",
        choices=["auto", "torchscript", "onnx"],
        help=(
            "Picker backend. Default 'auto' chooses by --picker_model suffix: "
            ".onnx -> ONNX Runtime + external heap-NMS; "
            ".jit/.torchscript -> TorchScript. "
            "Use 'torchscript' or 'onnx' to override."
        ),
    )
    parser.add_argument(
        "--onnx_providers",
        default="auto",
        help=(
            "ONNX Runtime provider list. Use 'auto' to select CUDA for --device cuda, "
            "CoreML for --device mps, and CPU otherwise. You can also pass an explicit "
            "comma-separated list such as CUDAExecutionProvider,CPUExecutionProvider."
        ),
    )
    parser.add_argument(
        "--onnx_prob_thresh",
        type=float,
        default=0.1,
        help="Probability threshold for external heap-NMS ONNX picker post-processing.",
    )
    parser.add_argument(
        "--onnx_nms_win",
        type=int,
        default=200,
        help="NMS window in samples for external heap-NMS ONNX picker post-processing.",
    )
    parser.add_argument(
        "--polar_model",
        default=None,
        help="Optional ONNX polar model path. Use empty string to disable.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Inference device: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size. Samples are still processed one by one inside each batch.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader worker processes for parallel waveform prefetching. "
            "0 = single-process (default). "
            "MPS: 1 worker overlaps HDF5 decode (CPU) with GPU inference; workers never "
            "touch MPS so this is safe — use with --multiprocessing_context spawn "
            "(macOS default). "
            "CUDA: 4–8 workers is a good starting point to hide HDF5 I/O latency. "
            "On Linux, workers use 'spawn' by default (see --multiprocessing_context) "
            "to avoid h5py + fork conflicts."
        ),
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help=(
            "Number of batches pre-loaded per worker (PyTorch default=2). "
            "Reduce to 1 if workers OOM on large waveforms. "
            "Ignored when num_workers=0."
        ),
    )
    parser.add_argument(
        "--multiprocessing_context",
        default="auto",
        help=(
            "Multiprocessing start method for DataLoader workers: "
            "'auto' (default) selects 'spawn' on Linux to avoid h5py+fork issues, "
            "and the OS default elsewhere. "
            "Other valid values: 'spawn', 'fork', 'forkserver'. "
            "If you use 'fork', the hdf5_worker_init_fn is automatically applied "
            "to reset inherited HDF5 file handles in each worker. "
            "Ignored when num_workers=0."
        ),
    )
    parser.add_argument(
        "--allowed_families",
        default="HH,BH,EH,HN",
        help='Comma-separated channel families, e.g. "HH,BH,EH,HN".',
    )
    parser.add_argument(
        "--allowed_z_only_channels",
        default="EHZ",
        help='Comma-separated Z-only channels, e.g. "EHZ".',
    )
    parser.add_argument(
        "--allow_z_only",
        action="store_true",
        default=True,
        help="Allow Z-only samples.",
    )
    parser.add_argument(
        "--no_z_only",
        action="store_false",
        dest="allow_z_only",
        help="Disable Z-only samples.",
    )
    parser.add_argument(
        "--replicate_z_only",
        action="store_true",
        default=True,
        help="Replicate Z-only samples to [Z, Z, Z].",
    )
    parser.add_argument(
        "--no_replicate_z_only",
        action="store_false",
        dest="replicate_z_only",
        help="Do not replicate Z-only samples.",
    )
    parser.add_argument(
        "--target_sampling_rate",
        type=float,
        default=100.0,
        help="Target sampling rate in Hz. Use -1 to disable resampling.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="Minimum pick confidence to write.",
    )
    parser.add_argument(
        "--snr_window_sec",
        type=float,
        default=2.0,
        help="SNR window length before and after pick, in seconds.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Print progress every N samples.",
    )
    parser.add_argument(
        "--flush_interval",
        type=int,
        default=100,
        help="Flush JSONL file every N processed samples. Use 0 to flush only at the end.",
    )
    parser.add_argument(
        "--gc_interval",
        type=int,
        default=500,
        help="Run gc.collect and empty device cache every N samples. Use 0 to disable.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing JSONL and skip already processed samples before waveform loading.",
    )
    parser.add_argument(
        "--no_resume",
        action="store_false",
        dest="resume",
        help="Disable resume mode and overwrite output JSONL.",
    )
    parser.add_argument(
        "--include_segments_metadata",
        action="store_true",
        default=False,
        help="Return segment metadata from the loader. Default False saves memory.",
    )
    parser.add_argument(
        "--no_keep_h5_open",
        action="store_false",
        dest="keep_h5_open",
        default=True,
        help="Disable per-process cached HDF5 handles.",
    )
    parser.add_argument(
        "--mps_empty_cache_interval",
        type=int,
        default=500,
        help="For MPS, synchronize + gc.collect + torch.mps.empty_cache every N processed samples. Use 0 to disable.",
    )
    parser.add_argument(
        "--cuda_empty_cache_interval",
        type=int,
        default=500,
        help="For CUDA, synchronize + torch.cuda.empty_cache every N processed samples. Use 0 to disable.",
    )
    parser.add_argument(
        "--reload_model_interval",
        type=int,
        default=-1,
        help=(
            "Reload TorchScript model every N processed samples. "
            "-1 (default) = auto: 50 for MPS (Metal allocator state accumulates "
            "with each TorchScript call; reload is the only way to reset it), "
            "0 (disabled) for CUDA and CPU. "
            "Increase if reload overhead is noticeable on very short waveforms. "
            "Set 0 to disable entirely."
        ),
    )
    parser.add_argument(
        "--no_overlap_mask",
        action="store_false",
        dest="use_overlap_mask",
        default=True,
        help="Disable overlap mask in fill_segments_to_array to reduce memory.",
    )
    parser.add_argument(
        "--h5_rdcc_nbytes",
        type=int,
        default=8 * 1024 * 1024,
        help=(
            "HDF5 raw-data chunk cache size in bytes per open file handle "
            "(default: 8388608 = 8 MB). h5py's built-in default is 1 MB. "
            "8 MB is enough for single-pass inference; only the current file's "
            "handle is kept open — stale file handles are closed as soon as the "
            "loader moves to the next file, so chunk-cache memory stays O(1). "
            "Raise to 64 MB for repeated-access / training workloads."
        ),
    )
    parser.add_argument(
        "--canonical_input_length",
        type=int,
        default=0,
        help=(
            "Pad or trim every waveform to exactly N samples before sending to "
            "the picker model. Default 0 = disabled (recommended). "
            "WARNING: many models (e.g. EQTransformer) use fixed-size internal "
            "windows and will crash or produce wrong results if the input tensor "
            "length differs from what the model expects. Only enable if you know "
            "your model supports variable-length inputs and benefits from a fixed "
            "shape (e.g. to control TorchScript Metal pipeline cache growth). "
            "Picks placed in the padded tail (sample_index >= original length) "
            "are filtered out automatically."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "Restart the process after processing N new samples. "
            "With --auto_restart (default), the script calls os.execv to replace "
            "itself with a fresh process, freeing all Metal / CoreML / HDF5 "
            "allocator state that Python cannot reclaim. "
            "--resume is enabled by default, so the new process skips already-written "
            "samples at the dataset level. "
            "0 = no limit (process all samples without restart). "
            "Recommended on Mac/MPS to keep RSS bounded: --max_samples 200."
        ),
    )
    parser.add_argument(
        "--auto_restart",
        action="store_true",
        default=True,
        help=(
            "When --max_samples is reached, use os.execv to restart this process "
            "instead of exiting with code 75. "
            "The OS reclaims all Metal / CoreML / HDF5 allocator state on process "
            "replacement; the new process resumes from where the previous one stopped. "
            "Default: True (recommended for MPS). "
            "Use --no_auto_restart to get exit-code-75 behavior for external bash loops."
        ),
    )
    parser.add_argument(
        "--no_auto_restart",
        action="store_false",
        dest="auto_restart",
        help="Disable auto os.execv restart; exit with code 75 when --max_samples is reached.",
    )
    parser.add_argument(
        "--sample_timeout",
        type=int,
        default=600,
        help=(
            "Maximum seconds allowed to process a single sample before the "
            "watchdog (SIGALRM) fires and the sample is written as an error "
            "record.  Prevents the script from hanging indefinitely on a stuck "
            "Metal / ONNX Runtime operation.  "
            "0 = disabled.  Default: 120 s.  "
            "Only effective on Unix/macOS (SIGALRM is not available on Windows)."
        ),
    )
    parser.add_argument(
        "--max_picks_per_sample",
        type=int,
        default=0,
        help=(
            "Maximum phase detections to write per station-day.  When more "
            "picks are detected (e.g. on a major-earthquake day with thousands "
            "of aftershocks), the top-N by confidence are kept and the rest "
            "are discarded.  This prevents the per-pick CoreML polarity loop "
            "from running for hundreds of seconds and triggering the "
            "--sample_timeout watchdog.  0 = no limit (default). "
            "Recommended on MPS/Apple Silicon when processing high-seismicity days: 500-2000."
        ),
    )
    parser.add_argument(
        "--max_duration_sec",
        type=float,
        default=90000.0,
        help=(
            "Maximum allowed time span for one consolidated channel waveform. "
            "Default 90000 s = 25 h. If a channel has bad start/end metadata and "
            "would allocate an abnormal array, the loader raises an error instead "
            "of appearing to hang."
        ),
    )
    parser.add_argument(
        "--slow_load_threshold",
        type=float,
        default=10.0,
        help="Print [SLOW_LOAD]/[SLOW_SAMPLE] when DataLoader/HDF5 batch fetch exceeds this many seconds.",
    )
    parser.add_argument(
        "--slow_infer_threshold",
        type=float,
        default=10.0,
        help="Print [SLOW_SAMPLE] when TorchScript model inference exceeds this many seconds.",
    )
    parser.add_argument(
        "--slow_post_threshold",
        type=float,
        default=10.0,
        help="Print [SLOW_SAMPLE] when post-processing / SNR / polarity / JSONL writing exceeds this many seconds.",
    )
    parser.add_argument(
        "--slow_total_threshold",
        type=float,
        default=30.0,
        help="Print [SLOW_SAMPLE] when total per-sample time excluding batch fetch exceeds this many seconds.",
    )
    parser.add_argument(
        "--flush_no_pick",
        action="store_true",
        default=False,
        help=(
            "Flush output JSONL immediately after every no_pick sentinel. "
            "Default False because flushing every no-pick sample can be slow on "
            "external disks or network filesystems."
        ),
    )
    args = parser.parse_args()

    target_sampling_rate = args.target_sampling_rate
    if target_sampling_rate is not None and target_sampling_rate <= 0:
        target_sampling_rate = None

    run_picker_to_jsonl(
        h5_input=args.h5_input,
        output_jsonl=args.output_jsonl,
        picker_model=args.picker_model,
        polar_model=args.polar_model if args.polar_model else None,
        picker_backend=args.picker_backend,
        onnx_providers=args.onnx_providers,
        onnx_prob_thresh=args.onnx_prob_thresh,
        onnx_nms_win=args.onnx_nms_win,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        allowed_families=parse_tuple_arg(args.allowed_families),
        allowed_z_only_channels=parse_tuple_arg(args.allowed_z_only_channels),
        allow_z_only=args.allow_z_only,
        replicate_z_only=args.replicate_z_only,
        target_sampling_rate=target_sampling_rate,
        min_confidence=args.min_confidence,
        snr_window_sec=args.snr_window_sec,
        progress_interval=args.progress_interval,
        resume=args.resume,
        flush_interval=args.flush_interval,
        gc_interval=args.gc_interval,
        include_segments_metadata=args.include_segments_metadata,
        keep_h5_open=args.keep_h5_open,
        use_overlap_mask=args.use_overlap_mask,
        mps_empty_cache_interval=args.mps_empty_cache_interval,
        cuda_empty_cache_interval=args.cuda_empty_cache_interval,
        reload_model_interval=args.reload_model_interval,
        multiprocessing_context=args.multiprocessing_context,
        prefetch_factor=args.prefetch_factor,
        h5_rdcc_nbytes=args.h5_rdcc_nbytes,
        max_samples=args.max_samples,
        canonical_input_length=args.canonical_input_length,
        auto_restart=args.auto_restart,
        sample_timeout=args.sample_timeout,
        max_picks_per_sample=args.max_picks_per_sample,
        max_duration_sec=args.max_duration_sec,
        slow_load_threshold=args.slow_load_threshold,
        slow_infer_threshold=args.slow_infer_threshold,
        slow_post_threshold=args.slow_post_threshold,
        slow_total_threshold=args.slow_total_threshold,
        flush_no_pick=args.flush_no_pick,
    )


if __name__ == "__main__":
    main()
