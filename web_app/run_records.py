"""Persistent run records for SAGE research workflows."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from sage_paths import sage_home


_FALLBACK_RUN_DIR = Path(__file__).parent / "outputs" / "runs"
_DEFAULT_RUN_DIR = Path(os.environ.get("SAGE_RUN_RECORD_DIR", sage_home("runs")))
try:
    _DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    _DEFAULT_RUN_DIR = _FALLBACK_RUN_DIR
    _DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)

RUN_RECORD_DIR = _DEFAULT_RUN_DIR

_lock = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _record_path(run_id: str) -> Path:
    return RUN_RECORD_DIR / f"{run_id}.json"


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(path: Path, record: dict):
    global RUN_RECORD_DIR
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(
            json.dumps(_json_safe(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)
    except PermissionError:
        if RUN_RECORD_DIR == _FALLBACK_RUN_DIR:
            raise
        RUN_RECORD_DIR = _FALLBACK_RUN_DIR
        RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
        fallback_path = RUN_RECORD_DIR / path.name
        fallback_tmp = fallback_path.with_suffix(".json.tmp")
        fallback_tmp.write_text(
            json.dumps(_json_safe(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fallback_tmp.replace(fallback_path)


def start_run(
    kind: str,
    request: str = "",
    session_id: str = "",
    metadata: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> str:
    rid = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record = {
        "run_id": rid,
        "kind": kind,
        "status": "running",
        "request": request,
        "session_id": session_id,
        "metadata": metadata or {},
        "started_at": _now(),
        "updated_at": _now(),
        "events": [],
        "artifacts": [],
    }
    with _lock:
        _write(_record_path(rid), record)
    return rid


def append_event(run_id: str, phase: str, message: str = "", data: Optional[dict] = None):
    with _lock:
        path = _record_path(run_id)
        record = _read(path)
        if not record:
            return
        record.setdefault("events", []).append({
            "ts": _now(),
            "phase": phase,
            "message": message,
            "data": data or {},
        })
        record["updated_at"] = _now()
        _write(path, record)


def finish_run(
    run_id: str,
    status: str,
    result: Optional[dict] = None,
    error: str = "",
    artifacts: Optional[list] = None,
):
    with _lock:
        path = _record_path(run_id)
        record = _read(path)
        if not record:
            return
        record["status"] = status
        record["finished_at"] = _now()
        record["updated_at"] = _now()
        record["duration_s"] = max(
            0.0,
            time.time() - _timestamp(record.get("started_at")),
        )
        if result is not None:
            record["result"] = result
        if error:
            record["error"] = error
        if artifacts:
            record["artifacts"] = artifacts
        _write(path, record)


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return time.time()


def list_runs(limit: int = 50) -> list[dict]:
    items = []
    for path in RUN_RECORD_DIR.glob("*.json"):
        rec = _read(path)
        if rec:
            items.append({
                "run_id": rec.get("run_id"),
                "kind": rec.get("kind"),
                "status": rec.get("status"),
                "request": rec.get("request", "")[:140],
                "started_at": rec.get("started_at"),
                "updated_at": rec.get("updated_at"),
                "duration_s": rec.get("duration_s"),
            })
    items.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return items[:limit]


def get_run(run_id: str) -> Optional[dict]:
    rec = _read(_record_path(run_id))
    return rec or None
