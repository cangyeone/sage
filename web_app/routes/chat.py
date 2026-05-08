"""聊天和任务管理路由"""
from flask import Blueprint, request, jsonify, send_file, Response, stream_with_context
import json
import os
import shlex
import sys
import threading
import subprocess
import re as _re
import time as _time
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from state import (
    tasks, _session_docs, _geo_agent_jobs, _lit_jobs, _chat_jobs,
    UPLOAD_FOLDER_CHAT, GEO_WORKSPACE_ROOT, _PROJECT_ROOT,
)
from helpers import (
    USER_PROFILE_MD,
    append_user_profile_to_system,
    get_user_profile_context,
    get_llm_config,
    llm_call,
    llm_stream,
    inject_workspace_context,
    get_workspace_config,
    get_kb_instance,
    safe_child_path,
)

bp = Blueprint('chat', __name__)


# ── Persistent chat history ────────────────────────────────────────────────

CHAT_HISTORY_DIR = _PROJECT_ROOT / "seismo_rag" / "chat_history"
CHAT_HISTORY_JSON = CHAT_HISTORY_DIR / "conversations.json"
USER_PROFILE_ARCHIVE_DIR = _PROJECT_ROOT / "seismo_rag" / "user_profiles"
USER_PROFILE_SOURCE_JSON = _PROJECT_ROOT / "seismo_rag" / "user_profile_source.json"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
USER_PROFILE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_conversation_id(value: str) -> str:
    return "".join(ch for ch in str(value or "default") if ch.isalnum() or ch in "_-")[:80] or "default"


def _safe_chat_pdf_upload_path(filename: str | None, session_id: str, upload_id: str) -> tuple[Path, str]:
    raw_name = Path(filename or "").name
    if Path(raw_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")
    safe_stem = secure_filename(Path(raw_name).stem) or "upload"
    safe_name = f"{safe_stem}.pdf"
    tmp_name = f"{_clean_conversation_id(session_id)}_{upload_id}_{safe_name}"
    return safe_child_path(UPLOAD_FOLDER_CHAT, tmp_name), safe_name


def _safe_pdf_upload_path(filename: str | None, session_id: str) -> tuple[Path, str]:
    """Compatibility wrapper used by tests and older upload code paths."""
    upload_id = _uuid.uuid4().hex[:10]
    return _safe_chat_pdf_upload_path(filename, session_id, upload_id)


def _extract_session_pdf_chunks(path: str | Path, upload_id: str, doc_name: str) -> tuple[list, list]:
    """Extract temporary chat-PDF text chunks using the current RAG extractors."""
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_extractors import extract_text, chunk_text

    pages = extract_text(str(path))
    chunks = []
    for page_idx, page_text in pages:
        for c in chunk_text(page_text, chunk_size=600):
            chunks.append({
                "page": page_idx + 1,
                "text": c,
                "doc_name": doc_name,
                "upload_id": upload_id,
            })
    return pages, chunks


def _clean_conversation_title(value: str) -> str:
    title = str(value or "").strip()
    if title in {"新对话", "旧对话", "New chat", "Old chat"}:
        return ""
    return title[:120]


def _conversation_to_markdown(conv: dict) -> str:
    title = str(conv.get("title") or conv.get("id") or "Conversation")
    lines = [
        f"# {title}",
        "",
        f"- id: `{conv.get('id', '')}`",
        f"- created_at: {conv.get('createdAt', '')}",
        f"- updated_at: {conv.get('updatedAt', '')}",
        "",
    ]
    for m in conv.get("messages") or []:
        role = str(m.get("role") or "assistant")
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        lines.extend([f"## {role}", "", content, ""])
    return "\n".join(lines).strip() + "\n"


def _load_persistent_conversations() -> dict:
    if not CHAT_HISTORY_JSON.exists():
        return {"conversations": [], "active_id": "", "projects": [], "active_project_id": ""}
    try:
        data = json.loads(CHAT_HISTORY_JSON.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"conversations": [], "active_id": ""}
        convs = data.get("conversations") or []
        if not isinstance(convs, list):
            convs = []
        projects = data.get("projects") or []
        if not isinstance(projects, list):
            projects = []
        return {
            "conversations": convs,
            "active_id": data.get("active_id") or "",
            "projects": projects,
            "active_project_id": data.get("active_project_id") or "",
        }
    except Exception:
        return {"conversations": [], "active_id": "", "projects": [], "active_project_id": ""}


def _save_persistent_conversations(
    conversations: list,
    active_id: str = "",
    projects: list | None = None,
    active_project_id: str = "",
) -> None:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_convs = []
    for c in conversations[:200]:
        if not isinstance(c, dict):
            continue
        cid = _clean_conversation_id(c.get("id") or "")
        if not cid:
            continue
        safe = {
            "id": cid,
            "title": _clean_conversation_title(c.get("title") or ""),
            "project_id": _clean_conversation_id(c.get("project_id") or "") if c.get("project_id") else "",
            "createdAt": c.get("createdAt"),
            "updatedAt": c.get("updatedAt"),
            "history": (c.get("history") or [])[-80:],
            "messages": (c.get("messages") or [])[-200:],
            "docs": c.get("docs") or [],
        }
        safe_convs.append(safe)

    safe_projects = []
    for p in (projects or []):
        if not isinstance(p, dict):
            continue
        pid = _clean_conversation_id(p.get("id") or "")
        if not pid:
            continue
        safe_projects.append({
            "id": pid,
            "title": str(p.get("title") or "")[:120],
            "preface": str(p.get("preface") or ""),
            "prompt": str(p.get("prompt") or ""),
            "createdAt": p.get("createdAt"),
            "updatedAt": p.get("updatedAt"),
            "docs": p.get("docs") or [],
        })

    payload = {
        "active_id": _clean_conversation_id(active_id),
        "active_project_id": _clean_conversation_id(active_project_id or "") if active_project_id else "",
        "projects": safe_projects,
        "conversations": safe_convs,
    }
    tmp = CHAT_HISTORY_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(CHAT_HISTORY_JSON)

    for conv in safe_convs:
        md_path = CHAT_HISTORY_DIR / f"{conv['id']}.md"
        md_path.write_text(_conversation_to_markdown(conv), encoding="utf-8")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _delete_kb_proj_folder(kb, proj_folder: str) -> int:
    """Remove all KB docs in a logical folder before replacing it."""
    if not kb or not proj_folder:
        return 0
    removed = 0
    for doc in list(kb.list_docs()):
        if (getattr(doc, "proj_folder", "") or "") == proj_folder:
            try:
                if kb.delete_doc(doc.doc_id):
                    removed += 1
            except Exception:
                pass
    return removed


def _save_user_profile(content: str, conversations: list | None = None) -> None:
    """Persist the canonical profile plus a timestamped archive and source manifest."""
    now = datetime.now().isoformat(timespec="seconds")
    if not content.lstrip().startswith("#"):
        content = "# SAGE User Profile\n\n" + content.strip() + "\n"
    if "Updated:" not in content[:500]:
        content = content.rstrip() + f"\n\n---\nUpdated: {now}\n"

    _atomic_write_text(USER_PROFILE_MD, content)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _atomic_write_text(USER_PROFILE_ARCHIVE_DIR / f"user_profile_{stamp}.md", content)
    if conversations is not None:
        source = {
            "updated_at": now,
            "profile_path": str(USER_PROFILE_MD),
            "archive_dir": str(USER_PROFILE_ARCHIVE_DIR),
            "n_conversations": len(conversations),
            "conversation_ids": [c.get("id") for c in conversations if isinstance(c, dict)],
        }
        tmp = USER_PROFILE_SOURCE_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(USER_PROFILE_SOURCE_JSON)


@bp.route('/api/chat/conversations', methods=['GET'])
def chat_conversations_get():
    data = _load_persistent_conversations()
    return jsonify({"ok": True, **data})


@bp.route('/api/chat/conversations', methods=['POST'])
def chat_conversations_save():
    data = request.json or {}
    conversations = data.get("conversations") or []
    if not isinstance(conversations, list):
        return jsonify({"ok": False, "error": "conversations must be a list"}), 400
    try:
        _save_persistent_conversations(
            conversations,
            data.get("active_id") or "",
            data.get("projects") or [],
            data.get("active_project_id") or "",
        )
        return jsonify({"ok": True, "path": str(CHAT_HISTORY_JSON)})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@bp.route('/api/chat/conversations/<conversation_id>', methods=['DELETE'])
def chat_conversation_delete(conversation_id):
    """Delete one persisted conversation and its Markdown mirror."""
    cid = _clean_conversation_id(conversation_id)
    data = _load_persistent_conversations()
    conversations = [c for c in data.get("conversations", []) if _clean_conversation_id(c.get("id")) != cid]
    active_id = data.get("active_id") or ""
    if active_id == cid:
        active_id = conversations[0].get("id", "") if conversations else ""
    try:
        _save_persistent_conversations(conversations, active_id, data.get("projects") or [], data.get("active_project_id") or "")
        md_path = CHAT_HISTORY_DIR / f"{cid}.md"
        if md_path.exists():
            md_path.unlink()
        _session_docs.pop(cid, None)
        _chat_jobs.pop(cid, None)
        return jsonify({"ok": True, "deleted": cid})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@bp.route('/api/chat/project/promote', methods=['POST'])
def chat_project_promote():
    """Persist a project's shared context and conversation summaries into the KB."""
    data = request.json or {}
    project = data.get("project") or {}
    conversations = data.get("conversations") or []
    pid = _clean_conversation_id(project.get("id") or "project")
    title = str(project.get("title") or pid)
    lines = [
        f"# Project: {title}",
        "",
        f"- Project ID: `{pid}`",
        f"- Source: SAGE Chat Project",
        "",
    ]
    if project.get("preface"):
        lines.extend(["## Preface / Research Background", "", str(project.get("preface")), ""])
    if project.get("prompt"):
        lines.extend(["## Shared Prompt", "", str(project.get("prompt")), ""])
    docs = project.get("docs") or []
    if docs:
        lines.extend(["## Project Literature", ""])
        for d in docs:
            lines.append(f"- {d.get('name')} ({d.get('n_chunks', 0)} chunks)")
        lines.append("")
    if conversations:
        lines.extend(["## Conversation Summaries", ""])
        for c in conversations[:50]:
            lines.append(f"### {c.get('title') or c.get('id')}")
            for m in (c.get("messages") or [])[-20:]:
                content = str(m.get("content") or "").strip()
                if content:
                    lines.append(f"- {m.get('role', '')}: {content[:1500]}")
            lines.append("")

    project_dir = CHAT_HISTORY_DIR / "projects"
    project_dir.mkdir(parents=True, exist_ok=True)
    md_path = project_dir / f"{pid}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        kb = get_kb_instance()
        if not kb:
            return jsonify({"ok": False, "error": "Knowledge base unavailable"}), 500
        logs = []
        proj_folder = f"chat_project/{pid}"
        removed_old = _delete_kb_proj_folder(kb, proj_folder)
        meta = kb.add_document(str(md_path), progress_cb=lambda m: logs.append(m), proj_folder=proj_folder, source_type="chat_project")
        return jsonify({
            "ok": True,
            "doc_id": meta.doc_id,
            "doc_name": meta.doc_name,
            "n_chunks": meta.n_chunks,
            "removed_old": removed_old,
            "path": str(md_path),
            "logs": logs[-20:],
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@bp.route('/api/chat/conversation/promote', methods=['POST'])
def chat_conversation_promote():
    """Persist the active chat conversation as one removable KB document."""
    data = request.json or {}
    conv = data.get("conversation") or {}
    cid = _clean_conversation_id(conv.get("id") or "conversation")
    title = _clean_conversation_title(conv.get("title") or "") or cid
    md = _conversation_to_markdown({**conv, "title": title})

    chat_dir = CHAT_HISTORY_DIR / "knowledge_chats"
    chat_dir.mkdir(parents=True, exist_ok=True)
    md_path = chat_dir / f"{cid}.md"
    md_path.write_text(md, encoding="utf-8")

    try:
        kb = get_kb_instance()
        if not kb:
            return jsonify({"ok": False, "error": "Knowledge base unavailable"}), 500
        logs = []
        proj_folder = f"chat/{cid}"
        removed_old = _delete_kb_proj_folder(kb, proj_folder)
        meta = kb.add_document(
            str(md_path),
            progress_cb=lambda m: logs.append(m),
            proj_folder=proj_folder,
            source_type="chat_conversation",
        )
        return jsonify({
            "ok": True,
            "doc_id": meta.doc_id,
            "doc_name": meta.doc_name,
            "n_chunks": meta.n_chunks,
            "removed_old": removed_old,
            "path": str(md_path),
            "logs": logs[-20:],
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500



# ── Task management ──────────────────────────────────────────────────────

_DEVICE_RE = _re.compile(r"^(cpu|mps|cuda(?::\d+)?)$")


def _require_text(data: dict, key: str, label: str) -> str:
    value = data.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{label} required")
    return str(value).strip()


def _optional_text(data: dict, key: str, default: str) -> str:
    value = str(data.get(key, default) or "").strip()
    return value or default


def _validated_device(data: dict) -> str:
    device = _optional_text(data, "device", "cpu")
    if not _DEVICE_RE.match(device):
        raise ValueError("Invalid device; expected cpu, mps, cuda, or cuda:N")
    return device


def _command_display(argv: list[str]) -> str:
    return shlex.join(str(arg) for arg in argv)


def _build_pick_command(data: dict, task_id: str) -> list[str]:
    return [
        sys.executable,
        str(_PROJECT_ROOT / "pnsn" / "picker.py"),
        "-i", _require_text(data, "input_dir", "Input directory"),
        "-o", str(_PROJECT_ROOT / "web_app" / "outputs" / task_id),
        "-m", _optional_text(data, "model", "pnsn/pickers/pnsn.v3.jit"),
        "-d", _validated_device(data),
    ]


def _build_association_command(data: dict, task_id: str) -> list[str]:
    method_scripts = {
        "fastlink": "pnsn/fastlinker.py",
        "real": "pnsn/reallinker.py",
        "gamma": "pnsn/gammalink.py",
    }
    method = _optional_text(data, "method", "fastlink")
    script = method_scripts.get(method)
    if not script:
        raise ValueError("Invalid method; expected fastlink, real, or gamma")
    argv = [
        sys.executable,
        str(_PROJECT_ROOT / script),
        "-i", _require_text(data, "input_file", "Input file"),
        "-o", str(_PROJECT_ROOT / "web_app" / "outputs" / f"{task_id}.txt"),
        "-s", _require_text(data, "station_file", "Station file"),
    ]
    if method == "fastlink":
        argv.extend(["-d", _validated_device(data)])
    return argv


def _build_polarity_command(data: dict, task_id: str) -> list[str]:
    return [
        sys.executable,
        str(_PROJECT_ROOT / "seismic_cli.py"),
        "polarity",
        "-i", _require_text(data, "input_file", "Input file"),
        "-w", _require_text(data, "waveform_dir", "Waveform directory"),
        "-o", str(_PROJECT_ROOT / "web_app" / "outputs" / f"{task_id}_polarity.txt"),
        "--model", _optional_text(data, "model", "pnsn/pickers/polar.onnx"),
        "--min-confidence", str(data.get("min_confidence", 0.5)),
        "--phase", _optional_text(data, "phase", "Pg"),
    ]


def run_task(task_id, command, task_type, cwd=None):
    """Run a seismic processing task in background"""
    try:
        tasks[task_id]['status'] = 'running'
        tasks[task_id]['start_time'] = datetime.now().isoformat()

        argv = list(command) if isinstance(command, (list, tuple)) else shlex.split(str(command))
        result = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=cwd or os.getcwd()
        )

        tasks[task_id]['end_time'] = datetime.now().isoformat()
        tasks[task_id]['returncode'] = result.returncode
        tasks[task_id]['stdout'] = result.stdout[-5000:] if result.stdout else ""  # Last 5000 chars
        tasks[task_id]['stderr'] = result.stderr[-5000:] if result.stderr else ""

        if result.returncode == 0:
            tasks[task_id]['status'] = 'completed'
        else:
            tasks[task_id]['status'] = 'failed'

    except subprocess.TimeoutExpired:
        tasks[task_id]['status'] = 'timeout'
        tasks[task_id]['stderr'] = "Task timed out (1 hour limit)"
    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['stderr'] = str(e)


@bp.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    return jsonify({
        'tasks': {k: {kk: vv for kk, vv in v.items() if kk not in ['stdout', 'stderr']}
                  for k, v in tasks.items()}
    })


@bp.route('/api/task/<task_id>', methods=['GET'])
def get_task(task_id):
    """Get task status and results"""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id].copy()
    task['logs'] = {
        'stdout': task.get('stdout', ''),
        'stderr': task.get('stderr', '')
    }
    task.pop('stdout', None)

    return jsonify(task)


@bp.route('/api/chat_picks/<task_id>', methods=['GET'])
def get_chat_picks(task_id):
    """Poll pick task status; returns parsed picks when done."""
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404
    task = tasks[task_id]
    status = task.get('status', 'running')
    if status == 'running':
        return jsonify({'status': 'running'})
    if status in ('error', 'failed', 'timeout'):
        return jsonify({'status': status, 'error': task.get('stderr', '')})
    # Parse picks file
    picks_file = task.get('picks_file', '')
    picks = []
    if os.path.exists(picks_file):
        with open(picks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                try:
                    picks.append({
                        'phase': parts[0],
                        'time_s': float(parts[1]),
                        'confidence': float(parts[2]),
                        'abs_time': parts[3],
                        'snr': float(parts[4]) if len(parts) > 4 else 0.0,
                        'station': parts[6] if len(parts) > 6 else '',
                        'polarity': parts[7] if len(parts) > 7 else 'N',
                    })
                except (ValueError, IndexError):
                    continue
    return jsonify({'status': 'completed', 'picks': picks, 'n_picks': len(picks)})


@bp.route('/api/pick', methods=['POST'])
def submit_picking():
    """Submit phase picking job"""
    data = request.json or {}

    task_id = f"pick_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    try:
        cmd = _build_pick_command(data, task_id)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'phase_picking',
        'status': 'queued',
        'command': _command_display(cmd),
        'parameters': data,
        'created_at': datetime.now().isoformat()
    }

    # Start task in background
    thread = threading.Thread(target=run_task, args=(task_id, cmd, 'picking'))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id, 'message': 'Task submitted'})


@bp.route('/api/associate', methods=['POST'])
def submit_association():
    """Submit phase association job"""
    data = request.json or {}

    task_id = f"assoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    try:
        cmd = _build_association_command(data, task_id)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'phase_association',
        'status': 'queued',
        'command': _command_display(cmd),
        'parameters': data,
        'created_at': datetime.now().isoformat()
    }

    # Start task in background
    thread = threading.Thread(target=run_task, args=(task_id, cmd, 'association'))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id, 'message': 'Task submitted'})


@bp.route('/api/polarity', methods=['POST'])
def submit_polarity():
    """Submit polarity analysis job"""
    data = request.json or {}

    task_id = f"polar_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    try:
        cmd = _build_polarity_command(data, task_id)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'polarity_analysis',
        'status': 'queued',
        'command': _command_display(cmd),
        'parameters': data,
        'created_at': datetime.now().isoformat()
    }

    # Start task in background
    thread = threading.Thread(target=run_task, args=(task_id, cmd, 'polarity'))
    thread.daemon = True
    thread.start()

    return jsonify({'task_id': task_id, 'message': 'Task submitted'})


@bp.route('/api/output/<filename>', methods=['GET'])
def download_output(filename):
    """Download output file"""
    filepath = os.path.join('web_app/outputs', filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


# ── Literature Loop Agent ───────────────────────────────────────────────────

def _lit_gc():
    cutoff = _time.time() - 1800  # 30-min TTL (reports are larger than code results)
    for k in [k for k, v in _lit_jobs.items() if v.get("ts", 0) < cutoff]:
        _lit_jobs.pop(k, None)


@bp.route('/api/literature_loop', methods=['POST'])
def literature_loop():
    """Start an async literature-loop interpretation job."""
    data          = request.json or {}
    question      = (data.get("question") or "").strip()
    study_area    = (data.get("study_area") or "").strip()
    max_iters     = int(data.get("max_iterations", 3))
    top_k         = int(data.get("rag_top_k", 8))

    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    _lit_gc()
    job_id = "lit_" + _uuid.uuid4().hex[:10]
    _lit_jobs[job_id] = {
        "status":   "running",
        "progress": [],
        "result":   None,
        "error":    None,
        "ts":       _time.time(),
    }

    def _run():
        try:
            import sys as _sys
            import os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from sage_agents import LiteratureLoopAgent

            def _prog(d):
                phase = d.get("phase", "")
                msg   = d.get("message") or d.get("msg", "")
                _lit_jobs[job_id]["progress"].append(
                    {"phase": phase, "message": msg, "ts": _time.time()}
                )

            agent  = LiteratureLoopAgent(llm_cfg=get_llm_config(), top_k=top_k)
            result = agent.run(question, study_area, max_iterations=max_iters,
                               on_progress=_prog)
            _lit_jobs[job_id]["status"] = "done"
            _lit_jobs[job_id]["result"] = agent.result_to_dict(result)
        except Exception as exc:
            _lit_jobs[job_id]["status"] = "error"
            _lit_jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@bp.route('/api/literature_loop/poll/<job_id>', methods=['GET'])
def literature_loop_poll(job_id):
    """Poll for literature-loop job status."""
    job = _lit_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({
        "ok":       True,
        "status":   job["status"],
        "progress": job["progress"],
        "result":   job["result"],
        "error":    job["error"],
    })


# ── Evidence-Driven Geo Agent ─────────────────────────────────────────────────

def _geo_agent_gc():
    """Discard jobs older than 45 minutes."""
    cutoff = _time.time() - 2700
    for k in [k for k, v in _geo_agent_jobs.items() if v.get("ts", 0) < cutoff]:
        _geo_agent_jobs.pop(k, None)


@bp.route('/api/evidence_geo_agent', methods=['POST'])
def evidence_geo_agent():
    """Start an async evidence-driven geoscience interpretation job."""
    data         = request.json or {}
    question     = (data.get("question") or "").strip()
    study_area   = (data.get("study_area") or "").strip()

    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    _geo_agent_gc()
    job_id = "geo_" + _uuid.uuid4().hex[:10]
    session_id = _clean_conversation_id(data.get("session_id") or "default_geo")
    run_output_base = Path(data.get("output_dir") or "outputs/evidence_driven_geo_agent").expanduser()
    effective_output_dir = run_output_base / session_id / job_id
    _geo_agent_jobs[job_id] = {
        "status":   "running",
        "progress": [],
        "result":   None,
        "error":    None,
        "ts":       _time.time(),
        "session_id": session_id,
        "output_dir": str(effective_output_dir),
    }

    def _model_likely_supports_vision(llm_cfg: dict) -> bool:
        provider = str(llm_cfg.get("provider", "")).lower()
        model = str(llm_cfg.get("model", "")).lower()
        vision_markers = (
            "vision", "vl", "qwen-vl", "qwen2-vl", "qwen2.5-vl", "llava",
            "bakllava", "minicpm-v", "gemma3", "gpt-4o", "gpt-4.1", "o4",
            "claude-3", "glm-4v", "internvl", "pixtral", "molmo",
        )
        if any(m in model for m in vision_markers):
            return True
        if provider == "openai" and any(m in model for m in ("gpt-4o", "gpt-4.1", "o4")):
            return True
        return False

    def _prefetch_web_literature(data, cfg, progress_cb):
        """Search scholarly web sources and write a seed literature note into the workspace."""
        if not cfg.allow_web_search:
            return ""
        try:
            from sage_agents.evidence_driven_geo_agent import WebSearchTool
            query_bits = [question]
            if study_area:
                query_bits.append(study_area)
            query_bits.append("geology geophysics seismicity")
            query = " ".join(query_bits)
            progress_cb({"phase": "web_search", "message": f"Searching online literature: {query[:120]}"})
            tool = WebSearchTool(cfg)
            result = tool.literature_search(
                query=query,
                max_results=int(data.get("web_max_results", 8)),
                sources=data.get("web_search_sources") or ["semantic_scholar"],
            )
            if result.get("warning"):
                progress_cb({"phase": "web_search", "message": result["warning"]})
            papers = result.get("papers", [])
            if not papers:
                msg = result.get("warning") or result.get("error") or "No online literature found."
                progress_cb({"phase": "warning", "message": msg})
                return ""

            seed_dir = Path(cfg.output_dir).expanduser() / "literature"
            seed_dir.mkdir(parents=True, exist_ok=True)
            seed = seed_dir / "web_literature_seed.md"
            lines = [
                "# Online Literature Seed",
                "",
                f"Query: {query}",
                "",
                "These records were fetched before the interpretation loop so the agent can treat online literature as traceable evidence.",
                "",
            ]
            for i, p in enumerate(papers, 1):
                authors = ", ".join(p.get("authors", []) or [])
                lines.extend([
                    f"## [{i}] {p.get('title', 'Untitled')}",
                    f"- Year: {p.get('year') or ''}",
                    f"- Authors: {authors}",
                    f"- DOI: {p.get('doi') or ''}",
                    f"- URL: {p.get('url') or ''}",
                    "",
                    p.get("abstract") or "(No abstract returned.)",
                    "",
                ])
            seed.write_text("\n".join(lines), encoding="utf-8")
            progress_cb({"phase": "web_search", "message": f"Saved {len(papers)} online literature records to {seed}"})
            return str(seed)
        except Exception as exc:
            progress_cb({"phase": "warning", "message": f"Online literature prefetch failed: {str(exc)[:160]}"})
            return ""

    def _run():
        try:
            import sys as _sys
            import os as _os
            _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from sage_agents import EvidenceDrivenGeoAgent, AgentConfig
            ws_cfg = get_workspace_config()
            authorized_roots = []
            if ws_cfg.get("enabled"):
                authorized_roots.extend(ws_cfg.get("paths") or [])
            authorized_roots.extend(data.get("authorized_roots") or [])
            authorized_roots = [str(p).strip() for p in authorized_roots if str(p).strip()]

            # Build config from request
            cfg = AgentConfig(
                workspace_root=data.get("workspace_root") or ".",
                literature_root=data.get("literature_root") or "",
                output_dir=str(effective_output_dir),
                authorized_roots=authorized_roots,
                allow_python=bool(data.get("allow_python", True)),
                allow_shell=bool(data.get("allow_shell", False)),
                allow_web_search=bool(data.get("allow_web_search", False)),
                use_multimodal=bool(data.get("use_multimodal", False)),
                use_rag=bool(data.get("use_rag", True)),
                use_local_files=bool(data.get("use_local_files", True)),
                produce_latex=bool(data.get("produce_latex", True)),
                use_code_engine=bool(data.get("use_code_engine", True)),
                web_search_sources=data.get("web_search_sources") or ["openalex", "semantic_scholar"],
                max_iterations=int(data.get("max_iterations", 3)),
                max_tool_calls_per_iter=int(data.get("max_tool_calls_per_iter", 8)),
                rag_top_k=int(data.get("rag_top_k", 8)),
                score_threshold=float(data.get("score_threshold", 0.35)),
                code_timeout_s=int(data.get("code_timeout_s", 60)),
            )

            def _prog(d):
                phase = d.get("phase", "")
                msg   = d.get("message") or d.get("msg", "")
                _geo_agent_jobs[job_id]["progress"].append(
                    {"phase": phase, "message": msg, "ts": _time.time()}
                )

            seed_path = _prefetch_web_literature(data, cfg, _prog)
            if seed_path and not cfg.literature_root:
                cfg.literature_root = str(Path(seed_path).parent)

            llm_cfg = get_llm_config()
            if cfg.use_multimodal and not _model_likely_supports_vision(llm_cfg):
                _prog({
                    "phase": "warning",
                    "message": (
                        "Multimodal image/table parsing was requested, but the selected model "
                        f"({llm_cfg.get('provider','')}/{llm_cfg.get('model','')}) does not look vision-capable. "
                        "The agent will still parse text/CSV tables and will warn if image analysis fails."
                    ),
                })

            profile = get_user_profile_context(max_chars=2500)
            question_for_agent = question
            project_context = (data.get("project_context") or "").strip()
            geo_project_context = (data.get("geo_project_context") or "").strip()
            project_ids = data.get("project_ids") or []
            geo_project_ids = data.get("geo_project_ids") or []
            if isinstance(project_ids, list) and project_ids:
                question_for_agent += "\n\n===== Referenced Chat Project IDs =====\n" + ", ".join(str(x) for x in project_ids[:12])
            if project_context and project_context not in question_for_agent:
                question_for_agent += "\n\n===== Project shared context =====\n" + project_context[:12000]
            if isinstance(geo_project_ids, list) and geo_project_ids:
                question_for_agent += "\n\n===== Referenced upstream Geo Project IDs =====\n" + ", ".join(str(x) for x in geo_project_ids[:12])
            if geo_project_context and geo_project_context not in question_for_agent:
                question_for_agent += (
                    "\n\n===== Upstream interpretation projects: evidence chains to inherit and re-audit =====\n"
                    "Use these as prior research assets, not as unquestioned truth. For each inherited claim, "
                    "seek evidence-of-evidence, check reliability/relevance, preserve upstream_evidence links, "
                    "and list missing verification data.\n"
                    + geo_project_context[:18000]
                )
            question_for_agent += (
                "\n\n===== Required reasoning protocol =====\n"
                "During the investigation, iteratively test hypotheses by collecting evidence and evidence-of-evidence. "
                "For figures and tables from papers or uploaded images, extract quantitative values when possible. "
                "For every important evidence record, estimate relevance and reliability, identify upstream evidence, "
                "state verification_status, and describe verification_needed. Explicitly report missing information and "
                "rank which evidence is most relevant and most reliable only when the ranking is grounded in the "
                "collected evidence table. Every evidence record must include a source_excerpt that can be traced "
                "to a tool output, file, RAG chunk, web result, figure/table extraction, or upstream project record. "
                "Do not invent evidence IDs, citations, star ratings, scores, locations, numbers, methods, or source names; "
                "if support is missing, say evidence is insufficient and list the missing source.\n"
            )
            if profile:
                question_for_agent += (
                    "\n\n===== Long-term user profile (soft context; do not mention unless useful) =====\n"
                    + profile
                )

            agent  = EvidenceDrivenGeoAgent(config=cfg, llm_cfg=llm_cfg)
            result = agent.run(question_for_agent, study_area, on_progress=_prog)
            _geo_agent_jobs[job_id]["status"] = "done"
            _geo_agent_jobs[job_id]["result"] = result
            if isinstance(result, dict):
                result.setdefault("_run_output_dir", str(effective_output_dir))
                result.setdefault("_session_id", session_id)
        except Exception as exc:
            _geo_agent_jobs[job_id]["status"] = "error"
            _geo_agent_jobs[job_id]["error"]  = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@bp.route('/api/evidence_geo_agent/poll/<job_id>', methods=['GET'])
def evidence_geo_agent_poll(job_id):
    """Poll for evidence-geo-agent job status and result."""
    job = _geo_agent_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    return jsonify({
        "ok":       True,
        "status":   job["status"],       # "running" | "done" | "error"
        "progress": job["progress"],     # list of {phase, message, ts}
        "result":   job["result"],       # None while running, full dict when done
        "error":    job["error"],
    })


# ── EvidenceGeoAgent — file upload for workspace ───────────────────────────

_GEO_ALLOWED_EXTS = {
    ".pdf", ".png", ".jpg", ".jpeg",
    ".csv", ".txt", ".md", ".json",
    ".yaml", ".yml", ".bib", ".dat",
    ".sac", ".mseed", ".xml",
}


@bp.route('/api/evidence_geo_agent/upload', methods=['POST'])
def evidence_geo_agent_upload():
    """Upload a research file into the agent's workspace."""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file provided"}), 400

    f          = request.files['file']
    session_id = (request.form.get('session_id') or 'default').replace('/', '_').replace('..', '_')
    orig_name  = Path(f.filename).name if f.filename else 'upload'
    ext        = Path(orig_name).suffix.lower()

    if ext not in _GEO_ALLOWED_EXTS:
        return jsonify({"ok": False, "error": f"File type '{ext}' not allowed"}), 400

    # Create session workspace
    ws_dir = GEO_WORKSPACE_ROOT / session_id
    ws_dir.mkdir(parents=True, exist_ok=True)

    # Determine sub-folder by type
    if ext == '.pdf':
        sub = 'literature';  ftype = 'pdf'
    elif ext in {'.png', '.jpg', '.jpeg'}:
        sub = 'figures';     ftype = 'image'
    elif ext == '.csv':
        sub = 'data';        ftype = 'data'
    else:
        sub = 'misc';        ftype = 'text'

    sub_dir = ws_dir / sub
    sub_dir.mkdir(exist_ok=True)

    dest = sub_dir / orig_name
    f.save(str(dest))

    return jsonify({
        "ok":        True,
        "path":      str(dest),
        "file_type": ftype,
        "session_workspace": str(ws_dir),
    })


# ── EvidenceGeoAgent — inline web / scholar search ────────────────────────

@bp.route('/api/evidence_geo_agent/web_search', methods=['POST'])
def evidence_geo_agent_web_search():
    """Lightweight inline web search used by the frontend search panel."""
    data         = request.json or {}
    query        = (data.get('query') or '').strip()
    search_type  = data.get('search_type', 'scholar')
    max_results  = int(data.get('max_results', 10))

    if not query:
        return jsonify({"ok": False, "error": "query is required"}), 400

    import sys as _sys
    import os as _os
    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)

    try:
        from sage_agents.evidence_driven_geo_agent import AgentConfig, WebSearchTool
        cfg  = AgentConfig(allow_web_search=True)
        tool = WebSearchTool(cfg)

        if search_type in ('literature', 'multi'):
            result = tool.literature_search(query, max_results=max_results, sources=data.get('sources') or ['semantic_scholar'])
        elif search_type == 'openalex':
            result = tool.openalex_search(query, max_results=max_results)
        elif search_type == 'arxiv':
            result = tool.arxiv_search(query, max_results=max_results)
        elif search_type in ('scholar', 'semantic_scholar'):
            result = tool.scholar_search(query, max_results=max_results)
        else:
            result = tool.web_search(query, max_results=max_results)

        if 'error' in result:
            return jsonify({"ok": False, "error": result['error']})

        # Normalise to a flat list
        items = result.get('results', result.get('papers', []))
        return jsonify({"ok": True, "results": items[:max_results]})

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)})


# ── EvidenceGeoAgent — serve a generated figure by path ───────────────────

@bp.route('/api/evidence_geo_agent/figure', methods=['GET'])
def evidence_geo_agent_figure():
    """Serve a generated figure PNG from the agent's output directory."""
    import os as _os
    fig_path = request.args.get('path', '').strip()
    if not fig_path:
        return jsonify({"error": "path required"}), 400

    p = Path(fig_path)
    # Security: only serve files that exist and have image extensions
    if p.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.svg'}:
        return jsonify({"error": "Unsupported file type"}), 400
    if not p.exists():
        return jsonify({"error": "File not found"}), 404

    # Resolve and check it stays within the project root or GEO_WORKSPACE_ROOT
    proj_root = Path(__file__).parent.parent.resolve()
    try:
        p.resolve().relative_to(proj_root)
    except ValueError:
        try:
            p.resolve().relative_to(GEO_WORKSPACE_ROOT.resolve())
        except ValueError:
            return jsonify({"error": "Access denied"}), 403

    mime = 'image/svg+xml' if p.suffix.lower() == '.svg' else 'image/png'
    return send_file(str(p.resolve()), mimetype=mime)


# ── 聊天界面临时文档上传 ────────────────────────────────────────────────────

@bp.route('/api/chat/upload', methods=['POST'])
def chat_upload_pdf():
    """Upload a PDF for the current chat session (temporary RAG, not persisted)."""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files['file']
    session_id = _clean_conversation_id(request.form.get('session_id', 'default'))
    project_id = _clean_conversation_id(request.form.get('project_id', ''))

    upload_id = _uuid.uuid4().hex[:10]
    try:
        tmp_path, _safe_name = _safe_chat_pdf_upload_path(f.filename, session_id, upload_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    f.save(str(tmp_path))

    try:
        # Extract text (no BGE-M3, just raw text for session context)
        pages, chunks = _extract_session_pdf_chunks(tmp_path, upload_id, f.filename)
        for chunk in chunks:
            chunk.setdefault("doc_name", f.filename)
            chunk.setdefault("upload_id", upload_id)

        if session_id not in _session_docs:
            _session_docs[session_id] = {"chunks": [], "doc_names": [], "files": {}}

        _session_docs[session_id]["chunks"].extend(chunks)
        _session_docs[session_id]["doc_names"].append(f.filename)
        _session_docs[session_id].setdefault("files", {})[upload_id] = {
            "name": f.filename,
            "path": str(tmp_path),
            "n_pages": len(pages),
            "n_chunks": len(chunks),
            "permanent": False,
        }

        if project_id:
            project_key = f"project_{project_id}"
            if project_key not in _session_docs:
                _session_docs[project_key] = {"chunks": [], "doc_names": [], "files": {}}
            _session_docs[project_key]["chunks"].extend(chunks)
            if f.filename not in _session_docs[project_key]["doc_names"]:
                _session_docs[project_key]["doc_names"].append(f.filename)
            _session_docs[project_key].setdefault("files", {})[upload_id] = {
                "name": f.filename,
                "path": str(tmp_path),
                "n_pages": len(pages),
                "n_chunks": len(chunks),
                "permanent": False,
            }

        return jsonify({
            "ok": True,
            "doc_name": f.filename,
            "upload_id": upload_id,
            "n_pages":  len(pages),
            "n_chunks": len(chunks),
            "session_id": session_id,
            "project_id": project_id,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route('/api/chat/promote_doc', methods=['POST'])
def chat_promote_doc():
    """Persist a temporary chat PDF into the permanent knowledge base."""
    data = request.get_json(silent=True) or {}
    session_id = _clean_conversation_id(data.get("session_id", "default"))
    upload_id = data.get("upload_id", "")
    info = _session_docs.get(session_id, {}).get("files", {}).get(upload_id)
    if not info:
        return jsonify({"ok": False, "error": "Temporary document not found"}), 404

    path = Path(info.get("path", ""))
    if not path.exists():
        return jsonify({"ok": False, "error": "Temporary file has expired"}), 404

    try:
        kb = get_kb_instance()
        if not kb:
            return jsonify({"ok": False, "error": "Knowledge base unavailable"}), 500
        logs = []
        meta = kb.add_pdf(str(path), progress_cb=lambda m: logs.append(m), source_type="upload")
        info["permanent"] = True
        return jsonify({
            "ok": True,
            "doc_id": meta.doc_id,
            "doc_name": meta.doc_name,
            "n_chunks": meta.n_chunks,
            "logs": logs[-20:],
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def _remove_doc_from_session(session_key: str, upload_id: str = "", doc_name: str = "") -> bool:
    session = _session_docs.get(session_key)
    if not session:
        return False

    files = session.setdefault("files", {})
    removed_infos = []
    if upload_id and upload_id in files:
        removed_infos.append(files.pop(upload_id))
    elif doc_name:
        for fid, info in list(files.items()):
            if info.get("name") == doc_name:
                removed_infos.append(files.pop(fid))

    removed_upload_ids = {upload_id} if upload_id else set()
    removed_names = {doc_name} if doc_name else set()
    for info in removed_infos:
        if info.get("name"):
            removed_names.add(info.get("name"))
        try:
            if not info.get("permanent"):
                Path(info.get("path", "")).unlink(missing_ok=True)
        except Exception:
            pass

    def _should_remove_chunk(chunk: dict) -> bool:
        return (
            (upload_id and chunk.get("upload_id") in removed_upload_ids)
            or (doc_name and chunk.get("doc_name") in removed_names)
        )

    before = len(session.get("chunks") or [])
    session["chunks"] = [
        c for c in (session.get("chunks") or [])
        if not _should_remove_chunk(c)
    ]

    remaining_names = {
        info.get("name")
        for info in files.values()
        if info.get("name")
    } | {
        c.get("doc_name")
        for c in session.get("chunks", [])
        if c.get("doc_name")
    }
    if remaining_names:
        session["doc_names"] = [
            n for n in (session.get("doc_names") or [])
            if n in remaining_names
        ]
    else:
        session["doc_names"] = []

    if not session.get("chunks") and not session.get("files") and not session.get("doc_names"):
        _session_docs.pop(session_key, None)

    return bool(removed_infos) or len(session.get("chunks") or []) != before


@bp.route('/api/chat/remove_session_doc', methods=['POST'])
def chat_remove_session_doc():
    data = request.get_json(silent=True) or {}
    sid = _clean_conversation_id(data.get('session_id', 'default'))
    project_id = _clean_conversation_id(data.get('project_id', ''))
    upload_id = str(data.get("upload_id") or "").strip()
    doc_name = str(data.get("doc_name") or "").strip()
    if not upload_id and not doc_name:
        return jsonify({"ok": False, "error": "Missing document identifier"}), 400

    removed = _remove_doc_from_session(sid, upload_id=upload_id, doc_name=doc_name)
    if project_id:
        removed = _remove_doc_from_session(f"project_{project_id}", upload_id=upload_id, doc_name=doc_name) or removed
    session = _session_docs.get(sid, {})
    return jsonify({
        "ok": True,
        "removed": removed,
        "doc_names": session.get("doc_names", []),
    })


@bp.route('/api/chat/clear_session', methods=['POST'])
def chat_clear_session():
    data = request.get_json(silent=True) or {}
    sid = _clean_conversation_id(data.get('session_id', 'default'))
    for info in _session_docs.get(sid, {}).get("files", {}).values():
        try:
            Path(info.get("path", "")).unlink(missing_ok=True)
        except Exception:
            pass
    _session_docs.pop(sid, None)
    return jsonify({"ok": True})


# ── RAG 增强对话 ──────────────────────────────────────────────────────────────

@bp.route('/api/chat/rag', methods=['POST'])
def chat_rag():
    """RAG-aware chat endpoint."""
    data       = request.json or {}
    user_msg   = data.get("message", "").strip()
    session_id = _clean_conversation_id(data.get("session_id", "default"))
    mode       = data.get("mode", "rag")   # "rag" | "paper_read"

    if not user_msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    llm_cfg = get_llm_config()
    if not llm_cfg.get("api_base"):
        return jsonify({
            "ok": True,
            "response": (
                "当前没有可用的 LLM 后端。\n"
                "请在 **LLM 设置** 页面配置后端（Ollama / 在线 API）。"
            ),
            "sources": [],
        })

    # ── 检索上下文 ──────────────────────────────────────────────────────────
    context_parts = []
    sources = []

    # 0. 工作目录文件系统上下文（用户授权后注入）
    workspace_path = data.get("workspace", "")
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            context_parts.append("===== 本地文件系统 =====\n" + ws_ctx)

    # 1. 会话文档（临时上传）
    session = _session_docs.get(session_id, {})
    if session.get("chunks"):
        # 简单 TF-IDF 式关键词匹配（无需 GPU）
        query_words = set(user_msg.lower().split())
        scored = []
        for c in session["chunks"]:
            words = set(c["text"].lower().split())
            score = len(query_words & words) / (len(query_words) + 1)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:4]
        for score, c in top:
            if score > 0 or mode == "paper_read":
                context_parts.append(
                    f"[上传文档 第{c['page']}页]\n{c['text']}"
                )
        if session.get("doc_names"):
            sources.extend(session["doc_names"])

    # 2. 持久知识库（BGE-M3 向量检索 / TF-IDF 回退）
    try:
        kb = get_kb_instance()
        if kb and not kb.is_empty:
            kb_hits = kb.retrieve(user_msg, top_k=5, score_threshold=0.45)
            if kb_hits:
                lines = ["The following passages were retrieved from the knowledge base. "
                         "Use them only if they are directly relevant to the question:\n"]
                total = 0
                for chunk, score in kb_hits:
                    entry = (
                        f"[Source: {chunk.doc_name}, page {chunk.page + 1}, "
                        f"relevance {score:.2f}]\n{chunk.text}\n"
                    )
                    if total + len(entry) > 2500:
                        break
                    lines.append(entry)
                    total += len(entry)
                    if chunk.doc_name not in sources:
                        sources.append(chunk.doc_name)
                context_parts.append("\n".join(lines))
    except Exception:
        pass

    # 3. seismo_skill 技能文档（按用户消息检索最相关技能，注入代码示例）
    try:
        from helpers import get_skill_loader
        sl = get_skill_loader()
        if sl is not None:
            skill_ctx, skill_rag_ctx = sl.build_skill_context_with_rag(
                user_msg, max_skill_chars=3000, max_rag_chars=2500, top_k=2
            )
            if skill_ctx:
                context_parts.append("===== 可用技能与函数示例 =====\n" + skill_ctx)
            if skill_rag_ctx:
                context_parts.append("===== 技能绑定知识库 =====\n" + skill_rag_ctx)
    except Exception:
        pass

    # ── 构建提示 ─────────────────────────────────────────────────────────────
    if mode == "paper_read":
        system = (
            "你是一位专业的地震学文献解读专家。\n"
            "请基于以下论文内容，用清晰的中文解读、总结或回答用户的问题。\n"
            "回答时请：\n"
            "1. 点明核心方法/创新点\n"
            "2. 解释关键公式或算法（必要时给出 Python 代码示例）\n"
            "3. 说明实验结果与结论\n"
            "4. 指出局限性或未来工作（如有）\n"
        )
    else:
        if context_parts:
            system = (
                "You are SAGE, an expert seismology assistant with deep knowledge of "
                "seismology and data processing.\n"
                "Relevant passages from the knowledge base are provided below. "
                "Use them to answer the question. "
                "If a passage is not directly relevant, rely on your own knowledge instead — "
                "do NOT cite or mention passages that are unrelated to the question.\n"
            )
        else:
            system = (
                "You are SAGE, an expert seismology assistant with deep knowledge of "
                "seismology and data processing.\n"
                "Answer the user's question using your own knowledge. "
                "Be concise and accurate.\n"
            )

    if context_parts:
        system += "\n\n===== Reference passages =====\n" + "\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]

    # 加入历史（前端传入）
    history = data.get("history", [])
    if history:
        # 在 system 和最后 user 消息之间插入历史
        messages = [messages[0]] + history[-6:] + [messages[-1]]

    try:
        answer = llm_call(messages, llm_cfg, max_tokens=2000)
        return jsonify({
            "ok": True,
            "response": answer,
            "sources": list(set(sources)),
        })
    except Exception as e:
        return jsonify({
            "ok": True,
            "response": (
                f"LLM 调用失败：{e}\n\n"
                "请检查 LLM 设置页面中的后端配置是否正确。"
            ),
            "sources": [],
        })


# ── 流式 RAG 对话 ─────────────────────────────────────────────────────────────

def _chat_web_search_context(data: dict, query: str):
    """Optional chat-side web/literature search context."""
    if not data.get("enable_web_search"):
        return "", []
    try:
        from sage_agents.evidence_driven_geo_agent import AgentConfig, WebSearchTool
        sources = data.get("web_search_sources") or ["openalex", "semantic_scholar"]
        tool = WebSearchTool(AgentConfig(allow_web_search=True, web_search_sources=sources))
        result = tool.literature_search(query=query, max_results=int(data.get("web_max_results", 6)), sources=sources)
        papers = result.get("papers", []) or []
        if not papers:
            return "", []
        lines = [
            "Use the following online literature/search records only when directly relevant. "
            "Cite records by source/title/URL; do not invent claims beyond the abstracts/snippets."
        ]
        refs = []
        for i, p in enumerate(papers[:8], 1):
            title = p.get("title") or "Untitled"
            authors = ", ".join(p.get("authors", []) or [])
            year = p.get("year") or ""
            url = p.get("url") or p.get("pdf_url") or ""
            doi = p.get("doi") or ""
            abstract = (p.get("abstract") or p.get("snippet") or "")[:900]
            src = p.get("source") or "web"
            lines.append(
                f"[Web {i} | {src}] {title}\n"
                f"Authors: {authors}\nYear: {year}\nDOI: {doi}\nURL: {url}\n"
                f"Abstract/snippet: {abstract}\n"
            )
            refs.append(f"{src}: {title}" + (f" ({url})" if url else ""))
        return "\n".join(lines), refs
    except Exception as exc:
        return f"Web search failed: {exc}", []

def _build_rag_messages(data: dict):
    """
    Shared helper: build (messages, sources, llm_cfg) for both /api/chat/rag
    and its streaming twin.  Returns None for llm_cfg if backend not configured.
    """
    import json as _json
    user_msg   = data.get("message", "").strip()
    session_id = _clean_conversation_id(data.get("session_id", "default"))
    mode       = data.get("mode", "rag")

    llm_cfg = get_llm_config()

    context_parts = []
    sources = []

    workspace_path = data.get("workspace", "")
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            context_parts.append("===== 本地文件系统 =====\n" + ws_ctx)

    session = _session_docs.get(session_id, {})
    project_id = _clean_conversation_id(data.get("project_id", ""))
    project_session = _session_docs.get(f"project_{project_id}", {}) if project_id else {}
    merged_chunks = list(project_session.get("chunks") or []) + list(session.get("chunks") or [])
    merged_doc_names = list(dict.fromkeys((project_session.get("doc_names") or []) + (session.get("doc_names") or [])))
    if merged_chunks:
        query_words = set(user_msg.lower().split())
        scored = []
        for c in merged_chunks:
            words = set(c["text"].lower().split())
            score = len(query_words & words) / (len(query_words) + 1)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, c in scored[:4]:
            if score > 0 or mode == "paper_read":
                context_parts.append(f"[上传文档 第{c['page']}页]\n{c['text']}")
        if merged_doc_names:
            sources.extend(merged_doc_names)

    project_context = (data.get("project_context") or "").strip()
    if project_context:
        context_parts.append("===== 项目共享上下文 =====\n" + project_context[:4000])

    web_ctx, web_sources = _chat_web_search_context(data, user_msg)
    if web_ctx:
        context_parts.append("===== Web literature/search context =====\n" + web_ctx)
    sources.extend(web_sources)

    try:
        kb = get_kb_instance()
        if kb and not kb.is_empty:
            kb_hits = kb.retrieve(user_msg, top_k=5, score_threshold=0.45)
            if kb_hits:
                lines = ["The following passages were retrieved from the knowledge base. "
                         "Use them only if they are directly relevant to the question:\n"]
                total = 0
                for chunk, score in kb_hits:
                    entry = (f"[Source: {chunk.doc_name}, page {chunk.page + 1}, "
                             f"relevance {score:.2f}]\n{chunk.text}\n")
                    if total + len(entry) > 2500:
                        break
                    lines.append(entry)
                    total += len(entry)
                    if chunk.doc_name not in sources:
                        sources.append(chunk.doc_name)
                context_parts.append("\n".join(lines))
    except Exception:
        pass

    try:
        from helpers import get_skill_loader
        sl = get_skill_loader()
        if sl is not None:
            skill_ctx, skill_rag_ctx = sl.build_skill_context_with_rag(
                user_msg, max_skill_chars=3000, max_rag_chars=2500, top_k=2
            )
            if skill_ctx:
                context_parts.append("===== 可用技能与函数示例 =====\n" + skill_ctx)
            if skill_rag_ctx:
                context_parts.append("===== 技能绑定知识库 =====\n" + skill_rag_ctx)
    except Exception:
        pass

    if mode == "paper_read":
        system = (
            "你是一位专业的地震学文献解读专家。\n"
            "请基于以下论文内容，用清晰的中文解读、总结或回答用户的问题。\n"
            "回答时请：\n"
            "1. 点明核心方法/创新点\n"
            "2. 解释关键公式或算法（必要时给出 Python 代码示例）\n"
            "3. 说明实验结果与结论\n"
            "4. 指出局限性或未来工作（如有）\n"
        )
    else:
        if context_parts:
            system = (
                "You are SAGE, an expert seismology assistant with deep knowledge of "
                "seismology and data processing.\n"
                "Relevant passages from the knowledge base are provided below. "
                "Use them to answer the question. "
                "If a passage is not directly relevant, rely on your own knowledge instead — "
                "do NOT cite or mention passages that are unrelated to the question.\n"
            )
        else:
            system = (
                "You are SAGE, an expert seismology assistant with deep knowledge of "
                "seismology and data processing.\n"
                "Answer the user's question using your own knowledge. "
                "Be concise and accurate.\n"
            )

    system = append_user_profile_to_system(system)

    # Think-mode must be model-neutral. OpenAI-compatible reasoning streams are
    # normalized in helpers.llm_stream; non-reasoning local/online models are
    # instructed to emit the same tags so the UI can show/collapse them.
    enable_think = bool(data.get("enable_think", False))
    
    if enable_think:
        system += (
            "\n\n如果需要推理，请严格把中间思考放在 <think>...</think> 标签内；"
            "标签外只输出给用户看的最终回答。"
        )

    if context_parts:
        system += "\n\n===== Reference passages =====\n" + "\n\n".join(context_parts)

    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user_msg}]

    history = data.get("history", [])
    if history:
        messages = [messages[0]] + history[-6:] + [messages[-1]]

    return messages, list(set(sources)), llm_cfg


@bp.route('/api/chat/rag/stream', methods=['POST'])
def chat_rag_stream():
    """
    Streaming version of /api/chat/rag.
    Returns text/event-stream SSE with events:
      data: {"type":"sources","sources":[...]}      — sent first
      data: {"type":"chunk","text":"..."}            — one per token
      data: {"type":"done"}
      data: {"type":"error","message":"..."}
    """
    import json as _json

    data = request.json or {}
    if not data.get("message", "").strip():
        return jsonify({"ok": False, "error": "Empty message"}), 400

    images = data.get("images") or []
    messages, sources, llm_cfg = _build_rag_messages(data)

    if not llm_cfg.get("api_base"):
        # Fall back to single-shot response wrapped as SSE
        def _no_backend():
            msg = ("当前没有可用的 LLM 后端。\n"
                   "请在 **LLM 设置** 页面配置后端（Ollama / 在线 API）。")
            yield f"data: {_json.dumps({'type':'sources','sources':[]})}\n\n"
            yield f"data: {_json.dumps({'type':'chunk','text':msg})}\n\n"
            yield f"data: {_json.dumps({'type':'done'})}\n\n"
        return Response(stream_with_context(_no_backend()),
                        mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    def generate():
        # Send sources immediately so the UI can show references
        yield f"data: {_json.dumps({'type':'sources','sources':sources})}\n\n"
        try:
            from helpers import llm_stream
            for chunk in llm_stream(messages, llm_cfg, max_tokens=2000,
                                    images=images if images else None):
                yield f"data: {_json.dumps({'type':'chunk','text':chunk})}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        yield f"data: {_json.dumps({'type':'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@bp.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """
    Streaming plain chat (no RAG).  Same SSE format as /api/chat/rag/stream.
    """
    import json as _json

    data    = request.json or {}
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    images = data.get("images") or []
    llm_cfg = get_llm_config()

    enable_think = bool(data.get("enable_think", False))
    system = (
        "You are SAGE, an expert seismology assistant with deep knowledge of "
        "seismology, geophysics and data processing.\n"
        "Answer the user's question using your own knowledge. Be concise and accurate.\n"
    )
    system = append_user_profile_to_system(system)
    if enable_think:
        system += (
            "\n\n如果需要推理，请严格把中间思考放在 <think>...</think> 标签内；"
            "标签外只输出给用户看的最终回答。"
        )

    workspace_path = data.get("workspace", "")
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            system += "\n\n===== 本地文件系统 =====\n" + ws_ctx

    project_context = (data.get("project_context") or "").strip()
    if project_context:
        system += "\n\n===== 项目共享上下文 =====\n" + project_context[:4000]

    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user_msg}]
    history = data.get("history", [])
    if history:
        messages = [messages[0]] + history[-6:] + [messages[-1]]

    def generate():
        yield f"data: {_json.dumps({'type':'sources','sources':[]})}\n\n"
        if not llm_cfg.get("api_base"):
            msg = "当前没有可用的 LLM 后端，请在 LLM 设置页面配置后端。"
            yield f"data: {_json.dumps({'type':'chunk','text':msg})}\n\n"
            yield f"data: {_json.dumps({'type':'done'})}\n\n"
            return
        try:
            from helpers import llm_stream
            for chunk in llm_stream(messages, llm_cfg, max_tokens=2000,
                                    images=images if images else None):
                yield f"data: {_json.dumps({'type':'chunk','text':chunk})}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'type':'error','message':str(exc)})}\n\n"
        yield f"data: {_json.dumps({'type':'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


# ── 后台异步聊天 Job（切换页面后仍能继续执行并取回结果）────────────────────────

_CHAT_JOB_TTL = 1800   # 30 分钟内可取回结果


def _chat_job_gc():
    cutoff = _time.time() - _CHAT_JOB_TTL
    for k in [k for k, v in list(_chat_jobs.items()) if v.get('ts', 0) < cutoff]:
        _chat_jobs.pop(k, None)


def _build_plain_messages(data: dict):
    """Build messages for plain (no-RAG) chat — mirrors chat_stream() logic."""
    user_msg     = data.get('message', '').strip()
    enable_think = bool(data.get('enable_think', False))
    llm_cfg      = get_llm_config()
    system = (
        'You are SAGE, an expert seismology assistant with deep knowledge of '
        'seismology, geophysics and data processing.\n'
        'Answer the user\'s question using your own knowledge. Be concise and accurate.\n'
    )
    system = append_user_profile_to_system(system)
    if enable_think:
        system += (
            '\n\n如果需要推理，请严格把中间思考放在 <think>...</think> 标签内；'
            '标签外只输出给用户看的最终回答。'
        )

    workspace_path = data.get('workspace', '')
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            system += '\n\n===== 本地文件系统 =====\n' + ws_ctx

    web_ctx, web_sources = _chat_web_search_context(data, user_msg)
    if web_ctx:
        system += '\n\n===== Web literature/search context =====\n' + web_ctx

    messages = [{'role': 'system', 'content': system},
                {'role': 'user',   'content': user_msg}]
    history = data.get('history', [])
    if history:
        messages = [messages[0]] + history[-6:] + [messages[-1]]
    return messages, web_sources, llm_cfg


@bp.route('/api/chat/submit', methods=['POST'])
def chat_submit():
    """
    Start a non-streaming background chat job.  Returns {job_id} immediately.
    The LLM call runs in a daemon thread — survives page navigation.

    Body fields (same as /api/chat/rag/stream) plus:
      type: 'rag' (default) | 'plain'
    """
    data = request.json or {}
    if not data.get('message', '').strip():
        return jsonify({'ok': False, 'error': 'Empty message'}), 400

    _chat_job_gc()
    job_id = 'chat_' + _uuid.uuid4().hex[:10]
    _chat_jobs[job_id] = {
        'status': 'running', 'answer': '', 'sources': [],
        'error': '', 'cancelled': False, 'ts': _time.time(),
    }

    chat_type = data.get('type', 'rag')
    images    = data.get('images') or []   # list of base64 strings for VL models

    def _run():
        try:
            if chat_type == 'plain':
                messages, sources, llm_cfg = _build_plain_messages(data)
            else:
                messages, sources, llm_cfg = _build_rag_messages(data)

            if not llm_cfg.get('api_base'):
                _chat_jobs[job_id].update(
                    answer='当前没有可用的 LLM 后端。\n请在 **LLM 设置** 页面配置后端（Ollama / 在线 API）。',
                    status='done',
                )
                return

            answer_parts = []
            for chunk in llm_stream(messages, llm_cfg, max_tokens=2000,
                                    images=images if images else None):
                if _chat_jobs[job_id].get('cancelled'):
                    _chat_jobs[job_id].update(status='cancelled', error='已停止当前回复')
                    return
                answer_parts.append(chunk)
                _chat_jobs[job_id].update(
                    answer=''.join(answer_parts),
                    sources=sources,
                    status='running',
                    ts=_time.time(),
                )
            if _chat_jobs[job_id].get('cancelled'):
                _chat_jobs[job_id].update(status='cancelled', error='已停止当前回复')
            else:
                _chat_jobs[job_id].update(answer=''.join(answer_parts), sources=sources, status='done')
        except Exception as exc:
            if _chat_jobs[job_id].get('cancelled'):
                _chat_jobs[job_id].update(status='cancelled', error='已停止当前回复')
            else:
                _chat_jobs[job_id].update(status='error', error=str(exc))

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@bp.route('/api/chat/job/<job_id>', methods=['GET'])
def chat_job_poll(job_id):
    """Poll a background chat job for its result."""
    job = _chat_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found or expired'}), 404
    return jsonify({
        'ok':      True,
        'status':  job['status'],   # 'running' | 'done' | 'error'
        'answer':  job['answer'],
        'sources': job['sources'],
        'error':   job['error'],
    })


@bp.route('/api/chat/job/<job_id>/cancel', methods=['POST'])
def chat_job_cancel(job_id):
    job = _chat_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found or expired'}), 404
    job['cancelled'] = True
    job['status'] = 'cancelled'
    job['error'] = '已停止当前回复'
    job['ts'] = _time.time()
    return jsonify({'ok': True})


@bp.route('/api/chat/memory', methods=['GET'])
def chat_memory_get():
    return jsonify({
        "ok": True,
        "path": str(USER_PROFILE_MD),
        "content": get_user_profile_context(max_chars=20000),
        "exists": USER_PROFILE_MD.exists(),
        "archive_dir": str(USER_PROFILE_ARCHIVE_DIR),
        "source_path": str(USER_PROFILE_SOURCE_JSON),
    })


@bp.route('/api/chat/memory', methods=['DELETE'])
def chat_memory_delete():
    """Delete persistent user profile / personalization files."""
    deleted = []
    for path in [USER_PROFILE_MD, USER_PROFILE_SOURCE_JSON]:
        try:
            if path.exists():
                path.unlink()
                deleted.append(str(path))
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
    try:
        if USER_PROFILE_ARCHIVE_DIR.exists():
            for p in USER_PROFILE_ARCHIVE_DIR.glob("user_profile_*.md"):
                p.unlink()
                deleted.append(str(p))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify({"ok": True, "deleted": deleted})


@bp.route('/api/chat/memory/summarize', methods=['POST'])
def chat_memory_summarize():
    """Summarize multiple local conversations into a persistent Markdown user profile."""
    data = request.json or {}
    conversations = data.get("conversations") or []
    if not conversations:
        conversations = _load_persistent_conversations().get("conversations") or []
    if not conversations:
        return jsonify({"ok": False, "error": "No conversations available"}), 400

    snippets = []
    for conv in conversations[:20]:
        title = conv.get("title") or conv.get("id") or "Untitled"
        lines = [f"## Conversation: {title}"]
        for m in (conv.get("history") or [])[-20:]:
            role = m.get("role", "")
            content = str(m.get("content", ""))[:1200]
            if content.strip():
                lines.append(f"- {role}: {content}")
        snippets.append("\n".join(lines))
    corpus = "\n\n".join(snippets)[:30000]

    fallback = (
        "# SAGE User Profile\n\n"
        f"Updated: {datetime.now().isoformat(timespec='seconds')}\n\n"
        "## Inferred identity and work context\n"
        "- The user works on seismology/geophysics workflows and SAGE development.\n\n"
        "## Habits and preferences\n"
        "- Prefers executable workflows, debugging ability, intermediate artifacts, and traceable reasoning.\n"
        "- Often asks for Chinese explanations with practical code support.\n\n"
        "## Knowledge level\n"
        "- Advanced technical user familiar with earthquake monitoring, phase picking, RAG, skills, and coding agents.\n\n"
        "## Intent hints\n"
        "- Explanation-style requests should use QA/RAG.\n"
        "- Requests involving implementation, plotting, detection, processing, or '上述/这个方法 + 代码' should route to coding.\n"
    )

    try:
        llm_cfg = get_llm_config()
        if llm_cfg.get("api_base"):
            prompt = (
                "请根据以下多个对话，生成一份简洁、可长期使用的 Markdown 用户画像。"
                "目标是帮助后续助手判断用户身份、研究方向、习惯、知识水平、偏好的回答深度、"
                "常见意图和路由偏好。不要记录隐私敏感信息，不要逐字复述对话。\n\n"
                "请使用以下结构：\n"
                "# SAGE User Profile\n"
                "## Inferred identity and work context\n"
                "## Research/software interests\n"
                "## Habits and preferences\n"
                "## Knowledge level\n"
                "## Intent and routing hints\n"
                "## Open uncertainties\n\n"
                f"对话材料：\n{corpus}"
            )
            content = llm_call([{"role": "user", "content": prompt}], llm_cfg, max_tokens=1800)
        else:
            content = fallback
    except Exception:
        content = fallback

    _save_user_profile(content, conversations=conversations)
    return jsonify({
        "ok": True,
        "path": str(USER_PROFILE_MD),
        "archive_dir": str(USER_PROFILE_ARCHIVE_DIR),
        "source_path": str(USER_PROFILE_SOURCE_JSON),
        "content": content,
    })
