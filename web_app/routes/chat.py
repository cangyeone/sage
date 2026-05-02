"""聊天和任务管理路由"""
from flask import Blueprint, request, jsonify, send_file, Response, stream_with_context
import os
import sys
import threading
import subprocess
import time as _time
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from state import (
    tasks, _session_docs, _geo_agent_jobs, _lit_jobs, _chat_jobs,
    UPLOAD_FOLDER_CHAT, GEO_WORKSPACE_ROOT, _PROJECT_ROOT,
)
from helpers import get_llm_config, llm_call, inject_workspace_context, get_kb_instance

bp = Blueprint('chat', __name__)


# ── Task management ──────────────────────────────────────────────────────

def run_task(task_id, command, task_type, cwd=None):
    """Run a seismic processing task in background"""
    try:
        tasks[task_id]['status'] = 'running'
        tasks[task_id]['start_time'] = datetime.now().isoformat()

        result = subprocess.run(
            command,
            shell=True,
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
    data = request.json

    if not data.get('input_dir'):
        return jsonify({'error': 'Input directory required'}), 400

    task_id = f"pick_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # Build command
    cmd = f"python pnsn/picker.py"
    cmd += f" -i {data['input_dir']}"
    cmd += f" -o web_app/outputs/{task_id}"
    cmd += f" -m {data.get('model', 'pnsn/pickers/pnsn.v3.jit')}"
    cmd += f" -d {data.get('device', 'cpu')}"

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'phase_picking',
        'status': 'queued',
        'command': cmd,
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
    data = request.json

    if not data.get('input_file'):
        return jsonify({'error': 'Input file required'}), 400

    if not data.get('station_file'):
        return jsonify({'error': 'Station file required'}), 400

    task_id = f"assoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # Select method
    method_scripts = {
        'fastlink': 'pnsn/fastlinker.py',
        'real': 'pnsn/reallinker.py',
        'gamma': 'pnsn/gammalink.py'
    }

    script = method_scripts.get(data.get('method', 'fastlink'), 'pnsn/fastlinker.py')

    # Build command
    cmd = f"python {script}"
    cmd += f" -i {data['input_file']}"
    cmd += f" -o web_app/outputs/{task_id}.txt"
    cmd += f" -s {data['station_file']}"

    if data.get('method') == 'fastlink':
        cmd += f" -d {data.get('device', 'cpu')}"

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'phase_association',
        'status': 'queued',
        'command': cmd,
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
    data = request.json

    if not data.get('input_file'):
        return jsonify({'error': 'Input file required'}), 400

    if not data.get('waveform_dir'):
        return jsonify({'error': 'Waveform directory required'}), 400

    task_id = f"polar_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # For now, use CLI tool
    cmd = f"python seismic_cli.py polarity"
    cmd += f" -i {data['input_file']}"
    cmd += f" -w {data['waveform_dir']}"
    cmd += f" -o web_app/outputs/{task_id}_polarity.txt"
    cmd += f" --model {data.get('model', 'pnsn/pickers/polar.onnx')}"
    cmd += f" --min-confidence {data.get('min_confidence', 0.5)}"
    cmd += f" --phase {data.get('phase', 'Pg')}"

    # Initialize task
    tasks[task_id] = {
        'id': task_id,
        'type': 'polarity_analysis',
        'status': 'queued',
        'command': cmd,
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
    _geo_agent_jobs[job_id] = {
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
            from sage_agents import EvidenceDrivenGeoAgent, AgentConfig

            # Build config from request
            cfg = AgentConfig(
                workspace_root=data.get("workspace_root") or ".",
                literature_root=data.get("literature_root") or "",
                output_dir=data.get("output_dir") or "outputs/evidence_driven_geo_agent",
                allow_python=bool(data.get("allow_python", True)),
                allow_shell=bool(data.get("allow_shell", False)),
                allow_web_search=bool(data.get("allow_web_search", False)),
                use_multimodal=bool(data.get("use_multimodal", False)),
                use_rag=bool(data.get("use_rag", True)),
                use_local_files=bool(data.get("use_local_files", True)),
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

            agent  = EvidenceDrivenGeoAgent(config=cfg, llm_cfg=get_llm_config())
            result = agent.run(question, study_area, on_progress=_prog)
            _geo_agent_jobs[job_id]["status"] = "done"
            _geo_agent_jobs[job_id]["result"] = result
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

        if search_type == 'scholar':
            result = tool.dispatch('scholar_search', {'query': query, 'max_results': max_results})
        else:
            result = tool.dispatch('web_search', {'query': query, 'max_results': max_results})

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
    session_id = request.form.get('session_id', 'default')

    if not f.filename.lower().endswith('.pdf'):
        return jsonify({"ok": False, "error": "Only PDF files are supported"}), 400

    tmp_path = UPLOAD_FOLDER_CHAT / f"{session_id}_{f.filename}"
    f.save(str(tmp_path))

    try:
        # Extract text (no BGE-M3, just raw text for session context)
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_engine import _extract_pdf_text, _chunk_text
        pages  = _extract_pdf_text(str(tmp_path))
        chunks = []
        for page_idx, page_text in pages:
            for c in _chunk_text(page_text, chunk_size=600):
                chunks.append({"page": page_idx + 1, "text": c})

        if session_id not in _session_docs:
            _session_docs[session_id] = {"chunks": [], "doc_names": []}

        _session_docs[session_id]["chunks"].extend(chunks)
        _session_docs[session_id]["doc_names"].append(f.filename)

        return jsonify({
            "ok": True,
            "doc_name": f.filename,
            "n_pages":  len(pages),
            "n_chunks": len(chunks),
            "session_id": session_id,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@bp.route('/api/chat/clear_session', methods=['POST'])
def chat_clear_session():
    sid = request.json.get('session_id', 'default')
    _session_docs.pop(sid, None)
    return jsonify({"ok": True})


# ── RAG 增强对话 ──────────────────────────────────────────────────────────────

@bp.route('/api/chat/rag', methods=['POST'])
def chat_rag():
    """RAG-aware chat endpoint."""
    data       = request.json or {}
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
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
            skill_ctx = sl.build_skill_context(user_msg, max_chars=3000, top_k=2)
            if skill_ctx:
                context_parts.append("===== 可用技能与函数示例 =====\n" + skill_ctx)
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

def _build_rag_messages(data: dict):
    """
    Shared helper: build (messages, sources, llm_cfg) for both /api/chat/rag
    and its streaming twin.  Returns None for llm_cfg if backend not configured.
    """
    import json as _json
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
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
    if session.get("chunks"):
        query_words = set(user_msg.lower().split())
        scored = []
        for c in session["chunks"]:
            words = set(c["text"].lower().split())
            score = len(query_words & words) / (len(query_words) + 1)
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, c in scored[:4]:
            if score > 0 or mode == "paper_read":
                context_parts.append(f"[上传文档 第{c['page']}页]\n{c['text']}")
        if session.get("doc_names"):
            sources.extend(session["doc_names"])

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
            skill_ctx = sl.build_skill_context(user_msg, max_chars=3000, top_k=2)
            if skill_ctx:
                context_parts.append("===== 可用技能与函数示例 =====\n" + skill_ctx)
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

    # Think-mode: model will produce <think>…</think> naturally for deepseek-r1/QwQ;
    # for other models we add a soft prompt to encourage reasoning.
    enable_think = bool(data.get("enable_think", False))
    model_name = llm_cfg.get("model", "").lower()
    is_reasoning_model = any(keyword in model_name for keyword in ["deepseek-r1", "qwq", "r1"])
    
    if enable_think:
        if is_reasoning_model:
            # For reasoning models, they naturally produce <think> tags
            system += (
                "\n\n请在正式回答前先进行详细推理，将思考过程放在 <think>…</think> 标签内，"
                "然后在标签之外给出最终回答。"
            )
        else:
            # For other models, use a softer prompt without requiring specific tags
            system += (
                "\n\n请在回答前先进行详细推理，然后给出最终回答。"
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
            for chunk in llm_stream(messages, llm_cfg, max_tokens=2000):
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

    llm_cfg = get_llm_config()

    enable_think = bool(data.get("enable_think", False))
    model_name = llm_cfg.get("model", "").lower()
    is_reasoning_model = any(keyword in model_name for keyword in ["deepseek-r1", "qwq", "r1"])
    
    system = (
        "You are SAGE, an expert seismology assistant with deep knowledge of "
        "seismology, geophysics and data processing.\n"
        "Answer the user's question using your own knowledge. Be concise and accurate.\n"
    )
    if enable_think:
        if is_reasoning_model:
            system += (
                "\n请在正式回答前先进行详细推理，将思考过程放在 <think>…</think> 标签内，"
                "然后在标签之外给出最终回答。"
            )
        else:
            # For other models, use a softer prompt without requiring specific tags
            system += (
                "\n\n请在回答前先进行详细推理，然后给出最终回答。"
            )

    workspace_path = data.get("workspace", "")
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            system += "\n\n===== 本地文件系统 =====\n" + ws_ctx

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
            for chunk in llm_stream(messages, llm_cfg, max_tokens=2000):
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
    model_name   = llm_cfg.get('model', '').lower()
    is_reasoning = any(k in model_name for k in ['deepseek-r1', 'qwq', 'r1', 'deepseek-reasoner'])

    system = (
        'You are SAGE, an expert seismology assistant with deep knowledge of '
        'seismology, geophysics and data processing.\n'
        'Answer the user\'s question using your own knowledge. Be concise and accurate.\n'
    )
    if enable_think:
        system += (
            '\n请在正式回答前先进行详细推理，将思考过程放在 <think>…</think> 标签内，'
            '然后在标签之外给出最终回答。'
            if is_reasoning else
            '\n\n请在回答前先进行详细推理，然后给出最终回答。'
        )

    workspace_path = data.get('workspace', '')
    if workspace_path:
        ws_ctx = inject_workspace_context(user_msg, workspace_path)
        if ws_ctx:
            system += '\n\n===== 本地文件系统 =====\n' + ws_ctx

    messages = [{'role': 'system', 'content': system},
                {'role': 'user',   'content': user_msg}]
    history = data.get('history', [])
    if history:
        messages = [messages[0]] + history[-6:] + [messages[-1]]
    return messages, [], llm_cfg


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
        'error': '', 'ts': _time.time(),
    }

    chat_type = data.get('type', 'rag')

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

            answer = llm_call(messages, llm_cfg, max_tokens=2000)
            _chat_jobs[job_id].update(answer=answer, sources=sources, status='done')
        except Exception as exc:
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
