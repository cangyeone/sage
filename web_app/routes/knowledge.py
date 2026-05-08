"""知识库管理路由"""
from flask import Blueprint, request, jsonify, send_file
import sys
import os
import threading
import time as _time
import uuid as _uuid
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from state import (
    _kb_dir_status, _kb_dir_jobs, _ref_kb_dir_status, _ref_kb_jobs,
    _PROJECT_ROOT, _REF_KNOWLEDGE_DIR, _REF_KB_MANIFEST_DIR, tasks,
    UPLOAD_FOLDER_CHAT, _code_engine_lock,
)
from helpers import (
    get_kb_instance,
    get_ref_indexer,
    get_llm_config,
    llm_call,
    safe_child_path,
)

bp = Blueprint('knowledge', __name__)


def _safe_pdf_upload_path(filename: str | None) -> tuple[Path, str]:
    raw_name = Path(filename or "").name
    if Path(raw_name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")
    safe_stem = secure_filename(Path(raw_name).stem) or "upload"
    safe_name = f"{safe_stem}.pdf"
    tmp_name = f"kb_{_uuid.uuid4().hex}_{safe_name}"
    try:
        return safe_child_path(UPLOAD_FOLDER_CHAT, tmp_name), safe_name
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Invalid upload path")


# ── Knowledge base status and list ────────────────────────────────────────

@bp.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        kb = get_kb_instance()
        if kb:
            return jsonify({"ok": True, **kb.status()})
        return jsonify({"ok": False, "error": "Knowledge base unavailable", "n_docs": 0, "n_chunks": 0, "n_vectors": 0})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "n_docs": 0, "n_chunks": 0, "n_vectors": 0})


@bp.route('/api/knowledge/embedding_config', methods=['GET'])
def get_embedding_config():
    """返回当前嵌入模型配置（路径）。"""
    try:
        cfg_file = Path.home() / ".seismicx" / "config.json"
        cfg = {}
        if cfg_file.exists():
            import json as _json
            cfg = _json.loads(cfg_file.read_text(encoding="utf-8"))
        model_path = cfg.get("embedding", {}).get("model_path", "")
        return jsonify({"ok": True, "model_path": model_path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "model_path": ""})


@bp.route('/api/knowledge/embedding_config', methods=['POST'])
def set_embedding_config():
    """保存嵌入模型本地路径到 ~/.seismicx/config.json"""
    try:
        import json as _json
        data = request.get_json(force=True) or {}
        model_path = str(data.get("model_path", "")).strip()

        cfg_file = Path.home() / ".seismicx" / "config.json"
        cfg = {}
        if cfg_file.exists():
            try:
                cfg = _json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        if "embedding" not in cfg:
            cfg["embedding"] = {}
        cfg["embedding"]["model_path"] = model_path
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        cfg_file.write_text(_json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        # 重置 EmbeddingModel 单例，下次构建时以新路径重新加载
        try:
            from rag_engine import EmbeddingModel
            EmbeddingModel.reset()
        except Exception:
            pass

        return jsonify({"ok": True, "model_path": model_path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route('/api/knowledge/list', methods=['GET'])
def knowledge_list():
    try:
        kb = get_kb_instance()
        if kb:
            docs = [
                {"doc_id": d.doc_id, "doc_name": d.doc_name,
                 "n_pages": d.n_pages, "n_chunks": d.n_chunks,
                 "added_at": d.added_at,
                 "size_kb": round(d.size_bytes / 1024, 1),
                 "proj_folder": getattr(d, "proj_folder", ""),
                 "source_type": getattr(d, "source_type", "upload")}
                for d in kb.list_docs()
            ]
            return jsonify({"ok": True, "docs": docs})
        return jsonify({"ok": False, "error": "Knowledge base unavailable", "docs": []})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "docs": []})


# ── Knowledge base file operations ────────────────────────────────────────

@bp.route('/api/knowledge/upload', methods=['POST'])
def knowledge_upload():
    """Upload & index one PDF into the persistent knowledge base."""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files['file']
    try:
        tmp_path, doc_name = _safe_pdf_upload_path(f.filename)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Save temporarily
    f.save(str(tmp_path))

    # Index in background — return immediately with task_id
    task_id = f"kb_idx_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    tasks[task_id] = {"id": task_id, "type": "kb_index",
                      "status": "running", "doc_name": doc_name}

    def _index(tid, path, name):
        try:
            kb = get_kb_instance()
            if kb:
                logs = []
                meta = kb.add_pdf(str(path), progress_cb=lambda m: logs.append(m))
                tasks[tid]["status"]   = "completed"
                tasks[tid]["doc_name"] = name
                tasks[tid]["n_chunks"] = meta.n_chunks
                tasks[tid]["logs"]     = logs
            else:
                tasks[tid]["status"] = "error"
                tasks[tid]["error"]  = "Knowledge base unavailable"
        except Exception as ex:
            tasks[tid]["status"] = "error"
            tasks[tid]["error"]  = str(ex)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_index, args=(task_id, tmp_path, doc_name),
                     daemon=True).start()
    return jsonify({"ok": True, "task_id": task_id})


@bp.route('/api/knowledge/index_status/<task_id>', methods=['GET'])
def knowledge_index_status(task_id):
    t = tasks.get(task_id, {})
    return jsonify(t)


@bp.route('/api/knowledge/delete/<doc_id>', methods=['DELETE'])
def knowledge_delete(doc_id):
    try:
        kb = get_kb_instance()
        if kb:
            ok = kb.delete_doc(doc_id)
            return jsonify({"ok": ok, "error": "" if ok else "Document not found"})
        return jsonify({"ok": False, "error": "Knowledge base unavailable"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@bp.route('/api/knowledge/clear', methods=['POST'])
def knowledge_clear():
    try:
        kb = get_kb_instance()
        if kb:
            kb.clear()
            return jsonify({"ok": True})
        return jsonify({"ok": False, "error": "Knowledge base unavailable"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── Knowledge directory (seismo_skill/docs/) scan & build ────────────────

@bp.route('/api/knowledge/dir_status', methods=['GET'])
def knowledge_dir_status():
    """返回 seismo_skill/docs/ 目录的实时扫描状态。"""
    try:
        import sys as _s
        _proj = str(_PROJECT_ROOT)
        if _proj not in _s.path:
            _s.path.insert(0, _proj)
        from seismo_skill.knowledge_indexer import KnowledgeIndexer
        indexer = KnowledgeIndexer()
        summary = indexer.manifest_summary()
        summary["ok"] = True
        _kb_dir_status.update(summary)
        _kb_dir_status["checked"] = True
        return jsonify(summary)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), **_kb_dir_status})


@bp.route('/api/knowledge/build_from_dir', methods=['POST'])
def knowledge_build_from_dir():
    """启动后台任务：扫描 seismo_skill/docs/ 并按模式构建 RAG 或文件夹型 Skill。"""
    data = request.get_json(silent=True) or {}
    mode = (data.get("mode") or "both").strip().lower()
    if mode not in {"rag", "skill", "both"}:
        mode = "both"
    job_id = f"kbdir_{_uuid.uuid4().hex[:8]}"
    stop_ev = threading.Event()
    _kb_dir_jobs[job_id] = {
        "status": "running", "log": [], "result": None,
        "progress": 0, "stop_event": stop_ev, "mode": mode,
    }

    def _run(jid):
        job       = _kb_dir_jobs[jid]
        log_lines = job["log"]
        stop_event = job["stop_event"]
        try:
            import sys as _s
            _proj = str(_PROJECT_ROOT)
            if _proj not in _s.path:
                _s.path.insert(0, _proj)
            from seismo_skill.knowledge_indexer import KnowledgeIndexer
            indexer = KnowledgeIndexer()

            # Count pending files upfront for progress tracking
            scan = indexer.scan()
            total = len(scan.new) + len(scan.modified) + len(scan.failed)
            import re as _re
            _file_line_re = _re.compile(r"^\[(\d+)/\d+\]")

            def _progress_cb(msg):
                log_lines.append(msg)
                # Count "[i/N]" lines to track per-file progress (0–90%)
                m = _file_line_re.match(msg.strip())
                if m and total > 0:
                    done = int(m.group(1))
                    job["progress"] = max(5, int(done / total * 90))

            log_lines.append(f"▶ 构建模式：{mode}")
            if mode == "rag":
                result = indexer.build(
                    progress_cb=_progress_cb,
                    stop_event=stop_event,
                    skip_skill_gen=True,
                )
            elif mode == "skill":
                result = indexer.build_folder_skills(
                    progress_cb=_progress_cb,
                    stop_event=stop_event,
                    use_llm=True,
                )
            else:
                result = indexer.build(
                    progress_cb=_progress_cb,
                    stop_event=stop_event,
                    skip_skill_gen=True,
                )
                if not result.interrupted and not stop_event.is_set():
                    skill_result = indexer.build_folder_skills(
                        progress_cb=_progress_cb,
                        stop_event=stop_event,
                        use_llm=True,
                    )
                    result.skills_generated.extend(skill_result.skills_generated)
                    result.skipped.extend(skill_result.skipped)
                    result.failed.extend(skill_result.failed)
                    result.interrupted = result.interrupted or skill_result.interrupted

            if result.interrupted:
                job["status"] = "stopped"
            else:
                job["status"] = "done"
                job["progress"] = 100

            job["result"] = {
                "indexed": result.indexed,
                "skills_generated": result.skills_generated,
                "skipped": result.skipped,
                "failed": result.failed,
                "interrupted": result.interrupted,
                "mode": mode,
            }
        except Exception as exc:
            job["status"] = "error"
            job["result"] = {"error": str(exc)}
            log_lines.append(f"❌ 错误：{exc}")

    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "mode": mode})


@bp.route('/api/knowledge/build_from_dir/<job_id>', methods=['GET'])
def knowledge_build_from_dir_status(job_id):
    """轮询 build_from_dir 任务状态。"""
    job = _kb_dir_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    # Exclude non-serialisable stop_event from response
    resp = {k: v for k, v in job.items() if k != "stop_event"}
    return jsonify({"ok": True, **resp})


@bp.route('/api/knowledge/build_from_dir/<job_id>', methods=['DELETE'])
def knowledge_build_from_dir_stop(job_id):
    """中断后台构建任务。"""
    job = _kb_dir_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    stop_ev = job.get("stop_event")
    if stop_ev:
        stop_ev.set()
    return jsonify({"ok": True, "message": "stop signal sent"})


@bp.route('/api/knowledge/project/<path:proj_name>', methods=['DELETE'])
def knowledge_delete_project(proj_name):
    """删除一个知识目录/知识库分组的索引和关联 Skill。"""
    try:
        import sys as _s
        _proj = str(_PROJECT_ROOT)
        if _proj not in _s.path:
            _s.path.insert(0, _proj)
        from seismo_skill.knowledge_indexer import (
            KnowledgeIndexer, _USER_SKILL_DIR, delete_generated_builtin_skill,
        )

        indexer = KnowledgeIndexer()
        kb = get_kb_instance()
        proj_path = Path(proj_name)
        proj_candidates = {
            proj_name,
            proj_path.name,
            proj_path.stem,
        }

        # 1. 先从 RAG 元数据删除该 proj_folder 下的文档。
        # Chat/Project 入库的整体资料只存在 RAG metadata 中，不一定存在
        # KnowledgeIndexer manifest，所以必须以 RAG 元数据为准。
        removed_docs = []
        if kb:
            for doc in list(kb.list_docs()):
                folder = getattr(doc, "proj_folder", "") or ""
                if folder == proj_name:
                    try:
                        if kb.delete_doc(doc.doc_id):
                            removed_docs.append(doc.doc_id)
                    except Exception:
                        pass

        # 2. 兼容 skill docs / reference library 的目录索引 manifest。
        keys_to_del = []
        skill_names_to_del = set()
        for rel, entry in list(indexer._manifest.items()):
            rel_path = Path(rel)
            if (
                entry.get("proj_folder") in proj_candidates
                or rel == proj_name
                or rel_path.name == proj_name
                or rel_path.stem == proj_name
                or rel.startswith(proj_name + "/")
                or rel.startswith(proj_name + "\\")
            ):
                doc_id = entry.get("doc_id")
                if doc_id and doc_id not in removed_docs:
                    try:
                        if kb:
                            if kb.delete_doc(doc_id):
                                removed_docs.append(doc_id)
                    except Exception:
                        pass
                if entry.get("skill_name"):
                    skill_names_to_del.add(entry.get("skill_name"))
                keys_to_del.append(rel)

        for k in keys_to_del:
            indexer._manifest.pop(k, None)
        if keys_to_del:
            indexer._save_manifest()

        # 3. 删除由目录索引生成的 Skill 文件（Chat/Project 入库不会有这个条目）
        proj_entries = []
        for key in list(indexer._proj_manifest.keys()):
            entry = indexer._proj_manifest.get(key) or {}
            source_name = Path(str(entry.get("source_path", ""))).name
            source_stem = Path(str(entry.get("source_path", ""))).stem
            if key in proj_candidates or source_name in proj_candidates or source_stem in proj_candidates:
                proj_entries.append(indexer._proj_manifest.pop(key))
        if proj_entries:
            indexer._save_proj_manifest()
        for proj_entry in proj_entries:
            skill_name = proj_entry.get("skill_name", "")
            if skill_name:
                skill_names_to_del.add(skill_name)

        for skill_name in skill_names_to_del:
            skill_file = _USER_SKILL_DIR / f"{skill_name}.md"
            skill_file.unlink(missing_ok=True)
            delete_generated_builtin_skill(skill_name)
        if skill_names_to_del:
            # Invalidate skill cache
            try:
                from seismo_skill import skill_loader as _sl
                _sl.invalidate_cache()
            except Exception:
                pass

        return jsonify({
            "ok": True,
            "proj_name": proj_name,
            "removed_files": len(keys_to_del),
            "removed_docs": len(removed_docs),
            "skill_deleted": bool(skill_names_to_del),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Reference knowledge (seismo_knowledge/) API ────────────────────────────

@bp.route('/api/ref_knowledge/dir_status', methods=['GET'])
def ref_knowledge_dir_status():
    """返回 seismo_knowledge/ 目录的实时扫描状态（无 Skill 生成）。"""
    try:
        indexer = get_ref_indexer()
        summary = indexer.manifest_summary()
        summary["ok"] = True
        summary["knowledge_dir"] = str(_REF_KNOWLEDGE_DIR)
        _ref_kb_dir_status.update(summary)
        _ref_kb_dir_status["checked"] = True
        return jsonify(summary)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), **_ref_kb_dir_status})


@bp.route('/api/ref_knowledge/build_from_dir', methods=['POST'])
def ref_knowledge_build():
    """启动后台任务：扫描 seismo_knowledge/ 并构建/更新 RAG 索引（不生成 Skill）。"""
    job_id = f"refkb_{_uuid.uuid4().hex[:8]}"
    stop_ev = threading.Event()
    _ref_kb_jobs[job_id] = {
        "status": "running", "log": [], "result": None,
        "progress": 0, "stop_event": stop_ev,
    }

    def _run(jid):
        job = _ref_kb_jobs[jid]
        log_lines = job["log"]
        stop_event = job["stop_event"]
        try:
            indexer = get_ref_indexer()
            scan = indexer.scan()
            total = len(scan.new) + len(scan.modified) + len(scan.failed)
            import re as _re
            _file_line_re = _re.compile(r"^\[(\d+)/\d+\]")

            def _progress_cb(msg):
                log_lines.append(msg)
                m = _file_line_re.match(msg.strip())
                if m and total > 0:
                    done = int(m.group(1))
                    job["progress"] = max(5, int(done / total * 90))

            result = indexer.build(
                progress_cb=_progress_cb,
                stop_event=stop_event,
                skip_skill_gen=True,
            )
            if result.interrupted:
                job["status"] = "stopped"
            else:
                job["status"] = "done"
                job["progress"] = 100
            job["result"] = {
                "indexed": result.indexed,
                "skipped": result.skipped,
                "failed": result.failed,
                "interrupted": result.interrupted,
            }
        except Exception as exc:
            job["status"] = "error"
            job["result"] = {"error": str(exc)}
            log_lines.append(f"❌ 错误：{exc}")

    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})


@bp.route('/api/ref_knowledge/build_from_dir/<job_id>', methods=['GET'])
def ref_knowledge_build_status(job_id):
    """轮询参考文献库构建任务状态。"""
    job = _ref_kb_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    resp = {k: v for k, v in job.items() if k != "stop_event"}
    return jsonify({"ok": True, **resp})


@bp.route('/api/ref_knowledge/build_from_dir/<job_id>', methods=['DELETE'])
def ref_knowledge_build_stop(job_id):
    """中断参考文献库构建任务。"""
    job = _ref_kb_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    stop_ev = job.get("stop_event")
    if stop_ev:
        stop_ev.set()
    return jsonify({"ok": True, "message": "stop signal sent"})


@bp.route('/api/ref_knowledge/collection/<path:coll_name>', methods=['DELETE'])
def ref_knowledge_delete_collection(coll_name):
    """删除一个参考文献集合的 RAG 索引（不删除原始文件）。"""
    try:
        indexer = get_ref_indexer()
        coll_path = Path(coll_name)
        coll_candidates = {
            coll_name,
            coll_path.name,
            coll_path.stem,
        }
        removed_docs = []
        keys_to_del = []
        for rel, entry in list(indexer._manifest.items()):
            rel_path = Path(rel)
            if (
                entry.get("proj_folder") in coll_candidates
                or rel == coll_name
                or rel_path.name == coll_name
                or rel_path.stem == coll_name
                or rel.startswith(coll_name + "/")
                or rel.startswith(coll_name + "\\")
            ):
                doc_id = entry.get("doc_id")
                if doc_id:
                    try:
                        kb = get_kb_instance()
                        if kb:
                            kb.delete_doc(doc_id)
                            removed_docs.append(doc_id)
                    except Exception:
                        pass
                keys_to_del.append(rel)

        for k in keys_to_del:
            indexer._manifest.pop(k, None)
        if keys_to_del:
            indexer._save_manifest()

        indexer._proj_manifest.pop(coll_name, None)
        indexer._proj_manifest.pop(Path(coll_name).stem, None)
        indexer._save_proj_manifest()

        return jsonify({
            "ok": True,
            "coll_name": coll_name,
            "removed_files": len(keys_to_del),
            "removed_docs": len(removed_docs),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Knowledge retrieval ────────────────────────────────────────────────────

@bp.route('/api/knowledge/retrieve', methods=['POST'])
def knowledge_retrieve():
    """直接检索知识库中高度相关的文献段落。"""
    data  = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "query 不能为空"}), 400

    top_k     = int(data.get("top_k", 8))
    threshold = float(data.get("score_threshold", 0.8))

    try:
        kb = get_kb_instance()
        if kb and kb.is_empty:
            return jsonify({"ok": True, "query": query, "n_results": 0, "results": [],
                            "message": "知识库为空，请先上传文献 PDF"})

        if kb:
            results = kb.retrieve_relevant_docs(query, top_k=top_k, score_threshold=threshold)
            return jsonify({
                "ok":       True,
                "query":    query,
                "n_results": len(results),
                "results":  results,
            })
        return jsonify({"ok": False, "error": "Knowledge base unavailable"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
