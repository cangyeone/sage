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


def _proj_match(value: str | None, candidates: set[str]) -> bool:
    val = str(value or "").strip()
    if not val:
        return False
    p = Path(val)
    return val in candidates or p.name in candidates or p.stem in candidates


def _bulk_delete_kb_docs(kb, doc_ids: set[str]) -> int:
    """Delete many KB docs with one FAISS rebuild instead of one rebuild per doc."""
    if not kb or not doc_ids:
        return 0
    removed = 0
    try:
        docs = getattr(kb, "_docs", None)
        chunks = getattr(kb, "_chunks", None)
        if isinstance(docs, dict) and isinstance(chunks, dict):
            for doc_id in list(doc_ids):
                meta = docs.pop(doc_id, None)
                if not meta:
                    continue
                removed += 1
                file_path = getattr(meta, "file_path", "") or ""
                if file_path:
                    try:
                        Path(file_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            for cid, ch in list(chunks.items()):
                if getattr(ch, "doc_id", "") in doc_ids:
                    chunks.pop(cid, None)
            try:
                kb._rebuild_faiss()
                kb._save_state()
            except Exception:
                pass
        else:
            for doc_id in list(doc_ids):
                try:
                    if kb.delete_doc(doc_id):
                        removed += 1
                except Exception:
                    pass

        try:
            from rag_engine import _get_simple_rag  # type: ignore
            sr = _get_simple_rag()
            for doc_id in list(doc_ids):
                try:
                    sr.delete_document(doc_id)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        for doc_id in list(doc_ids):
            try:
                if kb.delete_doc(doc_id):
                    removed += 1
            except Exception:
                pass
    return removed


def _delete_generated_skill_by_name(skill_name: str) -> bool:
    if not skill_name:
        return False
    deleted = False
    try:
        from seismo_skill.knowledge_indexer import _USER_SKILL_DIR, delete_generated_builtin_skill
        skill_file = _USER_SKILL_DIR / f"{skill_name}.md"
        if skill_file.exists():
            skill_file.unlink(missing_ok=True)
            deleted = True
        if delete_generated_builtin_skill(skill_name):
            deleted = True
    except Exception:
        pass
    try:
        from seismo_skill import skill_loader as _sl
        _sl.invalidate_cache()
    except Exception:
        pass
    return deleted


def _safe_asset_id(text: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text or "")).strip("._-") or "asset"


def _folder_size_bytes(path: Path) -> int:
    try:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
    except Exception:
        return 0


def _extract_skill_title(skill_md: Path, fallback: str) -> str:
    try:
        text = skill_md.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip() or fallback
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("name:"):
                return s.split(":", 1)[1].strip() or fallback
    except Exception:
        pass
    return fallback


def _list_generated_skill_assets() -> list[dict]:
    """Return generated OpenAI-style SKILL folders as right-panel knowledge assets."""
    assets: list[dict] = []
    seen: set[str] = set()
    try:
        from seismo_skill.knowledge_indexer import (
            _BUILTIN_SKILL_DIR,
            _DOC_SKILL_GENERATOR,
            _USER_SKILL_DIR,
            KnowledgeIndexer,
        )
        indexer = KnowledgeIndexer()

        def _add_skill(folder: Path, proj_folder: str = "", status: str = "skill_asset"):
            key = str(folder.resolve())
            if key in seen:
                return
            seen.add(key)
            skill_md = folder / "SKILL.md"
            if status == "skill_asset":
                try:
                    text = skill_md.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    text = ""
                if f"generated_by: {_DOC_SKILL_GENERATOR}" not in text:
                    return
            title = _extract_skill_title(skill_md, folder.name) if skill_md.exists() else folder.name.replace(".building", "")
            subskills = len(list((folder / "subskills").glob("*.md"))) if (folder / "subskills").exists() else 0
            refs = len(list((folder / "references").glob("*.md"))) if (folder / "references").exists() else 0
            try:
                mtime = datetime.fromtimestamp(folder.stat().st_mtime).isoformat(timespec="seconds")
            except Exception:
                mtime = datetime.now().isoformat(timespec="seconds")
            assets.append({
                "doc_id": f"skill__{_safe_asset_id(folder.name)}",
                "doc_name": title,
                "n_pages": subskills,
                "n_chunks": refs,
                "added_at": mtime,
                "size_kb": round(_folder_size_bytes(folder) / 1024, 1),
                "proj_folder": proj_folder,
                "source_type": status,
                "skill_name": folder.name.replace(".building", "").strip("."),
                "skill_path": str(folder),
            })

        for proj_name, entry in list(getattr(indexer, "_proj_manifest", {}).items()):
            folder = Path(str(entry.get("skill_path") or ""))
            if folder.exists() and (folder / "SKILL.md").exists():
                _add_skill(folder, proj_folder=str(proj_name), status="skill_asset")

        for root in (_USER_SKILL_DIR, _BUILTIN_SKILL_DIR):
            if not root.exists():
                continue
            for folder in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
                if not folder.is_dir():
                    continue
                if folder.name.endswith(".building"):
                    _add_skill(folder, proj_folder=folder.name.replace(".building", "").strip("._"), status="skill_building")
                elif (folder / "SKILL.md").exists():
                    _add_skill(folder, status="skill_asset")
    except Exception:
        pass
    return assets


def _cleanup_generated_skill_artifacts_for_project(proj_name: str, skill_names: set[str] | None = None) -> int:
    """Remove generated skill folders/temp build dirs for a docs project, including failed builds."""
    removed = 0
    try:
        import shutil
        from seismo_skill.knowledge_indexer import (
            _BUILTIN_SKILL_DIR,
            _DOC_SKILL_GENERATOR,
            _USER_SKILL_DIR,
            _generated_skill_slug,
            _safe_skill_slug,
        )

        names = {str(n or "").strip() for n in (skill_names or set()) if str(n or "").strip()}
        source_bits = {
            str(proj_name or "").strip(),
            Path(str(proj_name or "")).name,
            Path(str(proj_name or "")).stem,
        }
        for bit in list(source_bits):
            if not bit:
                continue
            names.add(_safe_skill_slug(bit))
            names.add(_generated_skill_slug(bit))

        normalized = {n.lower().strip("._-") for n in names if n}
        for root in (_USER_SKILL_DIR, _BUILTIN_SKILL_DIR):
            if not root.exists():
                continue
            for child in list(root.iterdir()):
                child_key = child.name.lower().replace(".building", "").strip("._-")
                is_temp_build = child.name.endswith(".building")
                name_matches = child_key in normalized or any(n and n in child_key for n in normalized)
                generated_marker = False
                skill_md = child / "SKILL.md" if child.is_dir() else child
                if skill_md.exists() and skill_md.is_file():
                    try:
                        text = skill_md.read_text(encoding="utf-8", errors="ignore")
                        generated_marker = f"generated_by: {_DOC_SKILL_GENERATOR}" in text
                    except Exception:
                        generated_marker = False
                # Failed builds often leave hidden ".<skill>.building" folders without SKILL.md.
                # Completed folders are deleted only when they are docs-generated to avoid
                # removing hand-written user skills with the same name.
                if not name_matches or (not is_temp_build and not generated_marker):
                    continue
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
                removed += 1
        if removed:
            try:
                from seismo_skill import skill_loader as _sl
                _sl.invalidate_cache()
            except Exception:
                pass
    except Exception:
        pass
    return removed


def _clear_skill_docs_persistent_rag(indexer, kb) -> tuple[int, int]:
    """Remove persistent RAG index entries created from seismo_skill/docs, keeping generated SKILLs."""
    doc_ids = set()
    keys_to_del = []
    try:
        for rel, entry in list(getattr(indexer, "_manifest", {}).items()):
            if entry.get("status") != "indexed":
                continue
            doc_id = entry.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)
            keys_to_del.append(rel)
        removed_docs = _bulk_delete_kb_docs(kb, doc_ids)
        for rel in keys_to_del:
            indexer._manifest.pop(rel, None)
        if keys_to_del:
            indexer._save_manifest()
        return removed_docs, len(keys_to_del)
    except Exception:
        return 0, 0


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
        skill_assets = _list_generated_skill_assets()
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
            docs.extend(skill_assets)
            return jsonify({"ok": True, "docs": docs})
        return jsonify({"ok": True, "docs": skill_assets, "warning": "Knowledge base unavailable"})
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
        if str(doc_id).startswith("skill__"):
            skill_key = str(doc_id)[len("skill__"):]
            try:
                import sys as _s
                import shutil as _shutil
                _proj = str(_PROJECT_ROOT)
                if _proj not in _s.path:
                    _s.path.insert(0, _proj)
                from seismo_skill.knowledge_indexer import KnowledgeIndexer, _BUILTIN_SKILL_DIR, _USER_SKILL_DIR
                indexer = KnowledgeIndexer()
                skill_names = {skill_key}
                proj_names = set()
                direct_deleted = 0
                for root in (_USER_SKILL_DIR, _BUILTIN_SKILL_DIR):
                    if not root.exists():
                        continue
                    for folder in list(root.iterdir()):
                        if folder.is_dir() and _safe_asset_id(folder.name) == skill_key:
                            _shutil.rmtree(folder, ignore_errors=True)
                            direct_deleted += 1
                for key, entry in list(indexer._proj_manifest.items()):
                    skill_name = str(entry.get("skill_name") or "")
                    skill_path = Path(str(entry.get("skill_path") or ""))
                    if (
                        _safe_asset_id(skill_name) == skill_key
                        or _safe_asset_id(skill_path.name) == skill_key
                        or skill_path.name == skill_key
                    ):
                        proj_names.add(str(key))
                        if skill_name:
                            skill_names.add(skill_name)
                        indexer._proj_manifest.pop(key, None)
                if proj_names or skill_names:
                    indexer._save_proj_manifest()
                skill_deleted = False
                for skill_name in skill_names:
                    skill_deleted = _delete_generated_skill_by_name(skill_name) or skill_deleted
                artifacts_deleted = 0
                for name in (proj_names | skill_names | {skill_key}):
                    artifacts_deleted += _cleanup_generated_skill_artifacts_for_project(name, skill_names)
                artifacts_deleted += direct_deleted
                return jsonify({
                    "ok": skill_deleted or artifacts_deleted > 0,
                    "removed_docs": 0,
                    "removed_manifest": len(proj_names),
                    "skill_deleted": skill_deleted,
                    "artifacts_deleted": artifacts_deleted,
                    "error": "" if (skill_deleted or artifacts_deleted > 0) else "Skill not found",
                })
            except Exception as exc:
                return jsonify({"ok": False, "error": str(exc)}), 500

        kb = get_kb_instance()
        if not kb:
            return jsonify({"ok": False, "error": "Knowledge base unavailable"})

        doc_meta = None
        try:
            doc_meta = getattr(kb, "_docs", {}).get(doc_id)
        except Exception:
            doc_meta = None

        removed_docs = _bulk_delete_kb_docs(kb, {doc_id})
        ok = removed_docs > 0

        removed_manifest = 0
        skill_deleted = False
        try:
            import sys as _s
            _proj = str(_PROJECT_ROOT)
            if _proj not in _s.path:
                _s.path.insert(0, _proj)
            from seismo_skill.knowledge_indexer import KnowledgeIndexer
            indexer = KnowledgeIndexer()
            touched_proj = getattr(doc_meta, "proj_folder", "") if doc_meta else ""
            skill_names = set()
            for rel, entry in list(indexer._manifest.items()):
                if entry.get("doc_id") == doc_id:
                    if entry.get("skill_name"):
                        skill_names.add(entry.get("skill_name"))
                    if entry.get("proj_folder"):
                        touched_proj = entry.get("proj_folder")
                    indexer._manifest.pop(rel, None)
                    removed_manifest += 1

            if touched_proj:
                has_project_docs = any(
                    getattr(d, "proj_folder", "") == touched_proj
                    for d in (kb.list_docs() if kb else [])
                ) or any(
                    e.get("proj_folder") == touched_proj
                    for e in indexer._manifest.values()
                )
                if not has_project_docs:
                    proj_entry = indexer._proj_manifest.pop(touched_proj, None)
                    if proj_entry and proj_entry.get("skill_name"):
                        skill_names.add(proj_entry.get("skill_name"))

            if removed_manifest:
                indexer._save_manifest()
            if skill_names:
                indexer._save_proj_manifest()
                for skill_name in skill_names:
                    skill_deleted = _delete_generated_skill_by_name(skill_name) or skill_deleted
            artifact_proj = touched_proj or (getattr(doc_meta, "doc_name", "") if doc_meta else "")
            artifact_removed = _cleanup_generated_skill_artifacts_for_project(artifact_proj, skill_names)
            skill_deleted = skill_deleted or artifact_removed > 0
        except Exception:
            pass

        return jsonify({
            "ok": ok,
            "error": "" if ok else "Document not found",
            "removed_docs": removed_docs,
            "removed_manifest": removed_manifest,
            "skill_deleted": skill_deleted,
            "artifacts_deleted": locals().get("artifact_removed", 0),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
    """启动后台任务：扫描 seismo_skill/docs/ 并构建 SKILL，可选 RAG/向量辅助。"""
    data = request.get_json(silent=True) or {}
    legacy_mode = (data.get("mode") or "").strip().lower()
    style = (data.get("style") or data.get("skill_style") or "").strip().lower()
    raw_rag_assist = data.get("rag_assist", True)
    if isinstance(raw_rag_assist, str):
        rag_assist = raw_rag_assist.strip().lower() not in {"0", "false", "no", "off"}
    else:
        rag_assist = bool(raw_rag_assist)
    try:
        rag_cluster_target = int(data.get("rag_cluster_target") or data.get("cluster_target") or 0)
    except Exception:
        rag_cluster_target = 0
    if rag_cluster_target < 0:
        rag_cluster_target = 0

    # Backward compatibility for older frontends/API clients.
    if legacy_mode in {"rag", "skill", "both"} and not style:
        if legacy_mode == "rag":
            style = "rag_only"
            rag_assist = True
        elif legacy_mode == "skill":
            style = "openai"
            rag_assist = False
        else:
            style = "openai"
            rag_assist = True
    if style not in {"openai", "traditional", "rag_only"}:
        style = "openai"
    job_id = f"kbdir_{_uuid.uuid4().hex[:8]}"
    stop_ev = threading.Event()
    _kb_dir_jobs[job_id] = {
        "status": "running", "log": [], "result": None,
        "progress": 0, "stop_event": stop_ev, "style": style,
        "rag_assist": rag_assist, "rag_cluster_target": rag_cluster_target,
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
            from seismo_skill.knowledge_indexer import BuildResult, KnowledgeIndexer
            indexer = KnowledgeIndexer()

            # Count pending files upfront for progress tracking
            scan = indexer.scan()
            total = len(scan.new) + len(scan.modified) + len(scan.failed)
            import re as _re
            _file_line_re = _re.compile(r"^\[(\d+)/\d+\]")
            _llm_batch_re = _re.compile(r".*LLM 批次\s+(\d+)/(\d+)")

            def _progress_cb(msg):
                log_lines.append(msg)
                # Count "[i/N]" lines to track per-file progress (0–90%)
                stripped = msg.strip()
                m = _file_line_re.match(stripped)
                if m and total > 0:
                    done = int(m.group(1))
                    job["progress"] = max(5, int(done / total * 90))
                    return
                if "复用已标准化文档缓存" in stripped:
                    job["progress"] = max(job.get("progress", 0), 72)
                    return
                if "Skill Builder Agent" in stripped:
                    job["progress"] = max(job.get("progress", 0), 80)
                    return
                if "DBSCAN 文档聚类" in stripped:
                    job["progress"] = max(job.get("progress", 0), 86)
                    return
                bm = _llm_batch_re.match(stripped)
                if bm:
                    done = int(bm.group(1))
                    batch_total = max(1, int(bm.group(2)))
                    job["progress"] = max(job.get("progress", 0), 88 + int(done / batch_total * 9))

            style_label = {
                "openai": "OpenAI-style 文件夹 SKILL",
                "traditional": "传统单文件 SKILL",
                "rag_only": "仅构建 RAG 索引",
            }.get(style, style)
            log_lines.append(f"▶ SKILL 结构：{style_label}")
            if style == "rag_only":
                log_lines.append("▶ 持久化 RAG 索引：开启")
            elif style == "openai":
                log_lines.append("▶ 持久化 RAG 索引：跳过（仅使用临时向量/聚类辅助 SKILL 构建）")
            else:
                log_lines.append("▶ 持久化 RAG 索引：开启（传统单文件 SKILL 依赖索引 chunks）")
            log_lines.append(f"▶ 临时向量辅助：{'开启' if rag_assist else '关闭'}")
            if rag_assist:
                log_lines.append(f"▶ 目标主题簇数：{rag_cluster_target if rag_cluster_target > 0 else '自动建议'}")
            if style == "rag_only":
                result = indexer.build(
                    progress_cb=_progress_cb,
                    stop_event=stop_event,
                    skip_skill_gen=True,
                )
            elif style == "traditional":
                # 传统单文件 Skill 由已索引 chunks 生成，因此保留索引步骤。
                result = indexer.build(
                    progress_cb=_progress_cb,
                    stop_event=stop_event,
                    skip_skill_gen=False,
                )
            else:
                # OpenAI-style folder skills are built from source documents.
                # RAG/embedding assist is temporary clustering inside the builder,
                # not a persistent RAG-indexing step; otherwise large docs are read
                # twice and users see "RAG completed" followed by another full read.
                result = BuildResult()
                if not result.interrupted and not stop_event.is_set():
                    skill_result = indexer.build_folder_skills(
                        progress_cb=_progress_cb,
                        stop_event=stop_event,
                        use_llm=True,
                        rag_assist=rag_assist,
                        rag_cluster_target=rag_cluster_target,
                    )
                    result.skills_generated.extend(skill_result.skills_generated)
                    result.skipped.extend(skill_result.skipped)
                    result.failed.extend(skill_result.failed)
                    result.interrupted = result.interrupted or skill_result.interrupted
                if not result.interrupted:
                    removed_docs, removed_manifest = _clear_skill_docs_persistent_rag(indexer, get_kb_instance())
                    if removed_docs or removed_manifest:
                        log_lines.append(
                            f"🧹 OpenAI-style 构建完成：已清理持久化 RAG 索引 "
                            f"({removed_docs} docs, {removed_manifest} manifest entries)，仅保留 SKILL。"
                        )
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
                "style": style,
                "rag_assist": rag_assist,
                "rag_cluster_target": rag_cluster_target,
            }
        except Exception as exc:
            job["status"] = "error"
            job["result"] = {"error": str(exc)}
            log_lines.append(f"❌ 错误：{exc}")

    threading.Thread(target=_run, args=(job_id,), daemon=True).start()
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "style": style,
        "rag_assist": rag_assist,
        "rag_cluster_target": rag_cluster_target,
    })


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
        from seismo_skill.knowledge_indexer import KnowledgeIndexer

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
        doc_ids_to_remove = set()
        if kb:
            for doc in list(kb.list_docs()):
                folder = getattr(doc, "proj_folder", "") or ""
                if _proj_match(folder, proj_candidates):
                    doc_ids_to_remove.add(doc.doc_id)

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
                if doc_id:
                    doc_ids_to_remove.add(doc_id)
                if entry.get("skill_name"):
                    skill_names_to_del.add(entry.get("skill_name"))
                keys_to_del.append(rel)

        removed_docs = _bulk_delete_kb_docs(kb, doc_ids_to_remove)

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

        skill_deleted = False
        for skill_name in skill_names_to_del:
            skill_deleted = _delete_generated_skill_by_name(skill_name) or skill_deleted
        artifacts_deleted = _cleanup_generated_skill_artifacts_for_project(proj_name, skill_names_to_del)
        skill_deleted = skill_deleted or artifacts_deleted > 0

        return jsonify({
            "ok": True,
            "proj_name": proj_name,
            "removed_files": len(keys_to_del),
            "removed_docs": removed_docs,
            "skill_deleted": skill_deleted,
            "artifacts_deleted": artifacts_deleted,
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
