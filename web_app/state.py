"""
state.py — 全局共享状态（锁、字典、路径常量）

所有路由模块从此处导入，避免循环依赖。
"""
import threading
from pathlib import Path

# ── 锁 ───────────────────────────────────────────────────────────────────────
# RLock 允许同一线程重入，防止 _get_code_engine 在 _run() 内嵌套调用时死锁
_code_engine_lock = threading.RLock()

# ── 任务状态 ─────────────────────────────────────────────────────────────────
tasks: dict = {}                # 传统任务: task_id → state
_pull_status: dict = {}         # Ollama pull 进度
_code_engines: dict = {}        # session_id → CodeEngine
_code_jobs: dict = {}           # job_id → {status, progress, result, ts}
_kb_dir_jobs: dict = {}         # job_id → {status, log, ...}
_ref_kb_jobs: dict = {}         # job_id → {status, log, ...}
_geo_agent_jobs: dict = {}      # job_id → {status, progress, result, error, ts}
_lit_jobs: dict = {}            # job_id → {status, progress, result, error, ts}
_chat_jobs: dict = {}           # job_id → {status, answer, sources, error, ts}
_session_docs: dict = {}        # session_id → {chunks, doc_names}

# ── Knowledge 目录扫描缓存 ────────────────────────────────────────────────────
_kb_dir_status: dict = {
    "checked": False,
    "pending_total": 0,
    "pending_new": 0,
    "pending_modified": 0,
    "pending_retry": 0,
    "indexed": 0,
    "failed": 0,
    "knowledge_dir": "",
    "error": "",
}
_ref_kb_dir_status: dict = {
    "checked": False, "projects": [], "total_files": 0,
    "pending_total": 0, "indexed": 0, "failed": 0,
    "knowledge_dir": "", "error": "",
}

# ── 路径常量 ─────────────────────────────────────────────────────────────────
_WEB_APP_DIR = Path(__file__).parent
_PROJECT_ROOT = _WEB_APP_DIR.parent

UPLOAD_FOLDER_CHAT = _WEB_APP_DIR / "uploads" / "chat_pdfs"
UPLOAD_FOLDER_CHAT.mkdir(parents=True, exist_ok=True)

GEO_WORKSPACE_ROOT = _WEB_APP_DIR / "uploads" / "geo_workspaces"
GEO_WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

_REF_KNOWLEDGE_DIR = _PROJECT_ROOT / "seismo_knowledge"
_REF_KB_MANIFEST_DIR = Path.home() / ".seismicx" / "ref_knowledge"
