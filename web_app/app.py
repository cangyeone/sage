#!/usr/bin/env python3
"""
SeismicX Web Interface — app.py (slim entry point)

路由已按领域拆分到 routes/ 子目录（Flask Blueprints）：
  routes/pages.py      — HTML 页面
  routes/llm.py        — /api/llm/*
  routes/workspace.py  — /api/workspace/*
  routes/skills.py     — /api/skills/*, /api/workflows/*
  routes/knowledge.py  — /api/knowledge/*, /api/ref_knowledge/*
  routes/code.py       — /api/chat/code/*, /api/chat/workflow, /api/chat/route
  routes/chat.py       — /api/chat, tasks, literature, geo agent, RAG

共享状态 → state.py
共享工具函数 → helpers.py

Usage:
    python web_app/app.py [--port PORT] [--host HOST] [--debug]
"""

from __future__ import annotations

# ── macOS fork safety — MUST be before numpy/scipy/obspy/torch imports ────────
# subprocess.run() uses fork() on macOS; if BLAS has multi-threaded already,
# the forked child inherits broken thread state → SIGSEGV in the parent process.
# Setting thread counts to 1 here (before numpy loads) prevents BLAS from
# creating multiple threads, making fork() safe.
import os as _os_fork_fix
_os_fork_fix.environ.setdefault('OMP_NUM_THREADS',           '1')
_os_fork_fix.environ.setdefault('OPENBLAS_NUM_THREADS',      '1')
_os_fork_fix.environ.setdefault('MKL_NUM_THREADS',           '1')
_os_fork_fix.environ.setdefault('VECLIB_MAXIMUM_THREADS',    '1')
_os_fork_fix.environ.setdefault('NUMEXPR_NUM_THREADS',       '1')
_os_fork_fix.environ.setdefault('KMP_DUPLICATE_LIB_OK',      'TRUE')
_os_fork_fix.environ.setdefault('OBJC_DISABLE_INITIALIZE_FORK_SAFETY', 'YES')
del _os_fork_fix
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import threading
from pathlib import Path

# ── sys.path 设置（须在任何内部 import 之前）────────────────────────────────
_WEB_APP_DIR   = Path(__file__).parent
_PROJECT_ROOT  = _WEB_APP_DIR.parent
sys.path.insert(0, str(_WEB_APP_DIR))    # 使 rag_engine / simple_rag 等可直接 import
sys.path.insert(0, str(_PROJECT_ROOT))   # 使 seismo_code / seismo_skill 等可 import

from flask import Flask
# Pre-load PyYAML C extension in the main thread to prevent ObjC SIGSEGV in worker threads
try:
    import yaml as _yaml_preload
    _yaml_preload.safe_load("")   # triggers C extension init here, not in a worker thread
    del _yaml_preload
except Exception:
    pass

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER']       = str(_WEB_APP_DIR / 'uploads')
app.config['OUTPUT_FOLDER']       = str(_WEB_APP_DIR / 'outputs')
app.config['MAX_CONTENT_LENGTH']  = 500 * 1024 * 1024   # 500 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# ── 注册蓝图 ─────────────────────────────────────────────────────────────────
from routes import register_blueprints
register_blueprints(app)

# ── 启动时后台任务 ────────────────────────────────────────────────────────────

def _startup_scan_knowledge_dir():
    """后台扫描 seismo_skill/docs/ 目录，更新 _kb_dir_status 缓存。"""
    try:
        from state import _code_engine_lock, _kb_dir_status
        with _code_engine_lock:
            from seismo_skill.knowledge_indexer import KnowledgeIndexer
            indexer = KnowledgeIndexer()
            summary = indexer.manifest_summary()
            _kb_dir_status.update(summary)
            _kb_dir_status["checked"] = True
    except Exception as e:
        try:
            from state import _kb_dir_status
            _kb_dir_status["checked"] = True
            _kb_dir_status["error"]   = str(e)
        except Exception:
            pass


def _startup_init_code_engine():
    """在主线程预热 CodeEngine，避免 sentence-transformers 在请求线程首次加载的竞态。"""
    try:
        from state import _code_engine_lock, _code_engines
        with _code_engine_lock:
            from config_manager import get_config_manager
            llm_cfg = get_config_manager().get_llm_config()
            if llm_cfg.get('api_base'):
                from seismo_code.code_engine import CodeEngine
                if 'default' not in _code_engines:
                    _code_engines['default'] = CodeEngine(
                        llm_cfg, project_root=str(_PROJECT_ROOT)
                    )
    except Exception:
        pass   # LLM 尚未配置，静默跳过


# 同步预热（在接受请求前完成）
_startup_init_code_engine()

# 异步扫描文档目录（不阻塞启动）
threading.Thread(target=_startup_scan_knowledge_dir, daemon=True).start()


# ── 主入口 ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SeismicX Web Interface')
    parser.add_argument('--port',  type=int,  default=5010)
    parser.add_argument('--host',  type=str,  default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("SeismicX Web Interface")
    print("=" * 70)
    print(f"\nStarting on  http://{args.host}:{args.port}")
    print(f"  Chat:       http://localhost:{args.port}/chat")
    print(f"  Knowledge:  http://localhost:{args.port}/knowledge")
    print(f"  Skills:     http://localhost:{args.port}/skills")
    print(f"  LLM:        http://localhost:{args.port}/llm-settings")
    print("\nPress Ctrl+C to stop\n")

    # threaded=False + use_reloader=False：避免 sentence-transformers 在多线程/多进程
    # 模式下触发 SIGSEGV（C 扩展库的线程安全限制）
    app.run(
        host=args.host, port=args.port,
        debug=args.debug,
        threaded=False,
        use_reloader=False,
    )
