"""
helpers.py — 各路由模块共用的辅助函数

不包含路由装饰器，可被任意 Blueprint 直接 import。
"""
from __future__ import annotations

import os
import sys
import time as _time
import uuid as _uuid
from pathlib import Path
from typing import Optional

from state import (
    _code_engine_lock, _code_engines, _code_jobs,
    _PROJECT_ROOT, UPLOAD_FOLDER_CHAT,
)


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm_config() -> dict:
    """统一获取 LLM 配置（每次重新读取以反映最新设置）。"""
    try:
        from config_manager import LLMConfigManager
        return LLMConfigManager().get_llm_config()
    except Exception:
        return {}


def llm_call(messages: list, llm_cfg: dict, max_tokens: int = 2000) -> str:
    """向 LLM 发请求，返回回复文本；失败时抛出异常。"""
    import urllib.request
    import json as _json

    provider = llm_cfg.get("provider", "ollama")
    model    = llm_cfg.get("model", "")
    api_base = llm_cfg.get("api_base", "")
    api_key  = llm_cfg.get("api_key", "")

    if not api_base:
        raise ValueError("未配置 LLM 后端地址，请在 LLM 设置页面中选择模型")
    if not model:
        raise ValueError("未选择模型，请在 LLM 设置页面中选择一个 Ollama 模型")

    if provider == "ollama":
        url     = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": False,
                   "options": {"temperature": 0.6, "num_predict": max_tokens}}
    else:
        url     = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages,
                   "temperature": 0.6, "max_tokens": max_tokens}

    data    = _json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = _json.loads(resp.read().decode())

    if provider == "ollama":
        return body.get("message", {}).get("content", "").strip()
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def llm_stream(messages: list, llm_cfg: dict, max_tokens: int = 2000):
    """
    Generator that yields text chunks from the LLM stream.
    Supports Ollama (plain NDJSON stream) and OpenAI-compatible SSE.
    """
    import urllib.request
    import json as _json

    provider = llm_cfg.get("provider", "ollama")
    model    = llm_cfg.get("model", "")
    api_base = llm_cfg.get("api_base", "")
    api_key  = llm_cfg.get("api_key", "")

    if not api_base:
        raise ValueError("未配置 LLM 后端地址")
    if not model:
        raise ValueError("未选择模型")

    if provider == "ollama":
        url     = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"temperature": 0.6, "num_predict": max_tokens}}
    else:
        url     = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": messages, "stream": True,
                   "temperature": 0.6, "max_tokens": max_tokens}

    data    = _json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # OpenAI-compat SSE: "data: {...}" or "data: [DONE]"
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    return
                try:
                    obj   = _json.loads(line)
                    chunk = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if chunk:
                        yield chunk
                except Exception:
                    continue
            else:
                # Ollama plain NDJSON (no "data: " prefix)
                try:
                    obj   = _json.loads(line)
                    chunk = obj.get("message", {}).get("content", "")
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        return
                except Exception:
                    continue


# ── Workspace ─────────────────────────────────────────────────────────────────

import re as _re

def get_workspace_config() -> dict:
    try:
        from config_manager import LLMConfigManager
        return LLMConfigManager().config.get('workspace', {'enabled': False, 'path': ''})
    except Exception:
        return {'enabled': False, 'path': ''}


def save_workspace_config(enabled: bool, path: str):
    from config_manager import LLMConfigManager
    cfg = LLMConfigManager()
    cfg.config['workspace'] = {'enabled': enabled, 'path': path}
    cfg.save_config()


def inject_workspace_context(message: str, workspace_path: str) -> str:
    """If message mentions a path and workspace is enabled, inject directory listing."""
    if not workspace_path:
        return ''
    ws = get_workspace_config()
    if not ws.get('enabled'):
        return ''

    root     = os.path.expanduser(ws.get('path', ''))
    abs_root = os.path.realpath(root)
    paths_found = _re.findall(r'[~/][\w./\-]+', message)
    context_parts = []

    for p in paths_found:
        p_exp = os.path.expanduser(p)
        p_abs = (os.path.realpath(p_exp) if p_exp.startswith('/')
                 else os.path.realpath(os.path.join(abs_root, p_exp)))
        if not p_abs.startswith(abs_root):
            continue
        if os.path.isdir(p_abs):
            try:
                names = sorted(os.listdir(p_abs))
                lines = []
                for n in names[:60]:
                    full = os.path.join(p_abs, n)
                    lines.append(f'  {n}{"/" if os.path.isdir(full) else ""}')
                context_parts.append(f"目录 {p_abs} 内容（共 {len(names)} 项）：\n" + '\n'.join(lines))
            except Exception:
                pass
        elif os.path.isfile(p_abs):
            context_parts.append(f"文件 {p_abs} 存在（大小：{os.path.getsize(p_abs)} 字节）")

    return '\n\n'.join(context_parts)


# ── Module loaders ────────────────────────────────────────────────────────────

def get_skill_loader():
    try:
        proj = str(_PROJECT_ROOT)
        if proj not in sys.path:
            sys.path.insert(0, proj)
        import seismo_skill as _sl
        return _sl
    except Exception:
        return None


def get_workflow_runner():
    try:
        proj = str(_PROJECT_ROOT)
        if proj not in sys.path:
            sys.path.insert(0, proj)
        import seismo_skill.workflow_runner as _wr
        return _wr
    except Exception:
        return None


def get_kb_instance():
    try:
        from rag_engine import get_knowledge_base
        return get_knowledge_base()
    except Exception:
        return None


def get_ref_indexer():
    from state import _REF_KNOWLEDGE_DIR, _REF_KB_MANIFEST_DIR
    proj = str(_PROJECT_ROOT)
    if proj not in sys.path:
        sys.path.insert(0, proj)
    from seismo_skill.knowledge_indexer import KnowledgeIndexer
    _REF_KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    return KnowledgeIndexer(
        knowledge_dir=_REF_KNOWLEDGE_DIR,
        manifest_dir=_REF_KB_MANIFEST_DIR,
    )


def get_code_engine(session_id: str, llm_cfg: dict):
    """获取或创建 session 级别的 CodeEngine（在 _code_engine_lock 内调用）。"""
    proj = str(_PROJECT_ROOT)
    if proj not in sys.path:
        sys.path.insert(0, proj)
    with _code_engine_lock:
        from seismo_code.code_engine import CodeEngine
        if session_id not in _code_engines:
            _code_engines[session_id] = CodeEngine(llm_cfg, project_root=proj)
        else:
            _code_engines[session_id].llm_config = llm_cfg
        return _code_engines[session_id]


def gc_code_jobs():
    cutoff = _time.time() - 600
    stale = [k for k, v in _code_jobs.items() if v.get('ts', 0) < cutoff]
    for k in stale:
        _code_jobs.pop(k, None)


def serialize_code_result(result, skill_used: str) -> dict:
    """Serialize a CodeRunResult into the JSON payload the frontend expects."""
    import base64 as _b64

    gmt_script_map: dict = {}
    for line in (result.stdout or '').splitlines():
        if line.startswith('[GMT_SCRIPT] '):
            sp = line[len('[GMT_SCRIPT] '):].strip()
            if os.path.isfile(sp):
                try:
                    with open(sp, encoding='utf-8') as sf:
                        base = Path(sp).stem
                        gmt_script_map[base] = {'name': Path(sp).name, 'content': sf.read()}
                except Exception:
                    pass

    figure_paths = list(result.figures) if result.figures else []
    for out_path in (result.output_files or []):
        if os.path.splitext(out_path)[1].lower() in ('.png', '.svg', '.pdf') \
                and out_path not in figure_paths:
            figure_paths.append(out_path)

    figures = []
    for fig_path in figure_paths:
        try:
            with open(fig_path, 'rb') as f:
                fig_base = Path(fig_path).stem
                entry = {'name': Path(fig_path).name,
                         'data': _b64.b64encode(f.read()).decode('utf-8')}
                if fig_base in gmt_script_map:
                    entry['gmt_script'] = gmt_script_map[fig_base]
                figures.append(entry)
        except Exception:
            pass

    debug_trace = [
        {'attempt': d.attempt, 'diagnosis': d.diagnosis,
         'success': d.success, 'error': (d.error or '')[-400:]}
        for d in (result.debug_trace or [])
    ]

    downloads = []
    seen = set()
    _MIME = {'.py': 'text/x-python', '.sh': 'text/x-shellscript',
             '.txt': 'text/plain', '.png': 'image/png',
             '.svg': 'image/svg+xml', '.pdf': 'application/pdf',
             '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
             '.csv': 'text/csv', '.dat': 'text/plain'}

    def _add(path, mime):
        rp = os.path.realpath(path)
        if rp in seen or not os.path.isfile(rp):
            return
        try:
            with open(rp, 'rb') as _f:
                downloads.append({'name': Path(rp).name,
                                  'data': _b64.b64encode(_f.read()).decode('utf-8'),
                                  'mimetype': mime})
            seen.add(rp)
        except Exception:
            pass

    if result.script_path:
        _add(result.script_path, 'text/x-python')
    for p in (result.output_files or []):
        _add(p, _MIME.get(Path(p).suffix.lower(), 'application/octet-stream'))
    for p in figure_paths:
        _add(p, _MIME.get(Path(p).suffix.lower(), 'image/png'))

    script_b64 = next((d['data'] for d in downloads if d['name'].endswith('.py')), '')

    return {
        'ok':          True,
        'success':     result.success,
        'response':    result.response,
        'code':        result.code,
        'stdout':      result.stdout,
        'figures':     figures,
        'skill_used':  skill_used,
        'attempts':    result.attempts,
        'debug_trace': debug_trace,
        'plan':        result.plan,
        'script_b64':  script_b64,
        'downloads':   downloads,
    }
