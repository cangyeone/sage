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

USER_PROFILE_MD = _PROJECT_ROOT / "seismo_rag" / "user_profile.md"


def get_user_profile_context(max_chars: int = 3000) -> str:
    """Read the local long-term user profile as soft context."""
    try:
        if USER_PROFILE_MD.exists():
            return USER_PROFILE_MD.read_text(encoding="utf-8")[-max_chars:]
    except Exception:
        pass
    return ""


def append_user_profile_to_system(system: str, max_chars: int = 3000) -> str:
    """Append the user profile to an LLM system prompt when available."""
    profile = get_user_profile_context(max_chars=max_chars)
    if not profile:
        return system
    return (
        system
        + "\n\n===== Long-term user profile (local Markdown memory) =====\n"
        + "Use this only as soft context for estimating the user's background, habits, "
        + "preferred depth, knowledge level, and likely intent. Do not mention it unless useful.\n"
        + profile
    )


def path_is_within_root(path: str | Path, root: str | Path) -> bool:
    """Return True only when path resolves inside root."""
    try:
        root_path = Path(root).expanduser().resolve(strict=False)
        req_path = Path(path).expanduser().resolve(strict=False)
        req_path.relative_to(root_path)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def safe_child_path(base: str | Path, child_name: str) -> Path:
    """Build a path under base and reject traversal/symlink escape attempts."""
    base_resolved = Path(base).resolve()
    candidate = base_resolved / str(child_name or "")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError("Path escapes allowed directory") from exc
    return resolved


def resolve_coding_project_root(project_path: str | None = None) -> Path:
    """Resolve a user-selected coding workspace, falling back to the SAGE repo."""
    raw = str(project_path or "").strip()
    if not raw:
        return _PROJECT_ROOT.resolve()
    root = Path(raw).expanduser().resolve(strict=False)
    if not root.exists():
        raise ValueError(f"项目路径不存在: {root}")
    if not root.is_dir():
        raise ValueError(f"项目路径不是目录: {root}")
    return root


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm_config() -> dict:
    """统一获取 LLM 配置（每次重新读取以反映最新设置）。"""
    try:
        from config_manager import LLMConfigManager
        manager = LLMConfigManager()
        raw = manager.config
        active = raw.get("active_backend")

        if active == "online" and raw.get("online"):
            cfg = raw.get("online", {})
            return {
                "provider": cfg.get("provider", "custom"),
                "model": cfg.get("model", ""),
                "api_base": cfg.get("api_base", ""),
                "api_key": cfg.get("api_key", ""),
                "temperature": raw.get("llm", {}).get("temperature", 0.6),
                "max_tokens": raw.get("llm", {}).get("max_tokens", 2000),
            }
        if active == "vllm" and raw.get("vllm"):
            cfg = raw.get("vllm", {})
            port = cfg.get("port", 8001)
            return {
                "provider": "openai",
                "model": cfg.get("model", ""),
                "api_base": cfg.get("api_base") or f"http://localhost:{port}/v1",
                "api_key": cfg.get("api_key", ""),
                "temperature": raw.get("llm", {}).get("temperature", 0.6),
                "max_tokens": raw.get("llm", {}).get("max_tokens", 2000),
            }
        if active == "ollama" and raw.get("ollama"):
            cfg = raw.get("ollama", {})
            return {
                "provider": "ollama",
                "model": cfg.get("model", ""),
                "api_base": cfg.get("api_base", "http://localhost:11434"),
                "api_key": "",
                "temperature": raw.get("llm", {}).get("temperature", 0.6),
                "max_tokens": raw.get("llm", {}).get("max_tokens", 2000),
            }

        return manager.get_llm_config()
    except Exception:
        return {}


def _strip_data_url(img: str) -> str:
    """Remove 'data:image/...;base64,' prefix, returning raw base64."""
    if img.startswith("data:") and "," in img:
        return img.split(",", 1)[1]
    return img


def _inject_images_into_messages(messages: list, images: list, provider: str) -> list:
    """
    Return a shallow copy of messages with images embedded in the last user turn.
    Ollama: adds {"images": [raw_b64, ...]} field to the user message dict.
    OpenAI-compat: converts user content to the [{type:text},{type:image_url},...] array.
    """
    if not images:
        return messages

    msgs = [dict(m) for m in messages]   # shallow copy each message
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            if provider == "ollama":
                msgs[i]["images"] = [_strip_data_url(img) for img in images]
            else:
                text = msgs[i].get("content", "")
                content_arr: list = [{"type": "text", "text": text}]
                for img in images:
                    data_url = (img if img.startswith("data:")
                                else f"data:image/jpeg;base64,{img}")
                    content_arr.append(
                        {"type": "image_url", "image_url": {"url": data_url}}
                    )
                msgs[i]["content"] = content_arr
            break
    return msgs


def llm_call(messages: list, llm_cfg: dict, max_tokens: int = 2000,
             images: list = None) -> str:
    """
    向 LLM 发请求，返回回复文本；失败时抛出异常。
    images: 可选，base64 字符串列表（可含 data URL 前缀），用于多模态 VL 模型。
    """
    import urllib.request
    import json as _json

    provider = (llm_cfg.get("provider", "ollama") or "ollama").lower()
    model    = llm_cfg.get("model", "")
    api_base = llm_cfg.get("api_base", "")
    api_key  = llm_cfg.get("api_key", "")
    temperature = llm_cfg.get("temperature", 0.6)

    if not api_base:
        raise ValueError("未配置 LLM 后端地址，请在 LLM 设置页面中选择模型")
    if not model:
        raise ValueError("未选择模型，请在 LLM 设置页面中选择一个 Ollama 模型")

    msgs = _inject_images_into_messages(messages, images or [], provider)

    if provider == "ollama":
        endpoint = api_base.rstrip("/") + "/api/chat"
        payload  = {"model": model, "messages": msgs, "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}}
    else:
        endpoint = api_base.rstrip("/") + "/chat/completions"
        payload  = {"model": model, "messages": msgs,
                    "temperature": temperature, "max_tokens": max_tokens}

    data    = _json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"}
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = _json.loads(resp.read().decode())

    if provider == "ollama":
        return body.get("message", {}).get("content", "").strip()
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


def llm_stream(messages: list, llm_cfg: dict, max_tokens: int = 2000,
               images: list = None):
    """
    Generator that yields text chunks from the LLM stream.
    Supports Ollama (plain NDJSON stream) and OpenAI-compatible SSE.
    images: 可选，base64 字符串列表，用于多模态 VL 模型。
    """
    import urllib.request
    import json as _json

    provider = (llm_cfg.get("provider", "ollama") or "ollama").lower()
    model    = llm_cfg.get("model", "")
    api_base = llm_cfg.get("api_base", "")
    api_key  = llm_cfg.get("api_key", "")
    temperature = llm_cfg.get("temperature", 0.6)

    if not api_base:
        raise ValueError("未配置 LLM 后端地址")
    if not model:
        raise ValueError("未选择模型")

    msgs = _inject_images_into_messages(messages, images or [], provider)

    if provider == "ollama":
        url     = api_base.rstrip("/") + "/api/chat"
        payload = {"model": model, "messages": msgs, "stream": True,
                   "options": {"temperature": temperature, "num_predict": max_tokens}}
    else:
        url     = api_base.rstrip("/") + "/chat/completions"
        payload = {"model": model, "messages": msgs, "stream": True,
                   "temperature": temperature, "max_tokens": max_tokens}

    data    = _json.dumps(payload).encode()
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}" if api_key else "Bearer none"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=120) as resp:
        in_reasoning = False
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            # OpenAI-compat SSE: "data: {...}" or "data: [DONE]"
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    if in_reasoning:
                        yield "</think>"
                    return
                try:
                    obj = _json.loads(line)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("reasoning_text")
                        or ""
                    )
                    chunk = delta.get("content", "")
                    if reasoning:
                        if not in_reasoning:
                            yield "<think>"
                            in_reasoning = True
                        yield reasoning
                    if chunk:
                        if in_reasoning:
                            yield "</think>"
                            in_reasoning = False
                        yield chunk
                except Exception:
                    continue
            else:
                # Ollama plain NDJSON (no "data: " prefix)
                try:
                    obj = _json.loads(line)
                    msg = obj.get("message", {})
                    reasoning = msg.get("thinking", "") or msg.get("reasoning", "")
                    chunk = msg.get("content", "")
                    if reasoning:
                        if not in_reasoning:
                            yield "<think>"
                            in_reasoning = True
                        yield reasoning
                    if chunk:
                        if in_reasoning:
                            yield "</think>"
                            in_reasoning = False
                        yield chunk
                    if obj.get("done"):
                        if in_reasoning:
                            yield "</think>"
                        return
                except Exception:
                    continue


# ── Workspace ─────────────────────────────────────────────────────────────────

import re as _re

def get_workspace_config() -> dict:
    try:
        from config_manager import LLMConfigManager
        paths_cfg = LLMConfigManager().get_app_paths()
        ws = {
            'enabled': bool(paths_cfg.get('chat_workspace_enabled', False)),
            'paths': paths_cfg.get('chat_workspace_paths') or [],
            'path': '',
        }
        path = ws.get('path', '')
        paths = ws.get('paths') or []
        if isinstance(paths, str):
            paths = [p.strip() for p in _re.split(r'[\n,;]+', paths) if p.strip()]
        if path and path not in paths:
            paths.insert(0, path)
        ws['paths'] = paths
        ws['path'] = path or (paths[0] if paths else '')
        return ws
    except Exception:
        return {'enabled': False, 'path': '', 'paths': []}


def _normalise_workspace_paths(path_or_paths) -> list:
    if isinstance(path_or_paths, (list, tuple)):
        raw = []
        for item in path_or_paths:
            raw.extend(_re.split(r'[\n,;]+', str(item or '')))
    else:
        raw = _re.split(r'[\n,;]+', str(path_or_paths or ''))
    paths = []
    for p in raw:
        p = p.strip()
        if p and p not in paths:
            paths.append(p)
    return paths


def save_workspace_config(enabled: bool, path: str = '', paths=None):
    from config_manager import LLMConfigManager
    cfg = LLMConfigManager()
    all_paths = _normalise_workspace_paths(paths if paths is not None else path)
    if path and path not in all_paths:
        all_paths.insert(0, path)
    project_cfg = cfg.get_project_config()
    app_paths = project_cfg.setdefault('app_paths', cfg.get_app_paths())
    app_paths['chat_workspace_enabled'] = enabled
    app_paths['chat_workspace_paths'] = all_paths
    cfg.save_project_config(project_cfg)


def inject_workspace_context(message: str, workspace_path: str) -> str:
    """If message mentions a path and workspace is enabled, inject directory listing."""
    ws = get_workspace_config()
    if not ws.get('enabled'):
        return ''

    roots = _normalise_workspace_paths(workspace_path) or ws.get('paths') or [ws.get('path', '')]
    roots = [os.path.realpath(os.path.expanduser(r)) for r in roots if r]
    paths_found = _re.findall(r'[~/][\w./\-]+', message)
    context_parts = []

    for p in paths_found:
        p_exp = os.path.expanduser(p)
        candidate_paths = [os.path.realpath(p_exp)] if p_exp.startswith('/') else [
            os.path.realpath(os.path.join(root, p_exp)) for root in roots
        ]
        p_abs = ''
        for cand in candidate_paths:
            if any(os.path.commonpath([cand, root]) == root for root in roots):
                p_abs = cand
                break
        if not p_abs:
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


def get_code_engine(session_id: str, llm_cfg: dict, project_root: str | Path | None = None):
    """获取或创建 session/workspace 级别的 CodeEngine（在 _code_engine_lock 内调用）。"""
    sage_proj = str(_PROJECT_ROOT)
    if sage_proj not in sys.path:
        sys.path.insert(0, sage_proj)
    proj = str(resolve_coding_project_root(str(project_root or "")))
    engine_key = f"{session_id}::{proj}"
    with _code_engine_lock:
        from seismo_code.code_engine import CodeEngine
        if engine_key not in _code_engines:
            _code_engines[engine_key] = CodeEngine(llm_cfg, project_root=proj)
        else:
            _code_engines[engine_key].llm_config = llm_cfg
            _code_engines[engine_key].project_root = proj
        return _code_engines[engine_key]


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
    html_previews = []
    seen = set()
    _MIME = {'.py': 'text/x-python', '.sh': 'text/x-shellscript',
             '.txt': 'text/plain', '.png': 'image/png',
             '.svg': 'image/svg+xml', '.pdf': 'application/pdf',
             '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
             '.csv': 'text/csv', '.dat': 'text/plain',
             '.html': 'text/html', '.htm': 'text/html'}
    _HTML_PREVIEW_LIMIT = 15 * 1024 * 1024

    def _inline_plotly_if_needed(path, raw):
        if Path(path).suffix.lower() not in ('.html', '.htm') or b'cdn.plot.ly' not in raw:
            return raw
        try:
            import re as _re
            import plotly as _plotly
            plotly_js = Path(_plotly.__file__).parent / 'package_data' / 'plotly.min.js'
            if not plotly_js.is_file():
                return raw
            text = raw.decode('utf-8')
            inline_script = '<script type="text/javascript">' + plotly_js.read_text(encoding='utf-8') + '</script>'
            new_text = _re.sub(
                r'<script[^>]+src=["\']https?://cdn\.plot\.ly/plotly-[^"\']+\.min\.js["\'][^>]*>\s*</script>',
                lambda _m: inline_script,
                text,
                count=1,
            )
            return new_text.encode('utf-8') if new_text != text else raw
        except Exception:
            return raw

    def _add(path, mime):
        rp = os.path.realpath(path)
        if rp in seen or not os.path.isfile(rp):
            return
        try:
            with open(rp, 'rb') as _f:
                raw = _inline_plotly_if_needed(rp, _f.read())
                encoded = _b64.b64encode(raw).decode('utf-8')
                downloads.append({'name': Path(rp).name,
                                  'data': encoded,
                                  'mimetype': mime})
                if Path(rp).suffix.lower() in ('.html', '.htm') and len(raw) <= _HTML_PREVIEW_LIMIT:
                    html_previews.append({'name': Path(rp).name,
                                          'data': encoded,
                                          'mimetype': 'text/html'})
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
        'html_previews': html_previews,
        'artifact_paths': list(dict.fromkeys(
            [p for p in (
                [result.script_path] +
                list(result.figures or []) +
                list(result.output_files or [])
            ) if p]
        )),
        'exec_dir': getattr(result.exec_result, 'exec_dir', '') if result.exec_result else '',
    }
