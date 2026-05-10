"""LLM 配置路由"""
from flask import Blueprint, request, jsonify
import sys
import json
import threading
import time as _time
import urllib.request as _ur
from pathlib import Path
from state import _pull_status

bp = Blueprint('llm', __name__)

try:
    from backend_manager import ONLINE_PROVIDERS
except Exception:
    ONLINE_PROVIDERS = {
        "openai": {"display": "OpenAI", "api_base": "https://api.openai.com/v1", "default_model": "gpt-4o-mini", "need_key": True},
        "deepseek": {"display": "DeepSeek", "api_base": "https://api.deepseek.com/v1", "default_model": "deepseek-v4-flash", "need_key": True},
        "siliconflow": {"display": "SiliconFlow", "api_base": "https://api.siliconflow.cn/v1", "default_model": "Qwen/Qwen2.5-7B-Instruct", "need_key": True},
        "moonshot": {"display": "Moonshot / Kimi", "api_base": "https://api.moonshot.cn/v1", "default_model": "moonshot-v1-8k", "need_key": True},
        "dashscope": {"display": "DashScope", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "default_model": "qwen-turbo", "need_key": True},
        "zhipu": {"display": "Zhipu AI", "api_base": "https://open.bigmodel.cn/api/paas/v4", "default_model": "glm-4-flash", "need_key": True},
        "custom": {"display": "Custom OpenAI-compatible", "api_base": "", "default_model": "", "need_key": True},
    }


def _public_online_providers():
    """Provider metadata safe to send to the browser."""
    return {
        key: {
            'display': value.get('display', key),
            'api_base': value.get('api_base', ''),
            'default_model': value.get('default_model', ''),
            'need_key': value.get('need_key', True),
        }
        for key, value in ONLINE_PROVIDERS.items()
        if key != 'anthropic'
    }


def _mask_api_key(config):
    config = dict(config or {})
    if config.get('api_key'):
        key = config['api_key']
        config['api_key_masked'] = '****' + key[-4:] if len(key) > 4 else '****'
        config['api_key'] = ''
    return config


def _mask_key(value):
    if not value:
        return ''
    value = str(value)
    return '****' + value[-4:] if len(value) > 4 else '****'


def _public_search_config(raw_search):
    from config_manager import SEARCH_PROVIDER_DEFAULTS
    raw_search = raw_search or {}
    providers = raw_search.get('providers') or {}
    out = {}
    for key, meta in SEARCH_PROVIDER_DEFAULTS.items():
        cfg = providers.get(key) or {}
        out[key] = {
            'display': meta.get('display', key),
            'description': meta.get('description', ''),
            'requires_key': bool(meta.get('requires_key')),
            'enabled': bool(cfg.get('enabled', meta.get('enabled', False))),
            'api_key_masked': _mask_key(cfg.get('api_key', '')),
            'api_base': cfg.get('api_base', ''),
        }
    return {
        'providers': out,
        'default_sources': raw_search.get('default_sources') or ['openalex', 'semantic_scholar'],
    }


def _parse_model_list(data):
    models = []
    if isinstance(data, dict) and isinstance(data.get('data'), list):
        models = [m['id'] for m in data['data'] if isinstance(m, dict) and m.get('id')]
    elif isinstance(data, dict) and isinstance(data.get('models'), list):
        models = [m.get('id') or m.get('name', '') for m in data['models'] if isinstance(m, dict)]
    elif isinstance(data, list):
        models = [
            m.get('id') or m.get('name', '')
            for m in data
            if isinstance(m, dict)
        ]
    return sorted({m for m in models if m})


def _fallback_model_candidates(provider, api_base, current_model):
    candidates = []
    if current_model:
        candidates.append(current_model)
    preset = ONLINE_PROVIDERS.get(provider, {})
    if preset.get('default_model'):
        candidates.append(preset['default_model'])
    base_l = (api_base or '').lower()
    if 'aliyuncs.com' in base_l or 'dashscope' in base_l:
        candidates.extend(['qwen3.6-plus', 'qwen-plus', 'qwen-turbo'])
    return list(dict.fromkeys(candidates))


def _chat_model_available(api_base, api_key, model):
    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_tokens': 1,
        'temperature': 0,
    }).encode('utf-8')
    req = _ur.Request(
        f'{api_base.rstrip("/")}/chat/completions',
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    with _ur.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return bool(data.get('choices'))


@bp.route('/api/llm/config', methods=['GET'])
def llm_config_get():
    """Get current LLM configuration (reads directly from config.json)"""
    from config_manager import LLMConfigManager
    cfg_mgr = LLMConfigManager()
    raw_config = cfg_mgr.config
    llm_config = dict(raw_config.get('llm', {}))

    if raw_config.get('active_backend') == 'online' and raw_config.get('online'):
        online = raw_config.get('online', {})
        llm_config.update({
            'provider': online.get('provider', llm_config.get('provider', 'custom')),
            'model': online.get('model', llm_config.get('model', '')),
            'api_base': online.get('api_base', llm_config.get('api_base', '')),
            'api_key': online.get('api_key', llm_config.get('api_key', '')),
        })
    elif raw_config.get('active_backend') == 'ollama' and raw_config.get('ollama'):
        ollama = raw_config.get('ollama', {})
        llm_config.update({
            'provider': 'ollama',
            'model': ollama.get('model', llm_config.get('model', '')),
            'api_base': ollama.get('api_base', llm_config.get('api_base', 'http://localhost:11434')),
            'api_key': '',
        })

    return jsonify({
        'config': _mask_api_key(llm_config),
        'search': _public_search_config(cfg_mgr.get_search_config()),
        'app_paths': cfg_mgr.get_app_paths(),
        'coding_agent': cfg_mgr.get_coding_agent_config(),
        'online_providers': _public_online_providers(),
        'first_run': cfg_mgr.is_first_run(),
        'ollama_available': cfg_mgr.check_ollama_available()
    })


@bp.route('/api/llm/config', methods=['POST'])
def update_llm_config():
    """Update LLM configuration"""
    from config_manager import get_config_manager
    data = request.get_json(silent=True) or {}
    config = get_config_manager()

    try:
        cfg = config.config
        llm = cfg.setdefault('llm', {})
        provider = (data.get('provider') or llm.get('provider') or 'ollama').strip()
        if provider not in ['ollama', 'openai', 'deepseek', 'siliconflow', 'moonshot', 'dashscope', 'zhipu', 'custom']:
            raise ValueError(f'Invalid provider: {provider}')

        if provider == 'ollama':
            api_base = (data.get('api_base') or 'http://localhost:11434').strip().rstrip('/')
            model = (data.get('model') or llm.get('model') or '').strip()
            llm.update({
                'provider': 'ollama',
                'model': model,
                'api_base': api_base,
                'api_key': '',
            })
            cfg['active_backend'] = 'ollama'
            cfg.setdefault('ollama', {}).update({
                'model': model,
                'api_base': api_base,
            })
        else:
            preset = ONLINE_PROVIDERS.get(provider, {})
            online = cfg.setdefault('online', {})
            api_base = (data.get('api_base') or preset.get('api_base') or online.get('api_base') or llm.get('api_base') or '').strip().rstrip('/')
            model = (data.get('model') or preset.get('default_model') or online.get('model') or llm.get('model') or '').strip()
            api_key = data.get('api_key')
            if api_key is None or api_key == '':
                api_key = online.get('api_key') or llm.get('api_key', '')
            else:
                api_key = api_key.strip()

            llm.update({
                'provider': provider,
                'model': model,
                'api_base': api_base,
                'api_key': api_key,
            })
            cfg['active_backend'] = 'online'
            online.update({
                'provider': provider,
                'api_base': api_base,
                'api_key': api_key,
                'model': model,
            })

        if 'temperature' in data:
            llm['temperature'] = data['temperature']
        if 'max_tokens' in data:
            llm['max_tokens'] = data['max_tokens']

        # Mark first run complete if this is the first setup
        if config.is_first_run():
            cfg['first_run'] = False

        config.save_config()
        return jsonify({'message': 'Configuration updated successfully', 'config': _mask_api_key(llm)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/api/config/search', methods=['GET', 'POST'])
def search_config():
    """Get/update web/literature search provider configuration."""
    from config_manager import get_config_manager, SEARCH_PROVIDER_DEFAULTS
    mgr = get_config_manager()
    project_cfg = mgr.get_project_config()
    search = project_cfg.setdefault('search', {})
    providers = search.setdefault('providers', {})
    for key, meta in SEARCH_PROVIDER_DEFAULTS.items():
        providers.setdefault(key, {'enabled': meta['enabled'], 'api_key': '', 'api_base': ''})

    if request.method == 'GET':
        return jsonify({'ok': True, 'search': _public_search_config(search)})

    data = request.get_json(silent=True) or {}
    incoming = data.get('providers') or {}
    for key, item in incoming.items():
        if key not in SEARCH_PROVIDER_DEFAULTS or not isinstance(item, dict):
            continue
        cur = providers.setdefault(key, {'enabled': False, 'api_key': '', 'api_base': ''})
        if 'enabled' in item:
            cur['enabled'] = bool(item.get('enabled'))
        if item.get('api_key'):
            cur['api_key'] = str(item.get('api_key')).strip()
        if 'api_base' in item:
            cur['api_base'] = str(item.get('api_base') or '').strip().rstrip('/')
    if isinstance(data.get('default_sources'), list):
        search['default_sources'] = [
            s for s in data['default_sources']
            if s in SEARCH_PROVIDER_DEFAULTS
        ] or ['openalex', 'semantic_scholar']
    mgr.save_project_config(project_cfg)
    return jsonify({'ok': True, 'search': _public_search_config(search)})


@bp.route('/api/config/app_paths', methods=['GET', 'POST'])
def app_paths_config():
    """Get/update shared Chat and Parameter Optimizer path defaults."""
    from config_manager import get_config_manager
    mgr = get_config_manager()
    project_cfg = mgr.get_project_config()
    paths = project_cfg.setdefault('app_paths', mgr.get_app_paths())

    if request.method == 'GET':
        return jsonify({'ok': True, 'app_paths': paths})

    data = request.get_json(silent=True) or {}
    if 'chat_workspace_enabled' in data:
        paths['chat_workspace_enabled'] = bool(data.get('chat_workspace_enabled'))
    if 'chat_workspace_paths' in data:
        raw = data.get('chat_workspace_paths')
        if isinstance(raw, str):
            vals = [x.strip() for x in raw.replace(';', '\n').replace(',', '\n').splitlines() if x.strip()]
        elif isinstance(raw, list):
            vals = [str(x).strip() for x in raw if str(x).strip()]
        else:
            vals = []
        paths['chat_workspace_paths'] = vals
    for key in ('geo_workspace_root', 'geo_literature_root', 'geo_output_dir'):
        if key in data:
            paths[key] = str(data.get(key) or '').strip()
    mgr.save_project_config(project_cfg)
    return jsonify({'ok': True, 'app_paths': paths})


@bp.route('/api/config/coding_agent', methods=['GET', 'POST'])
def coding_agent_config():
    """Get/update the project-scoped coding backend configuration."""
    from config_manager import get_config_manager
    mgr = get_config_manager()
    project_cfg = mgr.get_project_config()
    cfg = project_cfg.setdefault('coding_agent', mgr.get_coding_agent_config())

    if request.method == 'GET':
        status = {}
        try:
            from seismo_code.external_coding_agent import external_agent_status
            status = external_agent_status(cfg)
        except Exception as exc:
            status = {'available': False, 'error': str(exc)}
        return jsonify({'ok': True, 'coding_agent': cfg, 'status': status})

    data = request.get_json(silent=True) or {}
    backend = str(data.get('backend') or cfg.get('backend') or 'builtin').strip()
    if backend not in {'builtin', 'aider', 'openhands'}:
        return jsonify({'ok': False, 'error': f'Invalid coding agent backend: {backend}'}), 400
    cfg['backend'] = backend
    # Backend selection is the source of truth. Keep `enabled` only for older clients.
    cfg['enabled'] = backend != 'builtin'
    for key in ('command', 'model', 'extra_args'):
        if key in data:
            cfg[key] = str(data.get(key) or '').strip()
    if 'timeout_s' in data:
        try:
            cfg['timeout_s'] = max(30, min(7200, int(data.get('timeout_s') or 900)))
        except Exception:
            cfg['timeout_s'] = 900
    if 'auto_approve' in data:
        cfg['auto_approve'] = bool(data.get('auto_approve'))
    if 'use_current_llm_env' in data:
        cfg['use_current_llm_env'] = bool(data.get('use_current_llm_env'))
    mgr.save_project_config(project_cfg)
    return jsonify({'ok': True, 'coding_agent': cfg})


@bp.route('/api/llm/ollama/models', methods=['GET'])
def get_ollama_models():
    """Get available Ollama models with a single short HTTP request."""
    from config_manager import get_config_manager
    config = get_config_manager()
    recommended = config.get_recommended_models().get('ollama', [])
    ollama_cfg = config.config.get('ollama', {}) if isinstance(config.config.get('ollama', {}), dict) else {}
    llm_cfg = config.config.get('llm', {}) if isinstance(config.config.get('llm', {}), dict) else {}
    saved_llm_base = llm_cfg.get('api_base') if llm_cfg.get('provider') == 'ollama' else ''
    api_base = (
        request.args.get('api_base')
        or ollama_cfg.get('api_base')
        or saved_llm_base
        or 'http://localhost:11434'
    ).strip().rstrip('/')
    if not api_base.startswith(('http://localhost', 'http://127.0.0.1', 'http://0.0.0.0', 'http://[::1]')):
        api_base = 'http://localhost:11434'

    models = []
    available = False
    error = ''
    try:
        req = _ur.Request(f'{api_base}/api/tags')
        with _ur.urlopen(req, timeout=2.5) as resp:
            available = resp.status == 200
            if available:
                data = json.loads(resp.read().decode('utf-8'))
                models = _parse_model_list(data)
    except Exception as exc:
        error = str(exc)

    return jsonify({
        'installed': models,
        'recommended': recommended,
        'ollama_available': available,
        'api_base': api_base,
        'error': error,
    })


@bp.route('/api/llm/ollama/pull', methods=['POST'])
def pull_ollama_model():
    """Pull an Ollama model and track progress."""
    data = request.get_json(silent=True) or {}
    model_name = data.get('model')

    if not model_name:
        return jsonify({'error': 'Model name required'}), 400

    _pull_status[model_name] = {'status': 'pulling', 'progress': 0, 'detail': '', 'error': ''}

    def pull_in_background():
        try:
            # Use Ollama streaming pull API to track progress
            url     = 'http://localhost:11434/api/pull'
            payload = json.dumps({'name': model_name, 'stream': True}).encode()
            req     = _ur.Request(url, data=payload, method='POST',
                                  headers={'Content-Type': 'application/json'})
            with _ur.urlopen(req, timeout=1800) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    status_msg = obj.get('status', '')
                    total      = obj.get('total', 0)
                    completed  = obj.get('completed', 0)
                    pct = int(completed * 100 / total) if total else 0
                    detail = f"{completed/1e9:.1f} GB / {total/1e9:.1f} GB" if total else status_msg
                    _pull_status[model_name] = {
                        'status': 'pulling', 'progress': pct,
                        'detail': detail, 'error': ''
                    }
            _pull_status[model_name] = {'status': 'done', 'progress': 100, 'detail': '', 'error': ''}
        except Exception as ex:
            _pull_status[model_name] = {'status': 'error', 'progress': 0,
                                        'detail': '', 'error': str(ex)}

    thread = threading.Thread(target=pull_in_background, daemon=True)
    thread.start()
    return jsonify({'message': f'Started pulling model: {model_name}'})


@bp.route('/api/llm/online/models', methods=['GET'])
def get_online_api_models():
    """从在线 API 获取可用模型列表。支持 GET 参数: provider, api_base, api_key, model。"""
    from config_manager import get_config_manager

    provider = request.args.get('provider', '').strip()
    api_base = request.args.get('api_base', '').strip()
    api_key  = request.args.get('api_key', '').strip()
    current_model = request.args.get('model', '').strip()

    if not api_base and provider in ONLINE_PROVIDERS:
        api_base = ONLINE_PROVIDERS[provider].get('api_base', '')

    # 如果请求没带 api_key，使用已保存的 online/llm key
    if not api_key:
        cfg = get_config_manager()
        if not current_model:
            current_model = (
                cfg.config.get('online', {}).get('model')
                or cfg.config.get('llm', {}).get('model', '')
            )
        api_key = (
            cfg.config.get('online', {}).get('api_key')
            or cfg.config.get('llm', {}).get('api_key', '')
        )

    if not api_base:
        return jsonify({'error': '请提供 api_base 参数', 'models': []}), 400
    if not api_key:
        return jsonify({'error': '请先填写并保存 API Key，或在参数中提供 api_key', 'models': []}), 400

    import urllib.request as _urq
    import urllib.error as _ure

    url = api_base.rstrip('/') + '/models'
    try:
        req = _urq.Request(
            url,
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json',
            }
        )
        with _urq.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        models = _parse_model_list(data)
        if not models:
            return jsonify({'error': '接口返回了空模型列表', 'models': []}), 502

        return jsonify({'models': models, 'source': 'models_endpoint'})

    except _ure.HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', errors='ignore')[:240]
        except Exception:
            pass
        msgs = {
            401: 'API Key 无效或已过期 (401)',
            403: '访问被拒绝，请检查权限 (403)',
            404: '该平台不支持模型列表接口 (404)',
        }
        candidates = _fallback_model_candidates(provider, api_base, current_model)
        if e.code in {400, 404, 405, 501} and candidates:
            verified = []
            for model in candidates[:4]:
                try:
                    if _chat_model_available(api_base, api_key, model):
                        verified.append(model)
                        break
                except Exception:
                    continue
            models = verified or candidates
            return jsonify({
                'models': models,
                'source': 'chat_probe' if verified else 'fallback',
                **({
                    'notice': (
                        f'已验证当前模型可用；该平台的 /models 列表接口返回 HTTP {e.code}。'
                    )
                } if verified else {
                    'warning': (
                        f'/models 接口不可用或返回 HTTP {e.code}，'
                        '但聊天接口可能仍可使用；已使用当前/预设模型。'
                    )
                }),
            })

        detail = msgs.get(e.code, f'HTTP {e.code}: {e.reason}')
        if body:
            detail += f' — {body}'
        return jsonify({'error': detail, 'models': []}), 502
    except (_ure.URLError, OSError) as e:
        return jsonify({'error': f'网络连接失败: {e.reason if hasattr(e, "reason") else str(e)}', 'models': []}), 502
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}', 'models': []}), 502


@bp.route('/api/llm/test', methods=['POST'])
def test_llm_connection():
    """轻量测试 LLM 连接：发送一条最短的 chat 消息，返回详细错误信息。"""
    from config_manager import get_config_manager

    manager = get_config_manager()
    cfg = dict(manager.get_llm_config())
    body = request.get_json(silent=True) or {}
    if body:
        saved_online = manager.config.get('online', {})
        saved_llm = manager.config.get('llm', {})
        cfg.update({
            'provider': body.get('provider', cfg.get('provider', 'ollama')),
            'model': body.get('model', cfg.get('model', '')),
            'api_base': body.get('api_base', cfg.get('api_base', '')),
        })
        if body.get('api_key'):
            cfg['api_key'] = body.get('api_key')
        else:
            cfg['api_key'] = saved_online.get('api_key') or saved_llm.get('api_key') or cfg.get('api_key', '')

    provider = cfg.get('provider', 'ollama')
    model    = cfg.get('model', '')
    api_base = cfg.get('api_base', 'http://localhost:11434')
    api_key  = cfg.get('api_key', '')

    if not model:
        return jsonify({'ok': False, 'error': '未配置模型，请先保存配置'})

    import urllib.request as _urq
    import urllib.error as _ure

    # ── Ollama 分支 ─────────────────────────────────────────────────
    if provider == 'ollama':
        try:
            req = _urq.Request(f'{api_base.rstrip("/")}/api/tags')
            with _urq.urlopen(req, timeout=5) as resp:
                ok = resp.status == 200
            if ok:
                return jsonify({'ok': True, 'message': f'Ollama 连接正常，当前模型: {model}'})
            return jsonify({'ok': False, 'error': 'Ollama 服务未响应，请运行 ollama serve'})
        except Exception as e:
            return jsonify({'ok': False, 'error': f'Ollama 连接失败: {str(e)}'})

    # ── 在线 API 分支 ────────────────────────────────────────────────
    if not api_key:
        return jsonify({'ok': False, 'error': '未配置 API Key，请填写后保存'})

    payload = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': 'hi'}],
        'max_tokens': 5,
        'temperature': 0,
    }).encode('utf-8')

    try:
        req = _urq.Request(
            f'{api_base.rstrip("/")}/chat/completions',
            data=payload,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {api_key}',
            },
            method='POST',
        )
        with _urq.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        if result.get('choices'):
            return jsonify({'ok': True, 'message': f'✓ {model} 响应正常'})
        return jsonify({'ok': False, 'error': '模型返回了空结果，请检查模型名称'})

    except _ure.HTTPError as e:
        body = ''
        try: body = e.read().decode('utf-8', errors='ignore')[:200]
        except Exception: pass
        msgs = {
            401: f'API Key 无效或已过期 (401)',
            403: f'访问被拒绝，请检查权限 (403)',
            404: f'模型 "{model}" 不存在，请确认模型名称 (404)',
            429: '请求频率超限，稍后再试 (429)',
        }
        err = msgs.get(e.code, f'HTTP {e.code}')
        if body: err += f' — {body}'
        return jsonify({'ok': False, 'error': err})
    except (_ure.URLError, OSError) as e:
        reason = e.reason if hasattr(e, 'reason') else str(e)
        if 'timed out' in str(reason).lower():
            return jsonify({'ok': False, 'error': '连接超时，请检查网络或 API 地址是否正确'})
        return jsonify({'ok': False, 'error': f'网络连接失败: {reason}'})
    except Exception as e:
        return jsonify({'ok': False, 'error': f'测试失败: {str(e)}'})


@bp.route('/api/llm/ollama/pull/status', methods=['GET'])
def pull_ollama_status():
    """Return current pull progress for a model."""
    model_name = request.args.get('model', '')
    if not model_name:
        return jsonify({'error': 'model param required'}), 400
    info = _pull_status.get(model_name, {'status': 'unknown', 'progress': 0, 'detail': '', 'error': ''})
    return jsonify(info)
