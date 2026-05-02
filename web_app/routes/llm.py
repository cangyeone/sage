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


@bp.route('/api/llm/config', methods=['GET'])
def llm_config_get():
    """Get current LLM configuration (reads directly from config.json)"""
    from config_manager import LLMConfigManager
    cfg_mgr = LLMConfigManager()
    llm_config = dict(cfg_mgr.config.get('llm', {}))

    # Hide API key for security
    if llm_config.get('api_key'):
        k = llm_config['api_key']
        llm_config['api_key_masked'] = '****' + k[-4:] if len(k) > 4 else '****'
        llm_config['api_key'] = ''   # don't send raw key to browser

    return jsonify({
        'config': llm_config,
        'first_run': cfg_mgr.is_first_run(),
        'ollama_available': cfg_mgr.check_ollama_available()
    })


@bp.route('/api/llm/config', methods=['POST'])
def update_llm_config():
    """Update LLM configuration"""
    from config_manager import get_config_manager
    data = request.json
    config = get_config_manager()

    try:
        if 'provider' in data:
            config.set_llm_provider(data['provider'])

        if 'model' in data:
            config.set_llm_model(data['model'])

        if 'api_key' in data and data['api_key']:
            config.set_api_key(data['api_key'])

        if 'api_base' in data:
            config.set_api_base(data['api_base'])

        if 'temperature' in data:
            config.config['llm']['temperature'] = data['temperature']
            config.save_config()

        if 'max_tokens' in data:
            config.config['llm']['max_tokens'] = data['max_tokens']
            config.save_config()

        # Mark first run complete if this is the first setup
        if config.is_first_run():
            config.mark_first_run_complete()

        return jsonify({'message': 'Configuration updated successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@bp.route('/api/llm/ollama/models', methods=['GET'])
def get_ollama_models():
    """Get available Ollama models"""
    from config_manager import get_config_manager
    config = get_config_manager()
    models = config.get_ollama_models()
    recommended = config.get_recommended_models().get('ollama', [])

    return jsonify({
        'installed': models,
        'recommended': recommended,
        'ollama_available': config.check_ollama_available()
    })


@bp.route('/api/llm/ollama/pull', methods=['POST'])
def pull_ollama_model():
    """Pull an Ollama model and track progress."""
    data = request.json
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
    """从在线 API 获取可用模型列表。支持 GET 参数: api_base, api_key (可选，优先用已保存的 key)"""
    from config_manager import get_config_manager

    api_base = request.args.get('api_base', '').strip()
    api_key  = request.args.get('api_key', '').strip()

    # 如果请求没带 api_key，使用已保存的
    if not api_key:
        cfg = get_config_manager()
        api_key = cfg.config.get('llm', {}).get('api_key', '')

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

        models = []
        if isinstance(data.get('data'), list):
            models = [m['id'] for m in data['data'] if isinstance(m, dict) and m.get('id')]
        elif isinstance(data.get('models'), list):
            models = [m.get('id') or m.get('name', '') for m in data['models'] if isinstance(m, dict)]

        models = sorted([m for m in models if m])
        if not models:
            return jsonify({'error': '接口返回了空模型列表', 'models': []}), 502

        return jsonify({'models': models})

    except _ure.HTTPError as e:
        msgs = {401: 'API Key 无效或已过期 (401)', 403: '访问被拒绝，请检查权限 (403)',
                404: '该平台不支持模型列表接口 (404)'}
        return jsonify({'error': msgs.get(e.code, f'HTTP {e.code}: {e.reason}'), 'models': []}), 502
    except (_ure.URLError, OSError) as e:
        return jsonify({'error': f'网络连接失败: {e.reason if hasattr(e, "reason") else str(e)}', 'models': []}), 502
    except Exception as e:
        return jsonify({'error': f'获取模型列表失败: {str(e)}', 'models': []}), 502


@bp.route('/api/llm/test', methods=['POST'])
def test_llm_connection():
    """轻量测试 LLM 连接：发送一条最短的 chat 消息，返回详细错误信息。"""
    from config_manager import get_config_manager

    cfg = get_config_manager().get_llm_config()
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
