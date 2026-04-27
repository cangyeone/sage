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


@bp.route('/api/llm/ollama/pull/status', methods=['GET'])
def pull_ollama_status():
    """Return current pull progress for a model."""
    model_name = request.args.get('model', '')
    if not model_name:
        return jsonify({'error': 'model param required'}), 400
    info = _pull_status.get(model_name, {'status': 'unknown', 'progress': 0, 'detail': '', 'error': ''})
    return jsonify(info)
