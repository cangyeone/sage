#!/usr/bin/env python3
"""
SeismicX Web Interface

Flask-based web application for seismic analysis skills:
- Phase picking
- Phase association
- Polarity analysis

Usage:
    python web_app/app.py [--port PORT] [--host HOST]
"""

import os
import sys
import subprocess
import threading
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for

# Add parent directory to path for config_manager
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_manager import get_config_manager
except ImportError:
    class DummyConfigManager:
        def is_first_run(self): return False
        def interactive_setup(self): pass
        def get_llm_config(self): return {}
        def set_llm_provider(self, p): pass
        def set_llm_model(self, m): pass
        def set_api_key(self, k): pass
        def set_api_base(self, b): pass
        def mark_first_run_complete(self): pass
        def check_ollama_available(self): return False
        def get_ollama_models(self): return []
        def pull_ollama_model(self, m): return False
        def get_recommended_models(self): return {}
    def get_config_manager():
        return DummyConfigManager()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_app/uploads'
app.config['OUTPUT_FOLDER'] = 'web_app/outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Task status tracking
tasks = {}


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


@app.route('/')
def index():
    """Redirect root to chat"""
    return redirect(url_for('chat_page'))


@app.route('/api/tasks', methods=['GET'])
def list_tasks():
    """List all tasks"""
    return jsonify({
        'tasks': {k: {kk: vv for kk, vv in v.items() if kk not in ['stdout', 'stderr']}
                  for k, v in tasks.items()}
    })


@app.route('/api/task/<task_id>', methods=['GET'])
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


@app.route('/api/chat_picks/<task_id>', methods=['GET'])
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


@app.route('/api/pick', methods=['POST'])  # kept for conversational_agent compatibility
def submit_picking():
    """Submit phase picking job"""
    data = request.json

    if not data.get('input_dir'):
        return jsonify({'error': 'Input directory required'}), 400

    task_id = f"pick_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"

    # Build command
    cmd = f"python pnsn/picker.py"
    cmd += f" -i {data['input_dir']}"
    cmd += f" -o {app.config['OUTPUT_FOLDER']}/{task_id}"
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


@app.route('/api/associate', methods=['POST'])
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
    cmd += f" -o {app.config['OUTPUT_FOLDER']}/{task_id}.txt"
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


@app.route('/api/polarity', methods=['POST'])
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
    cmd += f" -o {app.config['OUTPUT_FOLDER']}/{task_id}_polarity.txt"
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


@app.route('/api/output/<filename>', methods=['GET'])
def download_output(filename):
    """Download output file"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'running')
    })


# ==================== LLM Configuration Endpoints ====================

@app.route('/llm-settings')
def llm_settings_page():
    """LLM settings page"""
    return render_template('llm_settings.html')


# ==================== Chat Endpoints ====================

@app.route('/chat')
def chat_page():
    """Chat interface page"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat_message():
    """Process chat message"""
    data = request.json
    user_message = data.get('message', '')
    workspace_path = data.get('workspace', '')

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

    # Prepend workspace file listing to message if relevant
    if workspace_path:
        ws_ctx = _inject_workspace_context(user_message, workspace_path)
        if ws_ctx:
            user_message = user_message + "\n\n[系统已获取文件信息]\n" + ws_ctx

    # Import and use conversational agent
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from conversational_agent import get_agent

        agent = get_agent()
        result = agent.process_message(user_message)

        # process_message wraps execution_result under 'data'
        inner = result.get('data', {}) or {}
        inner_results = inner.get('results', {})

        # If the action is to display a plot, pass waveform_data + picks to frontend
        if result.get('action') == 'display_plot':
            waveform_data = inner_results.get('waveform_data')
            if waveform_data:
                result['waveform_data'] = waveform_data
                result['waveform_title'] = inner_results.get('title', '')
                picks_data = inner_results.get('picks_data')
                if picks_data:
                    result['picks_data'] = picks_data

        # Batch picking: launch picker.py in background, return task_id
        if result.get('action') == 'batch_picking_async':
            cmd = inner_results.get('command', '')
            cwd = inner_results.get('cwd', os.getcwd())
            picks_file = inner_results.get('picks_file', '')
            task_id = f"batch_pick_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            tasks[task_id] = {
                'id': task_id, 'type': 'batch_picking',
                'status': 'running', 'command': cmd,
                'picks_file': picks_file,
            }
            threading.Thread(
                target=run_task,
                args=(task_id, cmd, 'batch_picking'),
                kwargs={'cwd': cwd},
                daemon=True).start()
            result['pick_task_id'] = task_id
            result['picks_file'] = picks_file

        # SagePicker inline async: launch in background thread with live progress
        if result.get('action') == 'sage_picking_async':
            inp_dir    = inner_results.get('input_dir', '')
            mdl_path   = inner_results.get('model_path', '')
            incomplete = inner_results.get('incomplete', 'skip')
            out_base   = inner_results.get('output_base', '')
            picks_file = inner_results.get('picks_file', '')

            task_id = f"sage_pick_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            tasks[task_id] = {
                'id': task_id, 'type': 'sage_picking',
                'status': 'running',
                'picks_file': picks_file,
                'progress': {
                    'current': 0, 'total': 0,
                    'n_picks': 0, 'current_station': '',
                },
            }

            def _run_sage_inline(tid, _inp, _mdl, _mode, _out):
                try:
                    import sys as _sys
                    _proj = str(Path(__file__).parent.parent)
                    if _proj not in _sys.path:
                        _sys.path.insert(0, _proj)
                    from pnsn.sage_picker import SagePicker as _SP

                    def _cb(station, done, total, n_picks=0):
                        tasks[tid]['progress'] = {
                            'current': done,
                            'total': total,
                            'n_picks': n_picks,
                            'current_station': station,
                        }

                    picker = _SP(_mdl, samplerate=100.0)
                    res = picker.pick_directory(_inp, _out, incomplete=_mode,
                                               progress_cb=_cb)
                    tasks[tid]['status'] = 'completed'
                    tasks[tid]['result'] = {
                        'n_stations': res['n_stations'],
                        'n_picks': res['n_picks'],
                        'skipped': len(res.get('skipped', [])),
                        'output': res['output'],
                    }
                    tasks[tid]['progress']['current'] = res['n_stations']
                    tasks[tid]['progress']['total']   = res['n_stations']
                    tasks[tid]['progress']['n_picks'] = res['n_picks']
                except Exception as _e:
                    tasks[tid]['status'] = 'error'
                    tasks[tid]['stderr'] = str(_e)

            threading.Thread(
                target=_run_sage_inline,
                args=(task_id, inp_dir, mdl_path, incomplete, out_base),
                daemon=True).start()

            result['pick_task_id'] = task_id
            result['picks_file'] = picks_file

        # Async picking: start background task, return waveform immediately
        if result.get('action') == 'picking_async':
            cmd = inner_results.get('command', '')
            cwd = inner_results.get('cwd', os.getcwd())
            picks_file = inner_results.get('picks_file', '')
            tmp_dir = inner_results.get('tmp_dir', '')
            task_id = f"chat_pick_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            tasks[task_id] = {
                'id': task_id, 'type': 'chat_picking',
                'status': 'running', 'command': cmd,
                'picks_file': picks_file, 'tmp_dir': tmp_dir,
            }
            def _run_pick(tid, _cmd, _cwd, _picks_file, _tmp_dir):
                try:
                    import shutil
                    proc = subprocess.run(
                        _cmd, shell=True, capture_output=True,
                        text=True, timeout=300, cwd=_cwd)
                    tasks[tid]['returncode'] = proc.returncode
                    tasks[tid]['stdout'] = proc.stdout[-3000:]
                    tasks[tid]['stderr'] = proc.stderr[-3000:]
                    tasks[tid]['status'] = 'completed' if proc.returncode == 0 else 'failed'
                except Exception as e:
                    tasks[tid]['status'] = 'error'
                    tasks[tid]['stderr'] = str(e)
                finally:
                    import shutil
                    shutil.rmtree(_tmp_dir, ignore_errors=True)
            threading.Thread(
                target=_run_pick,
                args=(task_id, cmd, cwd, picks_file, tmp_dir),
                daemon=True).start()

            # Return waveform immediately + task_id for polling
            waveform_data = inner_results.get('waveform_data')
            if waveform_data:
                result['waveform_data'] = waveform_data
                result['waveform_title'] = inner_results.get('title', '')
            result['pick_task_id'] = task_id
            result['picks_file'] = picks_file

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'response': f'抱歉，处理您的消息时出错: {str(e)}',
            'action': 'error',
            'data': {}
        }), 500


@app.route('/api/llm/config', methods=['GET'])
def get_llm_config():
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


@app.route('/api/llm/config', methods=['POST'])
def update_llm_config():
    """Update LLM configuration"""
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


@app.route('/api/llm/ollama/models', methods=['GET'])
def get_ollama_models():
    """Get available Ollama models"""
    config = get_config_manager()
    models = config.get_ollama_models()
    recommended = config.get_recommended_models().get('ollama', [])
    
    return jsonify({
        'installed': models,
        'recommended': recommended,
        'ollama_available': config.check_ollama_available()
    })


@app.route('/api/llm/ollama/pull', methods=['POST'])
def pull_ollama_model():
    """Pull an Ollama model"""
    data = request.json
    model_name = data.get('model')
    
    if not model_name:
        return jsonify({'error': 'Model name required'}), 400
    
    config = get_config_manager()
    
    # Start pulling in background thread
    def pull_in_background():
        success = config.pull_ollama_model(model_name)
        # Could add a status tracking mechanism here
    
    thread = threading.Thread(target=pull_in_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': f'Started pulling model: {model_name}'})


# ==================== Knowledge Base & RAG ====================

# 临时会话 RAG：session_id → {"chunks": [...], "doc_names": [...]}
_session_docs: dict = {}

UPLOAD_FOLDER_CHAT = Path(__file__).parent / "uploads" / "chat_pdfs"
UPLOAD_FOLDER_CHAT.mkdir(parents=True, exist_ok=True)


def _get_llm_config() -> dict:
    """统一获取 LLM 配置，直接读 config.json（LLM 设置页保存的配置）。"""
    try:
        # 每次重新加载 config_manager 以获取最新配置
        from config_manager import LLMConfigManager
        return LLMConfigManager().get_llm_config()
    except Exception:
        return {}


def _llm_call(messages: list, llm_cfg: dict, max_tokens: int = 2000) -> str:
    """向 LLM 发请求，返回回复文本。失败时抛出异常。"""
    import urllib.request, json as _json
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
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = _json.loads(resp.read().decode())

    if provider == "ollama":
        return body.get("message", {}).get("content", "").strip()
    return body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


# ==================== Workspace (local filesystem access) ====================

import re as _re

def _get_workspace_config() -> dict:
    """Read workspace config from config.json."""
    try:
        from config_manager import LLMConfigManager
        cfg = LLMConfigManager()
        return cfg.config.get('workspace', {'enabled': False, 'path': ''})
    except Exception:
        return {'enabled': False, 'path': ''}

def _save_workspace_config(enabled: bool, path: str):
    from config_manager import LLMConfigManager
    cfg = LLMConfigManager()
    cfg.config['workspace'] = {'enabled': enabled, 'path': path}
    cfg.save_config()

@app.route('/api/workspace/config', methods=['GET'])
def workspace_config_get():
    return jsonify(_get_workspace_config())

@app.route('/api/workspace/config', methods=['POST'])
def workspace_config_post():
    data = request.json or {}
    _save_workspace_config(bool(data.get('enabled')), data.get('path', ''))
    return jsonify({'ok': True})

@app.route('/api/workspace/ls', methods=['GET'])
def workspace_ls():
    """List directory contents, sandboxed to the configured workspace root."""
    import os as _os
    ws = _get_workspace_config()
    if not ws.get('enabled'):
        return jsonify({'ok': False, 'error': '未启用工作目录访问'}), 403

    root = _os.path.expanduser(ws.get('path', ''))
    req_path = request.args.get('path', root)
    req_path = _os.path.expanduser(req_path)

    # Sandbox: must be inside the configured root
    abs_root = _os.path.realpath(root)
    abs_req  = _os.path.realpath(req_path)
    if not abs_req.startswith(abs_root):
        return jsonify({'ok': False, 'error': '路径超出授权目录范围'}), 403

    if not _os.path.exists(abs_req):
        return jsonify({'ok': False, 'error': f'路径不存在: {req_path}'}), 404

    try:
        entries = []
        if _os.path.isdir(abs_req):
            for name in sorted(_os.listdir(abs_req)):
                full = _os.path.join(abs_req, name)
                stat = _os.stat(full)
                entries.append({
                    'name': name,
                    'type': 'dir' if _os.path.isdir(full) else 'file',
                    'size': stat.st_size,
                    'path': full,
                })
        else:
            # Single file info
            stat = _os.stat(abs_req)
            entries.append({'name': _os.path.basename(abs_req), 'type': 'file',
                           'size': stat.st_size, 'path': abs_req})
        return jsonify({'ok': True, 'path': abs_req, 'entries': entries})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

def _inject_workspace_context(message: str, workspace_path: str) -> str:
    """If message mentions a path and workspace is enabled, inject directory listing."""
    import os as _os
    if not workspace_path:
        return ''
    ws = _get_workspace_config()
    if not ws.get('enabled'):
        return ''

    root = _os.path.expanduser(ws.get('path', ''))
    abs_root = _os.path.realpath(root)

    # Find any path-like strings in the message
    paths_found = _re.findall(r'[~/][\w./\-]+', message)
    context_parts = []

    for p in paths_found:
        p_exp = _os.path.expanduser(p)
        p_abs = _os.path.realpath(p_exp) if p_exp.startswith('/') else _os.path.realpath(_os.path.join(abs_root, p_exp))
        # Must be within root
        if not p_abs.startswith(abs_root):
            continue
        if _os.path.isdir(p_abs):
            try:
                names = sorted(_os.listdir(p_abs))
                lines = []
                for n in names[:60]:
                    full = _os.path.join(p_abs, n)
                    tag  = '/' if _os.path.isdir(full) else ''
                    lines.append(f'  {n}{tag}')
                context_parts.append(f"目录 {p_abs} 内容（共 {len(names)} 项）：\n" + '\n'.join(lines))
            except Exception:
                pass
        elif _os.path.isfile(p_abs):
            sz = _os.path.getsize(p_abs)
            context_parts.append(f"文件 {p_abs} 存在（大小：{sz} 字节）")

    return '\n\n'.join(context_parts)


# ── 技能管理 ────────────────────────────────────────────────────────────────

def _get_skill_loader():
    """按需导入 seismo_skill，避免启动时报错。"""
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        import seismo_skill as _sl
        return _sl
    except Exception:
        return None


@app.route('/skills')
def skills_page():
    return render_template('skills.html')


@app.route('/api/skills', methods=['GET'])
def skills_list():
    sl = _get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装', 'skills': []})
    sl.invalidate_cache()
    return jsonify({'ok': True, 'skills': sl.list_skills()})


@app.route('/api/skills/<name>', methods=['GET'])
def skills_get(name):
    sl = _get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    detail = sl.get_skill_detail(name)
    if detail is None:
        return jsonify({'ok': False, 'error': f'未找到技能：{name}'}), 404
    return jsonify({'ok': True, **detail})


@app.route('/api/skills', methods=['POST'])
def skills_save():
    """新建或更新用户自定义技能。"""
    sl = _get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    data = request.json or {}
    name = (data.get('name') or '').strip()
    text = (data.get('text') or '').strip()
    if not name or not text:
        return jsonify({'ok': False, 'error': 'name 和 text 不能为空'}), 400
    try:
        path = sl.skill_loader.save_user_skill(name, text)
        return jsonify({'ok': True, 'path': str(path)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/skills/<name>', methods=['DELETE'])
def skills_delete(name):
    sl = _get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'error': '技能模块未安装'}), 500
    ok = sl.skill_loader.delete_user_skill(name)
    if not ok:
        return jsonify({'ok': False, 'error': f'未找到可删除的用户技能：{name}（内置技能不可删除）'}), 404
    return jsonify({'ok': True})


@app.route('/api/skills/template', methods=['GET'])
def skills_template():
    """返回新技能的 Markdown 模板。"""
    sl = _get_skill_loader()
    if sl is None:
        return jsonify({'ok': False, 'template': ''})
    name = request.args.get('name', 'my_skill')
    title = request.args.get('title', '我的技能')
    keywords = request.args.get('keywords', '关键词1, 关键词2')
    desc = request.args.get('desc', '功能描述')
    tpl = sl.skill_loader.SKILL_TEMPLATE.format(
        name=name, title=title, keywords=keywords, description=desc
    )

    return jsonify({'ok': True, 'template': tpl})


@app.route('/knowledge')
def knowledge_page():
    """Knowledge base management page"""
    return render_template('knowledge.html')


# ── 知识库 API ──────────────────────────────────────────────────────────────

@app.route('/api/knowledge/status', methods=['GET'])
def knowledge_status():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_engine import get_knowledge_base
        kb = get_knowledge_base()
        return jsonify({"ok": True, **kb.status()})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "n_docs": 0, "n_chunks": 0, "n_vectors": 0})


@app.route('/api/knowledge/list', methods=['GET'])
def knowledge_list():
    try:
        from rag_engine import get_knowledge_base
        kb  = get_knowledge_base()
        docs = [
            {"doc_id": d.doc_id, "doc_name": d.doc_name,
             "n_pages": d.n_pages, "n_chunks": d.n_chunks,
             "added_at": d.added_at,
             "size_kb": round(d.size_bytes / 1024, 1)}
            for d in kb.list_docs()
        ]
        return jsonify({"ok": True, "docs": docs})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "docs": []})


@app.route('/api/knowledge/upload', methods=['POST'])
def knowledge_upload():
    """Upload & index one PDF into the persistent knowledge base."""
    if 'file' not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files['file']
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({"ok": False, "error": "Only PDF files are supported"}), 400

    # Save temporarily
    tmp_path = UPLOAD_FOLDER_CHAT / f.filename
    f.save(str(tmp_path))

    # Index in background — return immediately with task_id
    task_id = f"kb_idx_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    tasks[task_id] = {"id": task_id, "type": "kb_index",
                      "status": "running", "doc_name": f.filename}

    def _index(tid, path, name):
        try:
            from rag_engine import get_knowledge_base
            kb  = get_knowledge_base()
            logs = []
            meta = kb.add_pdf(str(path), progress_cb=lambda m: logs.append(m))
            tasks[tid]["status"]   = "completed"
            tasks[tid]["doc_name"] = name
            tasks[tid]["n_chunks"] = meta.n_chunks
            tasks[tid]["logs"]     = logs
        except Exception as ex:
            tasks[tid]["status"] = "error"
            tasks[tid]["error"]  = str(ex)
        finally:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    threading.Thread(target=_index, args=(task_id, tmp_path, f.filename),
                     daemon=True).start()
    return jsonify({"ok": True, "task_id": task_id})


@app.route('/api/knowledge/index_status/<task_id>', methods=['GET'])
def knowledge_index_status(task_id):
    t = tasks.get(task_id, {})
    return jsonify(t)


@app.route('/api/knowledge/delete/<doc_id>', methods=['DELETE'])
def knowledge_delete(doc_id):
    try:
        from rag_engine import get_knowledge_base
        ok = get_knowledge_base().delete_doc(doc_id)
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route('/api/knowledge/clear', methods=['POST'])
def knowledge_clear():
    try:
        from rag_engine import get_knowledge_base
        get_knowledge_base().clear()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# ── 聊天界面临时文档上传 ────────────────────────────────────────────────────

@app.route('/api/chat/upload', methods=['POST'])
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


@app.route('/api/chat/clear_session', methods=['POST'])
def chat_clear_session():
    sid = request.json.get('session_id', 'default')
    _session_docs.pop(sid, None)
    return jsonify({"ok": True})


# ── RAG 增强对话 ──────────────────────────────────────────────────────────────

@app.route('/api/chat/rag', methods=['POST'])
def chat_rag():
    """
    RAG-aware chat endpoint.
    Retrieves from:
      1. Session docs (uploaded in this chat session)
      2. Persistent knowledge base (if not empty)
    Then calls LLM directly with retrieved context.
    """
    data       = request.json or {}
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    mode       = data.get("mode", "rag")   # "rag" | "paper_read"

    if not user_msg:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    llm_cfg = _get_llm_config()
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
        ws_ctx = _inject_workspace_context(user_msg, workspace_path)
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

    # 2. 持久知识库（BGE-M3 向量检索）
    try:
        from rag_engine import get_knowledge_base
        kb = get_knowledge_base()
        if not kb.is_empty:
            kb_ctx = kb.build_rag_context(user_msg, top_k=4, max_chars=2000)
            if kb_ctx:
                context_parts.append(kb_ctx)
                for d in kb.list_docs():
                    if d.doc_name not in sources:
                        sources.append(d.doc_name)
    except Exception:
        pass

    # 3. seismo_skill 技能文档（按用户消息检索最相关技能，注入代码示例）
    try:
        sl = _get_skill_loader()
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
        system = (
            "你是 SAGE 系统内置的地震学专家助手，具备深厚的地震学和数据处理知识。\n"
            "请结合以下参考内容（来自知识库）准确回答用户问题。\n"
            "如果参考内容与问题无关，依靠自身知识回答即可。\n"
        )

    if context_parts:
        system += "\n\n===== 参考内容 =====\n" + "\n\n".join(context_parts)

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
        answer = _llm_call(messages, llm_cfg, max_tokens=2000)
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SeismicX Web Interface')
    parser.add_argument('--port', type=int, default=5010, help='Port to run on (default: 5000)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print("=" * 80)
    print("SeismicX Web Interface")
    print("=" * 80)
    print(f"\nStarting server on http://{args.host}:{args.port}")
    print("\nAvailable pages:")
    print("  - Chat:         http://localhost:{}/chat".format(args.port))
    print("  - Knowledge:    http://localhost:{}/knowledge".format(args.port))
    print("  - LLM Settings: http://localhost:{}/llm-settings".format(args.port))
    print("\nPress Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
