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
    """Main dashboard"""
    return render_template('index.html')


@app.route('/picker')
def picker_page():
    """Phase picking interface"""
    return render_template('picker.html')


@app.route('/associator')
def associator_page():
    """Phase association interface"""
    return render_template('associator.html')


@app.route('/polarity')
def polarity_page():
    """Polarity analysis interface"""
    return render_template('polarity.html')


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
    # Include logs
    task['logs'] = {
        'stdout': task.get('stdout', ''),
        'stderr': task.get('stderr', '')
    }
    # Keep stderr at top level too (for error display), remove stdout only
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


@app.route('/api/pick', methods=['POST'])
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

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

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
    """Get current LLM configuration"""
    config = get_config_manager()
    llm_config = config.get_llm_config()
    
    # Hide API key for security
    if 'api_key' in llm_config and llm_config['api_key']:
        llm_config['api_key_masked'] = '****' + llm_config['api_key'][-4:] if len(llm_config['api_key']) > 4 else '****'
    
    return jsonify({
        'config': llm_config,
        'first_run': config.is_first_run(),
        'ollama_available': config.check_ollama_available()
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
    print("\nAvailable endpoints:")
    print("  - Main Dashboard:   http://localhost:{}/".format(args.port))
    print("  - Phase Picker:     http://localhost:{}/picker".format(args.port))
    print("  - Associator:       http://localhost:{}/associator".format(args.port))
    print("  - Polarity:         http://localhost:{}/polarity".format(args.port))
    print("  - LLM Settings:     http://localhost:{}/llm-settings".format(args.port))
    print("  - API Docs:         http://localhost:{}/api/tasks".format(args.port))
    print("\nPress Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
