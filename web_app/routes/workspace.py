"""工作目录管理路由"""
from flask import Blueprint, request, jsonify
import os
from helpers import get_workspace_config, save_workspace_config, inject_workspace_context

bp = Blueprint('workspace', __name__)


@bp.route('/api/workspace/config', methods=['GET'])
def workspace_config_get():
    return jsonify(get_workspace_config())


@bp.route('/api/workspace/config', methods=['POST'])
def workspace_config_post():
    data = request.json or {}
    save_workspace_config(bool(data.get('enabled')), data.get('path', ''), data.get('paths'))
    return jsonify({'ok': True})


@bp.route('/api/workspace/ls', methods=['GET'])
def workspace_ls():
    """List directory contents, sandboxed to the configured workspace root."""
    ws = get_workspace_config()
    if not ws.get('enabled'):
        return jsonify({'ok': False, 'error': '未启用工作目录访问'}), 403

    roots = [os.path.realpath(os.path.expanduser(p)) for p in (ws.get('paths') or [ws.get('path', '')]) if p]
    if not roots:
        return jsonify({'ok': False, 'error': '未配置授权目录'}), 400
    root = roots[0]
    req_path = request.args.get('path', root)
    req_path = os.path.expanduser(req_path)

    abs_req  = os.path.realpath(req_path)
    if not any(os.path.commonpath([abs_req, abs_root]) == abs_root for abs_root in roots):
        return jsonify({'ok': False, 'error': '路径超出授权目录范围'}), 403

    if not os.path.exists(abs_req):
        return jsonify({'ok': False, 'error': f'路径不存在: {req_path}'}), 404

    try:
        entries = []
        if os.path.isdir(abs_req):
            for name in sorted(os.listdir(abs_req)):
                full = os.path.join(abs_req, name)
                stat = os.stat(full)
                entries.append({
                    'name': name,
                    'type': 'dir' if os.path.isdir(full) else 'file',
                    'size': stat.st_size,
                    'path': full,
                })
        else:
            # Single file info
            stat = os.stat(abs_req)
            entries.append({'name': os.path.basename(abs_req), 'type': 'file',
                           'size': stat.st_size, 'path': abs_req})
        return jsonify({'ok': True, 'path': abs_req, 'entries': entries})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
