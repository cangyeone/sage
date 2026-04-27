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
    save_workspace_config(bool(data.get('enabled')), data.get('path', ''))
    return jsonify({'ok': True})


@bp.route('/api/workspace/ls', methods=['GET'])
def workspace_ls():
    """List directory contents, sandboxed to the configured workspace root."""
    ws = get_workspace_config()
    if not ws.get('enabled'):
        return jsonify({'ok': False, 'error': '未启用工作目录访问'}), 403

    root = os.path.expanduser(ws.get('path', ''))
    req_path = request.args.get('path', root)
    req_path = os.path.expanduser(req_path)

    # Sandbox: must be inside the configured root
    abs_root = os.path.realpath(root)
    abs_req  = os.path.realpath(req_path)
    if not abs_req.startswith(abs_root):
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
