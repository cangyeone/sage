"""Run record API routes."""
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from helpers import path_is_within_root
from state import _PROJECT_ROOT
from run_records import get_run, list_runs


bp = Blueprint("records", __name__)


@bp.route("/api/runs", methods=["GET"])
def runs_list():
    limit = request.args.get("limit", "50")
    try:
        limit_i = max(1, min(200, int(limit)))
    except ValueError:
        limit_i = 50
    return jsonify({"ok": True, "runs": list_runs(limit_i)})


@bp.route("/api/runs/<run_id>", methods=["GET"])
def runs_get(run_id):
    rec = get_run(run_id)
    if not rec:
        return jsonify({"ok": False, "error": "run not found"}), 404
    return jsonify({"ok": True, "run": rec})


@bp.route("/api/runs/<run_id>/artifact/<int:index>", methods=["GET"])
def runs_artifact(run_id, index):
    rec = get_run(run_id)
    if not rec:
        return jsonify({"ok": False, "error": "run not found"}), 404
    artifacts = rec.get("artifacts") or []
    if index < 0 or index >= len(artifacts):
        return jsonify({"ok": False, "error": "artifact not found"}), 404

    path = Path(artifacts[index]).expanduser().resolve()
    if not path.is_file():
        return jsonify({"ok": False, "error": "artifact file missing"}), 404

    roots = [(_PROJECT_ROOT / "outputs").resolve()]
    output_dir = (rec.get("metadata") or {}).get("output_dir")
    if output_dir:
        roots.append(Path(output_dir).expanduser().resolve())
    if not any(path_is_within_root(path, root) for root in roots):
        return jsonify({"ok": False, "error": "artifact outside allowed run directory"}), 403

    return send_file(str(path), as_attachment=True, download_name=path.name)
