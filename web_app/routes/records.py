"""Run record API routes."""
from flask import Blueprint, jsonify, request

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
