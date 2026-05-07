"""Smoke demo route."""
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

from state import _PROJECT_ROOT
from helpers import path_is_within_root


bp = Blueprint("smoke", __name__)


@bp.route("/api/smoke_demo/run", methods=["POST"])
def smoke_demo_run():
    data = request.get_json(silent=True) or {}
    out_dir = data.get("output_dir")
    smoke_root = (_PROJECT_ROOT / "outputs" / "smoke_demo").resolve()
    if out_dir:
        output_dir = Path(out_dir).expanduser().resolve()
        if not path_is_within_root(output_dir, smoke_root):
            return jsonify({
                "ok": False,
                "error": f"output_dir must be inside {smoke_root}",
            }), 400
    else:
        output_dir = (
            smoke_root /
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    try:
        from examples.sage_smoke_demo import run_smoke_demo
        return jsonify(run_smoke_demo(output_dir))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
