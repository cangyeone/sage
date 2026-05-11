"""Parameter optimization agent routes.

This alpha backend treats parameter tuning as a project workflow: the user
defines modules, inputs/outputs, candidate parameters, and an objective; the
CodeEngine then writes and debugs the scripts inside the selected project
directory and saves the optimization trace for later scientific writing.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify, request, send_file

from helpers import get_llm_config, get_workspace_config

bp = Blueprint("parameter_optimization", __name__)

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _clean_id(value: str | None, default: str = "default") -> str:
    text = str(value or default)
    return "".join(ch for ch in text if ch.isalnum() or ch in "_-")[:80] or default


def _project_root_from_request(data: dict, project_id: str) -> Path:
    raw = (data.get("workspace_root") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    # Parameter optimization is a research sub-workflow. By default, keep its
    # durable run records inside the Science Analysis example workspace so the
    # science agent can reuse them as paper evidence.
    return (Path(__file__).resolve().parents[1] / "examples" / "science_analysis_agent").resolve()


def _is_authorized(path: Path, roots: list[str]) -> bool:
    if not roots:
        return True
    try:
        rp = path.resolve()
        for root in roots:
            if not root:
                continue
            rr = Path(root).expanduser().resolve()
            try:
                rp.relative_to(rr)
                return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def _build_workflow_prompt(data: dict, project_root: Path, output_dir: Path) -> str:
    workflow = data.get("workflow") or []
    if isinstance(workflow, str):
        workflow_text = workflow
    else:
        workflow_text = json.dumps(workflow, ensure_ascii=False, indent=2)
    objective = (data.get("objective") or "").strip()
    input_format = (data.get("input_format") or "").strip()
    final_output = (data.get("final_output") or "").strip()
    user_notes = (data.get("user_notes") or "").strip()
    multimodal_note = (
        "If image inputs are present and the configured model supports vision, analyze them; "
        "otherwise write an explicit warning and continue with text/numeric evidence."
        if data.get("use_multimodal", True)
        else "Do not rely on image understanding unless the user enables multimodal analysis."
    )
    return f"""You are SAGE Parameter Optimization Agent alpha.

Goal:
Convert the user's block/workflow description into a reproducible optimization run.
Use all relevant SKILLs and RAG evidence. You may combine multiple skills when useful.
Use CodeEngine discipline: inspect files, write small tests, run code, debug failures, and keep all outputs inside the project.

Project root:
{project_root}

Output directory:
{output_dir}

User-defined workflow modules:
{workflow_text}

Input format:
{input_format or "Infer from project files and user module definitions."}

Optimization objective:
{objective or "Infer a measurable target from the workflow and ask for clarification in the report if ambiguous."}

Expected final output:
{final_output or "best_parameters.json, optimization_history.csv, figures, and optimization_report.md"}

Additional user notes:
{user_notes or "(none)"}

Multimodal policy:
{multimodal_note}

Required execution:
1. Traverse the project directory and summarize usable data, scripts, images, model files, and prior outputs.
2. Convert the workflow modules into an executable DAG with explicit inputs, outputs, tunable parameters, objective metrics, and stopping rules.
3. Generate `optimization_plan.md` before heavy work.
4. Implement scripts under the provided output directory, not outside the project.
5. Run a small smoke test or mini test for each important function before the optimization loop.
6. Run a bounded optimization or mock-safe dry run when full training is too expensive; clearly label dry-run results.
7. Save `best_parameters.json`, `optimization_history.csv`, useful figures, and `optimization_report.md`.
8. Write a paper-ready `optimization_report.md`: objective, workflow DAG, search space, metrics, best parameters, failed/negative trials, limitations, and how the results support a scientific claim.
9. Include how these artifacts can be reused by the Science Analysis Agent for a report or paper.
10. Never fabricate results. If data/model requirements are missing, save a `missing_information.md` file with concrete next steps.
"""


def _summarize_result(result: Any, output_dir: Path) -> dict:
    return {
        "success": bool(getattr(result, "success", False)),
        "response": getattr(result, "response", "") or "",
        "code": getattr(result, "code", "") or "",
        "stdout": getattr(result, "stdout", "") or "",
        "figures": list(getattr(result, "figures", []) or []),
        "output_files": list(getattr(result, "output_files", []) or []),
        "attempts": getattr(result, "attempts", 0),
        "plan": list(getattr(result, "plan", []) or []),
        "script_path": getattr(result, "script_path", "") or "",
        "output_dir": str(output_dir),
    }


def _write_science_analysis_record(
    *,
    project_root: Path,
    project_id: str,
    job_id: str,
    data: dict,
    summary: dict,
    progress: list[dict],
    output_dir: Path,
) -> dict:
    """Persist optimization metadata as a Science Analysis input artifact."""
    record_dir = (
        project_root
        / "science_analysis_inputs"
        / "parameter_optimization"
        / _clean_id(project_id, "opt_project")
        / _clean_id(job_id, "opt_job")
    )
    record_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "parameter_optimization_run",
        "project_id": project_id,
        "job_id": job_id,
        "created_at": time.time(),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "request": data,
        "progress": progress,
        "summary": summary,
        "paper_reuse": {
            "recommended_section": "Methods/Results or Supplementary optimization experiments",
            "evidence_status": "verified_by_codeengine" if summary.get("success") else "incomplete_or_failed",
            "required_sources": [
                "workflow_request.json",
                "optimization_job_summary.json",
                "best_parameters.json",
                "optimization_history.csv",
                "optimization_report.md",
            ],
        },
    }
    json_path = record_dir / "parameter_optimization_run.json"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    artifacts = []
    for fp in list(summary.get("figures") or []) + list(summary.get("output_files") or []):
        artifacts.append(str(fp))
    progress_lines = "\n".join(
        f"- `{item.get('phase', 'log')}` {item.get('message', '')}" for item in progress[-80:]
    )
    artifact_lines = "\n".join(f"- `{fp}`" for fp in artifacts[:80]) or "- (none recorded)"
    md = f"""# Parameter Optimization Run

This file is a durable Science Analysis input record. It was generated by the
Parameter Optimization Agent and can be used by the Science Analysis Agent when
writing reports or papers.

## Metadata

- Project ID: `{project_id}`
- Job ID: `{job_id}`
- Project root: `{project_root}`
- Output directory: `{output_dir}`
- Success: `{bool(summary.get("success"))}`
- Code attempts: `{summary.get("attempts", 0)}`

## Research Objective

{(data.get("objective") or "No explicit objective was supplied.").strip()}

## Workflow Modules

```json
{json.dumps(data.get("workflow") or [], ensure_ascii=False, indent=2)}
```

## Paper Reuse Guidance

Use this run as optimization evidence only when the linked output files exist
and the reported metrics can be traced to `optimization_history.csv`,
`best_parameters.json`, or `optimization_report.md`. If the run failed or used
a dry run, describe it as a method/prototype result rather than a final finding.

## Generated Artifacts

{artifact_lines}

## Recent Runtime Log

{progress_lines or "- (no progress log recorded)"}
"""
    md_path = record_dir / "parameter_optimization_run.md"
    md_path.write_text(md, encoding="utf-8")
    return {
        "science_analysis_record_dir": str(record_dir),
        "science_analysis_record": str(md_path),
        "science_analysis_manifest": str(json_path),
    }


@bp.route("/api/parameter_optimization/start", methods=["POST"])
def parameter_optimization_start():
    data = request.get_json(silent=True) or {}
    project_id = _clean_id(data.get("project_id") or data.get("session_id"), "opt_project")
    project_root = _project_root_from_request(data, project_id)

    ws_cfg = get_workspace_config()
    authorized = []
    if ws_cfg.get("enabled"):
        authorized.extend(ws_cfg.get("paths") or [])
    authorized.extend(data.get("authorized_roots") or [])
    authorized.append(str(project_root))
    if not _is_authorized(project_root, authorized):
        return jsonify({"ok": False, "error": f"未授权访问项目目录：{project_root}"}), 403

    output_cfg = Path(data.get("output_dir") or "outputs/science_analysis_agent/parameter_optimization").expanduser()
    if output_cfg.is_absolute():
        try:
            output_cfg.resolve().relative_to(project_root)
            base_output = output_cfg
        except Exception:
            base_output = project_root / "outputs" / "science_analysis_agent" / "parameter_optimization"
    else:
        base_output = project_root / output_cfg
    job_id = "opt_" + uuid.uuid4().hex[:10]
    output_dir = base_output / project_id / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    cancel_event = threading.Event()
    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "progress": [],
            "result": None,
            "error": "",
            "cancel_event": cancel_event,
            "project_id": project_id,
            "project_root": str(project_root),
            "output_dir": str(output_dir),
            "created_at": time.time(),
        }

    def _emit(phase: str, message: str):
        with _lock:
            job = _jobs.get(job_id)
            if job is not None:
                job["progress"].append({"phase": phase, "message": message, "ts": time.time()})

    def _run():
        try:
            from seismo_code.code_engine import CodeEngine

            llm_cfg = get_llm_config()
            prompt = _build_workflow_prompt(data, project_root, output_dir)
            plan_path = output_dir / "workflow_request.json"
            plan_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            _emit("start", f"Parameter optimization workspace: {project_root}")
            _emit("skills", "CodeEngine will retrieve and combine relevant SKILLs and RAG context.")
            engine = CodeEngine(
                llm_config=llm_cfg,
                project_root=str(project_root),
                python_executable=llm_cfg.get("python_executable"),
            )

            def _progress(evt: dict):
                _emit(str(evt.get("phase") or "code"), str(evt.get("message") or evt.get("msg") or evt))

            result = engine.run(
                prompt,
                max_debug_rounds=int(data.get("max_debug_rounds") or 4),
                timeout=int(data.get("code_timeout_s") or 180),
                run_verify=True,
                on_progress=_progress,
                output_dir=str(output_dir),
                cancel_event=cancel_event,
            )
            summary = _summarize_result(result, output_dir)
            summary_path = output_dir / "optimization_job_summary.json"
            with _lock:
                progress = list((_jobs.get(job_id) or {}).get("progress") or [])
            record_paths = _write_science_analysis_record(
                project_root=project_root,
                project_id=project_id,
                job_id=job_id,
                data=data,
                summary=summary,
                progress=progress,
                output_dir=output_dir,
            )
            summary.update(record_paths)
            summary["run_kind"] = "parameter_optimization"
            summary["paper_reuse_note"] = (
                "This optimization run has been recorded under science_analysis_inputs "
                "so the Science Analysis Agent can cite it as experiment evidence."
            )
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            with _lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job["result"] = summary
                    job["status"] = "cancelled" if cancel_event.is_set() else "done"
        except Exception as exc:
            try:
                with _lock:
                    progress = list((_jobs.get(job_id) or {}).get("progress") or [])
                error_summary = {
                    "success": False,
                    "response": "",
                    "code": "",
                    "stdout": "",
                    "figures": [],
                    "output_files": [],
                    "attempts": 0,
                    "plan": [],
                    "script_path": "",
                    "output_dir": str(output_dir),
                    "error": str(exc),
                }
                record_paths = _write_science_analysis_record(
                    project_root=project_root,
                    project_id=project_id,
                    job_id=job_id,
                    data=data,
                    summary=error_summary,
                    progress=progress,
                    output_dir=output_dir,
                )
                error_summary.update(record_paths)
                (output_dir / "optimization_job_summary.json").write_text(
                    json.dumps(error_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            with _lock:
                job = _jobs.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    if "error_summary" in locals():
                        job["result"] = error_summary

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "project_root": str(project_root), "output_dir": str(output_dir)})


@bp.route("/api/parameter_optimization/poll/<job_id>", methods=["GET"])
def parameter_optimization_poll(job_id: str):
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    job.pop("cancel_event", None)
    return jsonify({"ok": True, **job})


@bp.route("/api/parameter_optimization/stop/<job_id>", methods=["POST"])
def parameter_optimization_stop(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"ok": False, "error": "job not found"}), 404
        event = job.get("cancel_event")
        if event is not None:
            event.set()
        job["status"] = "cancelling"
    return jsonify({"ok": True})


@bp.route("/api/parameter_optimization/artifact", methods=["GET"])
def parameter_optimization_artifact():
    raw = request.args.get("path") or ""
    if not raw:
        return jsonify({"ok": False, "error": "path is required"}), 400
    p = Path(raw).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 404
    allowed_suffixes = {".png", ".jpg", ".jpeg", ".svg", ".md", ".json", ".csv", ".txt", ".log", ".py"}
    if p.suffix.lower() not in allowed_suffixes:
        return jsonify({"ok": False, "error": "unsupported artifact type"}), 400
    mimetype = "image/png" if p.suffix.lower() == ".png" else None
    if p.suffix.lower() == ".svg":
        mimetype = "image/svg+xml"
    return send_file(str(p), mimetype=mimetype)
