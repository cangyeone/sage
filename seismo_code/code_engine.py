"""
code_engine.py — LLM-driven code generation and execution engine.

Single-request loop:  Plan → Code → Run → [Debug × N] → Verify → Return
Workflow loop:        Load → Topo-sort → per step: prompt+skill+RAG → Code → Run → [Debug × N]

Key improvements vs. original
------------------------------
1. Skill + RAG context forwarded to debugger
   The same skill docs used during code generation are passed to _debug_and_fix()
   so the debugger always knows which APIs are available when rewriting a fix.

2. No double-execution in workflow debug loop
   _debug_and_fix() accepts exec_dir and runs the fixed code inside the shared
   working directory directly — the outer loop no longer re-runs it.

3. Per-step semantic output check (_step_output_ok)
   After a successful exit code, verifies that expected output was actually
   produced (figures for plot steps, files for save steps). Triggers a targeted
   re-debug if the check fails.

Public API
----------
    engine = get_code_engine(llm_config)
    result = engine.run("Filter and plot waveforms in /data/")
    result = engine.run_workflow("my_workflow", user_request)

CLI:
    python -m seismo_code.code_engine --test    # no LLM required
    python -m seismo_code.code_engine --status
"""

from __future__ import annotations

import re
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .safe_executor import ExecutionResult, execute_bash, execute_code
from .ce_prompts import _CODEGEN_SYSTEM, _DEBUG_SYSTEM, _VERIFY_SYSTEM, _PLAN_SYSTEM
from .ce_utils import (
    CodeRunResult, DebugAttempt, StepResult, WorkflowRunResult,
    _call_llm, _extract_code, _is_bash_code, _extract_diagnosis,
    _extract_plan, _find_file_paths, _profile_file, _format_file_context,
    _pre_sanitize,
)

# Skill + RAG context builder — optional, graceful fallback
try:
    _root = str(Path(__file__).parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from seismo_skill import build_skill_context_with_rag as _build_ctx
except Exception:
    def _build_ctx(query: str, **_kw):  # type: ignore
        return "", ""


# ---------------------------------------------------------------------------
# CodeEngine
# ---------------------------------------------------------------------------

class CodeEngine:
    """
    Full-cycle code generation agent.

    Parameters
    ----------
    llm_config        : dict with keys provider/model/api_base/api_key/temperature/python_executable
    project_root      : root directory forwarded to safe_executor
    python_executable : interpreter for sandboxed execution
    """

    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        project_root: Optional[str] = None,
        python_executable: Optional[str] = None,
    ):
        if llm_config is None:
            llm_config = self._load_llm_config()
        self.llm_config        = llm_config
        self.project_root      = project_root or str(Path(__file__).parent.parent)
        self.python_executable = python_executable or llm_config.get("python_executable")
        self._history: List[Dict] = [{"role": "system", "content": _CODEGEN_SYSTEM}]
        self._last_exec_dir: Optional[str] = None

    # ── Config ────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_llm_config() -> Dict:
        try:
            from config_manager import LLMConfigManager
            cfg = LLMConfigManager().get_llm_config()
            if "python_executable" not in cfg:
                cfg["python_executable"] = sys.executable
            return cfg
        except Exception:
            return {"provider": "ollama", "model": "",
                    "api_base": "http://localhost:11434"}

    def is_llm_available(self) -> bool:
        try:
            provider = self.llm_config.get("provider", "ollama")
            api_base = self.llm_config.get("api_base", "http://localhost:11434")
            url = api_base.rstrip("/") + (
                "/api/tags" if provider == "ollama" else "/models"
            )
            urllib.request.urlopen(url, timeout=3)
            return True
        except Exception:
            return False

    # ── Progress emitter ──────────────────────────────────────────────────────

    def _emit(self, cb: Optional[Callable], phase: str, attempt: int, msg: str):
        if cb:
            try:
                cb({"phase": phase, "attempt": attempt, "message": msg})
            except Exception:
                pass

    # ── Executors ─────────────────────────────────────────────────────────────

    def _run_code(self, code: str, timeout: int) -> ExecutionResult:
        """Execute Python or bash code."""
        if _is_bash_code(code):
            clean = re.sub(r"^#\s*lang:bash\s*\n", "", code, count=1)
            return execute_bash(clean, project_root=self.project_root,
                                timeout=timeout, keep_dir=True)
        return execute_code(_pre_sanitize(code), project_root=self.project_root,
                            timeout=timeout, keep_dir=True,
                            python_executable=self.python_executable)

    def _run_code_in_dir(self, code: str, timeout: int,
                         shared_dir: Optional[str] = None) -> ExecutionResult:
        """Execute code inside a pre-existing shared directory."""
        clean = _pre_sanitize(code)
        extra_env = None
        if shared_dir:
            preamble = (f"import os as _wf_os\n"
                        f"_wf_os.chdir({shared_dir!r})\n"
                        f"_wf_os.environ['SAGE_OUTDIR'] = {shared_dir!r}\n")
            clean     = preamble + clean
            extra_env = {"SAGE_OUTDIR": shared_dir}
        return execute_code(clean, project_root=self.project_root,
                            timeout=timeout, keep_dir=True,
                            extra_env=extra_env,
                            python_executable=self.python_executable)

    # ── Output checkers ───────────────────────────────────────────────────────

    def _execution_success(self, exec_res: ExecutionResult) -> bool:
        """True when process exited cleanly with no traceback in output."""
        if not exec_res or not exec_res.success:
            return False
        combined = "\n".join([exec_res.stdout or "", exec_res.stderr or ""]).strip()
        if not combined:
            return True
        if re.search(r"Traceback \(most recent call last\):", combined, re.I):
            return False
        if re.search(
            r"^\s*(Error|Exception|AssertionError|ValueError|TypeError|NameError|"
            r"ImportError|ModuleNotFoundError|FileNotFoundError|OSError)[:\s]",
            combined, re.M,
        ):
            return False
        return True

    def _step_output_ok(self, step_desc: str,
                        exec_res: ExecutionResult) -> tuple[bool, str]:
        """
        Semantic output check applied after _execution_success() passes.

        Detects "silent success": process exited 0 but produced nothing when
        output was clearly expected.  Returns (ok, reason).
        ok=False causes the step to be treated as a debug target.
        """
        stdout  = (exec_res.stdout or "").strip()
        figures = exec_res.figures or []
        files   = exec_res.output_files or []

        # Check for expected figure output
        if re.search(r"plot|figure|图|绘制|visuali|chart|map|waveform|spectrogram|psd",
                     step_desc, re.I) and not figures:
            if exec_res.exec_dir:
                imgs = (list(Path(exec_res.exec_dir).glob("*.png"))
                        + list(Path(exec_res.exec_dir).glob("*.pdf")))
                if imgs:
                    return True, ""   # files exist, registry just missed them
            return False, f"No figure produced for step: {step_desc[:80]}"

        # Check for expected file output
        if (re.search(r"save|write|output|export|保存|输出|写入", step_desc, re.I)
                and not files and not figures and not stdout):
            if exec_res.exec_dir:
                new_files = [p for p in Path(exec_res.exec_dir).iterdir()
                             if p.is_file() and not p.name.startswith("run.")]
                if new_files:
                    return True, ""
            return False, f"No output files and no stdout for step: {step_desc[:80]}"

        return True, ""

    # ── Error context builder ─────────────────────────────────────────────────

    def _build_error_context(self, code: str, exec_res: ExecutionResult) -> str:
        parts  = []
        stderr = (exec_res.stderr or "").strip()
        stdout = exec_res.stdout.strip()

        is_bash = bool(re.search(
            r"(gmt |command not found|exit status \d|CalledProcessError|"
            r"run_gmt|GMT warning|GMT error|bash:|/bin/sh:)",
            stderr + stdout, re.I))
        is_py = bool(re.search(
            r"(Traceback \(most recent call last\)|Error:|Exception:|"
            r"SyntaxError|IndentationError|NameError|TypeError|ValueError)", stderr))

        if is_bash and not is_py:
            parts.append("=== ERROR TYPE: Bash/GMT script failure ===")
        elif is_py:
            parts.append("=== ERROR TYPE: Python runtime error ===")

        if stdout:
            parts.append("=== Partial stdout (last 1500 chars) ===\n" + stdout[-1500:])
        if stderr:
            parts.append("=== Traceback / stderr ===\n" + stderr[-3000:])
        if exec_res.error:
            parts.append("=== Error summary ===\n" + exec_res.error)
        if is_bash:
            gmt_err = re.findall(r"(?:GMT (?:Error|Warning)|error|Error).*", stderr, re.I)
            if gmt_err:
                parts.append("=== GMT/Bash key error lines ===\n" + "\n".join(gmt_err[-5:]))

        return "\n\n".join(parts) if parts else "No error details captured."

    # ── Debug + fix ───────────────────────────────────────────────────────────

    def _debug_and_fix(
        self,
        original_request: str,
        failed_code: str,
        exec_res: ExecutionResult,
        attempt: int,
        timeout: int,
        on_progress: Optional[Callable],
        file_contexts: Optional[List[str]] = None,
        skill_ctx: str = "",        # ← same skill docs as code-gen phase
        extra_rag_ctx: str = "",    # ← error-specific RAG docs
        exec_dir: Optional[str] = None,  # ← run fixed code in this dir (workflow)
    ) -> tuple[str, ExecutionResult, str]:
        """
        Ask the LLM debugger to fix failing code, then execute the fix.

        skill_ctx is forwarded from code generation so the debugger sees
        the same API documentation it should use when rewriting.
        exec_dir avoids double-execution: the fix is run here directly in
        the workflow's shared directory rather than re-run outside.

        Returns (fixed_code, new_exec_result, diagnosis).
        """
        error_ctx = self._build_error_context(failed_code, exec_res)

        file_ctx_str = ""
        if file_contexts:
            file_ctx_str = ("\n\n## Data file context (use EXACT column names)\n"
                            + "\n\n".join(file_contexts))

        # Build debug system prompt — skill context + error-specific RAG
        debug_system = _DEBUG_SYSTEM
        if skill_ctx:
            debug_system += (
                "\n\n## Skill documentation (same as code generation context)\n"
                + skill_ctx
                + "\n\nUse the skill APIs shown above when rewriting the fix.")
        if extra_rag_ctx:
            debug_system += (
                "\n\n## Error-targeted documentation\n"
                + extra_rag_ctx
                + "\n\nConsult the above to resolve API misuse or version-specific errors.")

        debug_messages = [
            {"role": "system", "content": debug_system},
            {"role": "user", "content": (
                f"## Original request\n{original_request}"
                f"{file_ctx_str}\n\n"
                f"## Failing code\n```python\n{failed_code}\n```\n\n"
                f"## Error output\n{error_ctx}\n\n"
                "Fix the code. Output [DIAGNOSIS] then the corrected ```python``` block."
            )},
        ]

        self._emit(on_progress, "debugging", attempt, f"Analyzing error (attempt {attempt})…")
        try:
            raw = _call_llm(debug_messages, self.llm_config, max_tokens=4096)
        except ConnectionError as e:
            return failed_code, exec_res, str(e)

        diagnosis  = _extract_diagnosis(raw)
        fixed_code = _extract_code(raw)

        self._emit(on_progress, "executing", attempt, f"Running fixed code (attempt {attempt})…")
        # Execute in shared dir (workflow) or fresh temp dir (single-request)
        if exec_dir:
            new_exec = self._run_code_in_dir(fixed_code, timeout, exec_dir)
        else:
            new_exec = self._run_code(fixed_code, timeout)

        return fixed_code, new_exec, diagnosis

    # ── Verify output ─────────────────────────────────────────────────────────

    def _verify_output(self, original_request: str,
                       exec_res: ExecutionResult) -> tuple[bool, str]:
        """Quick LLM sanity-check: did the output fulfil the request?"""
        files_list = "\n".join(
            [f"  [figure] {p}" for p in exec_res.figures]
            + [f"  [file]   {p}" for p in exec_res.output_files]
        ) or "  (none)"
        msgs = [
            {"role": "system", "content": _VERIFY_SYSTEM},
            {"role": "user", "content": (
                f"## User request\n{original_request}\n\n"
                f"## Stdout\n{exec_res.stdout.strip()[-2000:] or '(empty)'}\n\n"
                f"## Generated files\n{files_list}\n\n"
                "Reply PASS or FAIL: <reason>."
            )},
        ]
        try:
            verdict = _call_llm(msgs, self.llm_config, max_tokens=80).strip()
        except Exception:
            return True, ""
        if verdict.upper().startswith("PASS"):
            return True, ""
        m = re.match(r"FAIL[:\s]+(.*)", verdict, re.IGNORECASE)
        return False, m.group(1).strip() if m else verdict[:120]

    # ── Response builder ──────────────────────────────────────────────────────

    def _build_response(self, exec_res: Optional[ExecutionResult], attempts: int,
                        verify_pass: Optional[bool], verify_note: str,
                        success: Optional[bool] = None) -> str:
        if not exec_res:
            return "Execution failed — no result."
        if success is None:
            success = exec_res.success
        lines = []
        if success:
            lines.append("✓ Code ran successfully" if attempts == 1
                         else f"✓ Code succeeded after {attempts} attempts (auto-debugged)")
        else:
            lines.append(f"✗ Execution failed after {attempts} attempt(s)")
        if exec_res.stdout.strip():
            lines.append("Output:\n" + textwrap.indent(exec_res.stdout.strip(), "  "))
        if exec_res.figures:
            lines.append(f"Generated {len(exec_res.figures)} figure(s)")
        if exec_res.output_files:
            lines.append(f"Generated {len(exec_res.output_files)} file(s)")
        if not exec_res.success:
            err = (exec_res.stderr or exec_res.error or "").strip()
            if err:
                lines.append("Last error:\n" + textwrap.indent(err[-800:], "  "))
        if verify_pass is False:
            lines.append(f"⚠ Output check: {verify_note}")
        return "\n".join(lines)

    # ── Main entry point (single request) ────────────────────────────────────

    def run(
        self,
        user_request: str,
        data_hint: Optional[str] = None,
        max_debug_rounds: int = 4,
        timeout: int = 120,
        run_verify: bool = False,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> CodeRunResult:
        """Generate, execute, debug, and optionally verify Python code."""

        # 1. Profile files mentioned in the request
        file_contexts: List[str] = []
        all_text = user_request + (f"\n{data_hint}" if data_hint else "")
        for fp in _find_file_paths(all_text)[:3]:
            self._emit(on_progress, "analyzing", 0, "Analyzing file(s)…")
            file_contexts.append(
                _format_file_context(_profile_file(fp, self.project_root,
                                                    self.python_executable)))

        # 2. Build user message
        msg = user_request
        if data_hint:
            msg += f"\n\nData path: {data_hint}"
        if file_contexts:
            msg += "\n\n" + "\n\n".join(file_contexts)
        self._history.append({"role": "user", "content": msg})

        # 3. Build skill + RAG context (queried once; forwarded to debug loop)
        try:
            skill_ctx, rag_ctx = _build_ctx(
                user_request, max_skill_chars=12000, max_rag_chars=4000, top_k=5)
        except Exception:
            skill_ctx, rag_ctx = "", ""

        system = _CODEGEN_SYSTEM
        if skill_ctx:
            n = skill_ctx.count("### 技能：")
            system += "\n\n## Relevant skill docs\n" + skill_ctx
            if n > 1:
                system += ("\n\n## How to combine these skills\n"
                           "Identify which functions/patterns from each skill apply "
                           "and integrate them into a single coherent script.")
        if rag_ctx:
            system += ("\n\n## Knowledge Base (RAG)\n" + rag_ctx
                       + "\n\nUse the above to verify correct API usage before writing code.")

        messages = [{"role": "system", "content": system}] + \
                   [m for m in self._history if m["role"] != "system"]

        # 4. Plan (non-fatal)
        plan: List[str] = []
        self._emit(on_progress, "planning", 0, "Planning…")
        try:
            plan = _extract_plan(_call_llm(
                [{"role": "system", "content": _PLAN_SYSTEM},
                 {"role": "user", "content":
                  f"Request: {user_request}\n\n" + "\n".join(file_contexts)
                  + "\n\nList the execution steps."}],
                self.llm_config, max_tokens=400))
        except Exception:
            pass
        if plan:
            self._emit(on_progress, "planning", 0, "Plan: " + " → ".join(plan))

        # 5. Generate code
        self._emit(on_progress, "generating", 0, "Generating code…")
        try:
            code = _extract_code(_call_llm(messages, self.llm_config))
        except ConnectionError as e:
            return CodeRunResult(success=False, response=str(e), code="", exec_result=None)

        # 6. First execution
        self._emit(on_progress, "executing", 0, "Executing code…")
        exec_res = self._run_code(code, timeout)
        debug_trace: List[DebugAttempt] = []
        attempt = 0

        # 7. Debug loop — skill_ctx forwarded so debugger knows the same APIs
        while not self._execution_success(exec_res) and attempt < max_debug_rounds:
            attempt += 1
            err_summary = f"{exec_res.stdout}\n{exec_res.stderr}\n{exec_res.error}".strip()
            debug_trace.append(DebugAttempt(
                attempt=attempt, diagnosis="", code=code,
                error=err_summary, stdout=exec_res.stdout, success=False))

            try:
                _, dbg_rag = _build_ctx(f"{user_request} {err_summary[:400]}",
                                        max_skill_chars=1, max_rag_chars=3000, top_k=3)
            except Exception:
                dbg_rag = ""

            code, exec_res, diagnosis = self._debug_and_fix(
                original_request=user_request, failed_code=code,
                exec_res=exec_res, attempt=attempt, timeout=timeout,
                on_progress=on_progress, file_contexts=file_contexts,
                skill_ctx=skill_ctx, extra_rag_ctx=dbg_rag,
                # exec_dir=None → run in fresh temp dir (single-request mode)
            )
            debug_trace[-1].diagnosis = diagnosis

            if self._execution_success(exec_res):
                debug_trace.append(DebugAttempt(
                    attempt=attempt, diagnosis=f"Fixed: {diagnosis}",
                    code=code, error="", stdout=exec_res.stdout, success=True))
                self._emit(on_progress, "executing", attempt,
                           f"✓ Fixed after {attempt} debug round(s)")
                break
            self._emit(on_progress, "debugging", attempt,
                       f"Attempt {attempt} still failing, retrying…")

        # 8. Update conversation history
        final_success = self._execution_success(exec_res)
        summary = "Execution " + ("succeeded." if final_success else "failed.")
        if exec_res and exec_res.figures:
            summary += "\nFigures: " + str([Path(f).name for f in exec_res.figures])
        if exec_res and exec_res.output_files:
            summary += "\nFiles: "   + str([Path(f).name for f in exec_res.output_files])
        if exec_res and exec_res.stdout.strip():
            clean = "\n".join(l for l in exec_res.stdout.splitlines()
                              if not l.startswith("[FIGURE]")
                              and not l.startswith("[GMT_SCRIPT]")).strip()
            if clean:
                summary += f"\nOutput (truncated):\n{clean[:400]}"
        if exec_res and not final_success:
            err = (exec_res.stderr or exec_res.error or "").strip()
            if err:
                summary += f"\nError:\n{err[:300]}"
        self._history.append({
            "role": "assistant",
            "content": f"```python\n{code}\n```\n\n[Result] {summary}",
        })
        if exec_res:
            self._last_exec_dir = exec_res.exec_dir

        # 9. Verify (optional)
        verify_pass, verify_note = None, ""
        if run_verify and exec_res and exec_res.success:
            self._emit(on_progress, "verifying", attempt, "Verifying output…")
            verify_pass, verify_note = self._verify_output(user_request, exec_res)

        # 10. Save final script
        script_path = ""
        if code:
            try:
                import os, tempfile
                sd = (exec_res.exec_dir if (exec_res and exec_res.exec_dir)
                      else tempfile.mkdtemp(prefix="sage_script_"))
                script_path = os.path.join(sd, "analysis.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(f"# Generated by SeismicX — {user_request[:80]}\n"
                            f"# Attempts: {attempt+1}\n\n" + code)
            except Exception:
                pass

        total = attempt + 1
        response = self._build_response(exec_res, total, verify_pass, verify_note, final_success)
        self._emit(on_progress, "done", attempt, response)

        return CodeRunResult(
            success=final_success, response=response, code=code,
            exec_result=exec_res, attempts=total, debug_trace=debug_trace,
            verify_pass=verify_pass, verify_note=verify_note,
            plan=plan, script_path=script_path,
        )

    # ── Workflow helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _topo_sort(steps: List[Dict]) -> List[Dict]:
        """Topological order based on depends_on lists."""
        order: List[Dict] = []
        remaining = {s["id"]: s for s in steps}
        done: set = set()
        while remaining:
            ready = [sid for sid, s in remaining.items()
                     if all(d in done for d in s.get("depends_on", []))]
            if not ready:
                order.extend(remaining[s["id"]] for s in steps if s["id"] in remaining)
                break
            for s in steps:
                if s["id"] in ready and s["id"] in remaining:
                    order.append(remaining.pop(s["id"]))
                    done.add(s["id"])
        return order

    def _build_step_prompt(
        self,
        step: Dict, workflow: Dict,
        step_index: int, steps_total: int,
        available_files: List[str],
        completed_steps: List[StepResult],
        user_request: str,
    ) -> tuple[str, str, str]:
        """
        Build (system_content, user_message, skill_ctx) for a single workflow step.

        skill_ctx is returned so the caller can forward it to the debug loop
        without a second RAG API call.
        """
        step_id  = step["id"]
        skill_nm = step.get("skill", "")
        desc     = step["description"]

        # Per-step skill + RAG context
        try:
            skill_ctx, rag_ctx = _build_ctx(
                f"{skill_nm} {desc} {user_request}",
                max_skill_chars=8000, max_rag_chars=2000, top_k=3)
        except Exception:
            skill_ctx, rag_ctx = "", ""

        system = _CODEGEN_SYSTEM
        if skill_ctx:
            system += f"\n\n## Step skill documentation\n{skill_ctx}"
        if rag_ctx:
            system += (f"\n\n## Knowledge Base (RAG)\n{rag_ctx}\n\n"
                       "Use the above for correct API and parameter usage.")

        # Completed steps summary — include key stdout so next step has context
        prev = ""
        if completed_steps:
            lines = []
            for sr in completed_steps:
                icon  = "✓" if sr.success else "✗"
                flist = ", ".join(Path(f).name for f in sr.figures + sr.output_files)
                out   = sr.stdout.strip()[:400] if sr.stdout.strip() else ""
                entry = f"  {icon} {sr.step_id} [{sr.skill}]: {sr.description}"
                if flist:
                    entry += f"\n     Output files: {flist}"
                if out:
                    entry += f"\n     Key output:\n       {out}"
                lines.append(entry)
            prev = "## Completed steps\n" + "\n".join(lines)

        files_str = ""
        if available_files:
            files_str = ("## Files in working directory\n"
                         + "\n".join(f"  {f}" for f in available_files[:20]))

        guide = workflow.get("guide", "")
        guide_excerpt = (guide[:2000] + "\n...(truncated)" if len(guide) > 2000 else guide)

        user_msg = (
            f"# Workflow: {workflow['name']} — {workflow['title']}\n"
            f"# Step {step_index+1}/{steps_total}: [{step_id}] {desc}\n"
            f"# Skill: {skill_nm or '(general)'}\n\n"
        )
        if user_request:
            user_msg += f"## User request\n{user_request}\n\n"
        if prev:
            user_msg += prev + "\n\n"
        if files_str:
            user_msg += files_str + "\n\n"
        if guide_excerpt:
            user_msg += f"## Workflow guide (excerpt)\n{guide_excerpt}\n\n"
        user_msg += (f"## Current task\n"
                     f"Generate Python/Bash code for step `{step_id}`: {desc}\n\n"
                     "Output the code block only.")

        return system, user_msg, skill_ctx

    # ── Workflow runner ───────────────────────────────────────────────────────

    def run_workflow(
        self,
        workflow_name: str,
        user_request: str = "",
        data_hint: Optional[str] = None,
        max_debug_rounds: int = 3,
        timeout: int = 120,
        skip_on_failure: bool = False,
        on_progress: Optional[Callable[[Dict], None]] = None,
    ) -> WorkflowRunResult:
        """
        Execute a workflow step-by-step.

        Per-step behaviour
        ------------------
        1. Build prompt with fresh skill+RAG context for this step.
        2. Generate code via LLM.
        3. Execute in the shared working directory.
        4. On failure: debug loop with skill_ctx forwarded + exec_dir set
           so fixed code runs in shared_dir directly (no double-execution).
        5. After exit-0: semantic output check (_step_output_ok) for
           silent failures (e.g. plot step produced no figure).
        6. Append step outcome (code + key stdout) to shared LLM history.
        """

        # 0. Load workflow
        try:
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from seismo_skill.workflow_runner import load_workflow
            workflow = load_workflow(workflow_name)
        except Exception as e:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title="",
                success=False, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"Cannot load workflow '{workflow_name}': {e}")

        if workflow is None:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title="",
                success=False, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"Workflow '{workflow_name}' not found")

        steps_raw = workflow.get("steps", [])
        if not steps_raw:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title=workflow["title"],
                success=True, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response=f"Workflow '{workflow_name}' has no steps defined")

        # 1. Topological sort + state
        ordered_steps    = self._topo_sort(steps_raw)
        steps_total      = len(ordered_steps)
        step_results:     List[StepResult] = []
        all_figures:      List[str]        = []
        all_output_files: List[str]        = []
        failed_ids:       set              = set()
        shared_dir:       Optional[str]    = None
        wf_history:       List[Dict]       = []

        def _emit_wf(phase, step_id, step_n, msg):
            if on_progress:
                try:
                    on_progress({"phase": phase, "step_id": step_id,
                                 "step_n": step_n, "total": steps_total, "message": msg})
                except Exception:
                    pass

        # 2. Execute steps
        for step_n, step in enumerate(ordered_steps):
            step_id  = step["id"]
            skill_nm = step.get("skill", "")
            desc     = step["description"]
            deps     = step.get("depends_on", [])

            # Skip if dependency failed
            bad_deps = [d for d in deps if d in failed_ids]
            if bad_deps:
                _emit_wf("workflow_step", step_id, step_n,
                         f"Skipping {step_id} (dependency failed: {', '.join(bad_deps)})")
                step_results.append(StepResult(
                    step_id=step_id, skill=skill_nm, description=desc,
                    success=False, code="", skipped=True,
                    diagnosis=f"Dependency failed: {', '.join(bad_deps)}"))
                failed_ids.add(step_id)
                if not skip_on_failure:
                    break
                continue

            # Discover files already in the shared workspace
            available_files: List[str] = []
            if shared_dir and Path(shared_dir).exists():
                try:
                    available_files = sorted(
                        str(p) for p in Path(shared_dir).iterdir()
                        if p.is_file() and not p.name.startswith("run."))
                except Exception:
                    pass

            _emit_wf("workflow_step", step_id, step_n,
                     f"[{step_n+1}/{steps_total}] Generating step {step_id}…")

            # Build prompt — returns skill_ctx for the debug loop
            completed = [r for r in step_results if r.success]
            req_with_data = user_request + (f"\nData: {data_hint}" if data_hint else "")
            sys_content, user_msg, skill_ctx = self._build_step_prompt(
                step=step, workflow=workflow,
                step_index=step_n, steps_total=steps_total,
                available_files=available_files, completed_steps=completed,
                user_request=req_with_data)

            messages = ([{"role": "system", "content": sys_content}]
                        + wf_history
                        + [{"role": "user", "content": user_msg}])

            # Generate code
            try:
                code = _extract_code(_call_llm(messages, self.llm_config))
            except ConnectionError as e:
                step_results.append(StepResult(
                    step_id=step_id, skill=skill_nm, description=desc,
                    success=False, code="", diagnosis=str(e)))
                failed_ids.add(step_id)
                if not skip_on_failure:
                    break
                continue

            # Execute in shared dir
            _emit_wf("workflow_step", step_id, step_n,
                     f"[{step_n+1}/{steps_total}] Executing step {step_id}…")
            exec_res  = self._run_code_in_dir(code, timeout, shared_dir)
            attempt   = 0
            diagnosis = ""
            if shared_dir is None and exec_res.exec_dir:
                shared_dir = exec_res.exec_dir

            # Debug loop
            # The loop handles both runtime failures AND semantic output failures
            # (_step_output_ok). skill_ctx is forwarded; exec_dir avoids double-execution.
            while attempt < max_debug_rounds:
                exec_ok = self._execution_success(exec_res)
                if exec_ok:
                    out_ok, out_reason = self._step_output_ok(desc, exec_res)
                    if out_ok:
                        break   # genuine success
                    # Synthesise a failure so the debugger sees what went wrong
                    _emit_wf("workflow_step", step_id, step_n,
                             f"[{step_n+1}/{steps_total}] Output check failed "
                             f"({out_reason}), re-debugging…")
                    from dataclasses import replace as _dc_replace
                    exec_res = _dc_replace(
                        exec_res, success=False,
                        stderr=(exec_res.stderr or "") + f"\n[OUTPUT CHECK FAILED] {out_reason}")

                attempt += 1
                err_summary = f"{exec_res.stdout}\n{exec_res.stderr}\n{exec_res.error}".strip()
                _emit_wf("workflow_step", step_id, step_n,
                         f"[{step_n+1}/{steps_total}] Debugging {step_id} (round {attempt})…")

                try:
                    _, dbg_rag = _build_ctx(
                        f"{skill_nm} {desc} {err_summary[:300]}",
                        max_skill_chars=1, max_rag_chars=2000, top_k=3)
                except Exception:
                    dbg_rag = ""

                # _debug_and_fix runs the fix in shared_dir — no separate re-run needed
                code, exec_res, diagnosis = self._debug_and_fix(
                    original_request=f"{desc}\n{user_request}",
                    failed_code=code, exec_res=exec_res,
                    attempt=attempt, timeout=timeout, on_progress=None,
                    file_contexts=[f"Available: {f}" for f in available_files[:5]],
                    skill_ctx=skill_ctx,     # ← forwarded skill docs
                    extra_rag_ctx=dbg_rag,
                    exec_dir=shared_dir,     # ← no double-execution
                )
                if shared_dir is None and exec_res.exec_dir:
                    shared_dir = exec_res.exec_dir

            step_success = self._execution_success(exec_res)
            step_figs    = exec_res.figures      if exec_res else []
            step_files   = exec_res.output_files if exec_res else []
            all_figures.extend(step_figs)
            all_output_files.extend(step_files)

            sr = StepResult(
                step_id=step_id, skill=skill_nm, description=desc,
                success=step_success, code=code,
                stdout=(exec_res.stdout or "")[:2000] if exec_res else "",
                stderr=(exec_res.stderr or "")[:1000] if exec_res else "",
                figures=step_figs, output_files=step_files,
                attempts=attempt + 1, diagnosis=diagnosis)
            step_results.append(sr)

            if not step_success:
                failed_ids.add(step_id)
                _emit_wf("step_done", step_id, step_n,
                         f"✗ Step {step_id} failed ({attempt+1} attempts)")
                if not skip_on_failure:
                    break
            else:
                out_tag = f", {len(step_figs)} figure(s)" if step_figs else ""
                _emit_wf("step_done", step_id, step_n, f"✓ Step {step_id} done{out_tag}")

                # Append to shared history for next step's LLM context
                key_out = exec_res.stdout.strip()[:1000] if exec_res else ""
                step_summary = (
                    f"Step {step_id} done."
                    + (f" Files: {', '.join(Path(f).name for f in step_figs+step_files)}"
                       if step_figs + step_files else "")
                    + (f"\nKey output:\n{key_out}" if key_out else ""))
                wf_history.append({"role": "user", "content": user_msg})
                wf_history.append({
                    "role": "assistant",
                    "content": f"```python\n{code}\n```\n\n[Step result] {step_summary}"})

        # 3. Build summary
        steps_done    = sum(1 for r in step_results if r.success)
        wf_success    = steps_done == steps_total and bool(step_results)
        skipped_count = sum(1 for r in step_results if r.skipped)

        lines = [f"Workflow **{workflow['name']}** — {workflow['title']}",
                 f"Progress: {steps_done}/{steps_total} steps"
                 + (f", {skipped_count} skipped" if skipped_count else "")]
        for sr in step_results:
            icon  = "✓" if sr.success else ("↷" if sr.skipped else "✗")
            extra = f" ({sr.attempts} attempts)" if sr.attempts > 1 and not sr.skipped else ""
            lines.append(f"  {icon} [{sr.step_id}] {sr.description}{extra}")
        if all_figures:
            lines.append(f"Figures: {len(all_figures)}")
        if all_output_files:
            lines.append(f"Output files: {len(all_output_files)}")
        if shared_dir:
            lines.append(f"Working directory: {shared_dir}")

        response = "\n".join(lines)
        _emit_wf("workflow_done", "", steps_total, response)
        if shared_dir:
            self._last_exec_dir = shared_dir

        return WorkflowRunResult(
            workflow_name=workflow_name, workflow_title=workflow["title"],
            success=wf_success, steps_total=steps_total, steps_done=steps_done,
            step_results=step_results, all_figures=all_figures,
            all_output_files=all_output_files, response=response,
            exec_dir=shared_dir or "")

    # ── Session management ────────────────────────────────────────────────────

    def reset(self):
        self._history       = [{"role": "system", "content": _CODEGEN_SYSTEM}]
        self._last_exec_dir = None


# ---------------------------------------------------------------------------
# Singleton / factory
# ---------------------------------------------------------------------------

_engine_instance: Optional[CodeEngine] = None


def get_code_engine(llm_config: Optional[Dict] = None) -> CodeEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CodeEngine(llm_config)
    return _engine_instance


def reset_code_engine():
    global _engine_instance
    if _engine_instance:
        _engine_instance.reset()


# ---------------------------------------------------------------------------
# Self-tests  (python -m seismo_code.code_engine --test)
# ---------------------------------------------------------------------------

def _run_tests() -> bool:
    passed = failed = 0

    def ok(name):
        nonlocal passed; passed += 1; print(f"  ✓ {name}")

    def fail(name, e):
        nonlocal failed; failed += 1; print(f"  ✗ {name}: {e}")

    print("=" * 60)
    print("CodeEngine — self-tests (no LLM required)")
    print("=" * 60)

    # 1. _extract_code
    print("\n[1] _extract_code")
    try:
        c = _extract_code("```python\nprint('hi')\n```")
        assert "print('hi')" in c and "# lang:python" in c; ok("python block")
        b = _extract_code("```bash\necho hi\n```")
        assert "echo hi" in b and "# lang:bash" in b; ok("bash block")
        r = _extract_code("```\nsome code\n```")
        assert "some code" in r; ok("bare block")
    except Exception as e:
        fail("_extract_code", e)

    # 2. _is_bash_code
    print("\n[2] _is_bash_code")
    try:
        assert _is_bash_code("# lang:bash\necho hi");              ok("lang:bash tag")
        assert _is_bash_code("#!/bin/bash\necho hi");              ok("shebang")
        assert _is_bash_code("gmt begin map PNG\ngmt end");        ok("gmt begin")
        assert not _is_bash_code("import numpy as np");            ok("Python not bash")
    except Exception as e:
        fail("_is_bash_code", e)

    # 3. _pre_sanitize
    print("\n[3] _pre_sanitize")
    try:
        assert "pass" in _pre_sanitize("plt.show()\nprint('x')");  ok("plt.show() neutralised")
        assert "matplotlib.use" in _pre_sanitize("import cartopy"); ok("Agg injected for cartopy")
        assert "import pandas" in _pre_sanitize("df = pd.read_csv('f')"); ok("pandas auto-import")
        bash = "# lang:bash\necho hi"
        assert _pre_sanitize(bash) == bash;                         ok("bash unchanged")
    except Exception as e:
        fail("_pre_sanitize", e)

    # 4. _extract_plan
    print("\n[4] _extract_plan")
    try:
        steps = _extract_plan("[PLAN]\n1. Load CSV\n2. Filter\n3. Plot\n")
        assert len(steps) == 3 and "Load" in steps[0]; ok(f"{len(steps)} steps parsed")
    except Exception as e:
        fail("_extract_plan", e)

    # 5. _extract_diagnosis
    print("\n[5] _extract_diagnosis")
    try:
        d = _extract_diagnosis("[DIAGNOSIS]\nMissing column.\n\n```python\npass\n```")
        assert "Missing" in d; ok("diagnosis extracted")
        d2 = _extract_diagnosis("Some error text"); assert d2; ok("fallback works")
    except Exception as e:
        fail("_extract_diagnosis", e)

    # 6. _find_file_paths
    print("\n[6] _find_file_paths")
    try:
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        assert tmp in _find_file_paths(f"process {tmp} please"); ok("existing path found")
        assert not _find_file_paths("/nonexistent/path.csv");     ok("missing path ignored")
        os.unlink(tmp)
    except Exception as e:
        fail("_find_file_paths", e)

    # 7. CodeEngine instantiation
    print("\n[7] CodeEngine")
    try:
        eng = CodeEngine(llm_config={"provider": "ollama", "model": "x",
                                     "api_base": "http://localhost:11434"})
        assert not eng.is_llm_available(); ok("instantiation + is_llm_available()")
    except Exception as e:
        fail("CodeEngine", e)

    # 8. _execution_success heuristics
    print("\n[8] _execution_success")
    try:
        from .safe_executor import ExecutionResult
        eng = CodeEngine(llm_config={})
        clean = ExecutionResult(success=True, stdout="ok", stderr="", error="",
                                figures=[], output_files=[], exec_dir="")
        assert eng._execution_success(clean); ok("clean result → success")
        bad = ExecutionResult(
            success=True,
            stdout="Traceback (most recent call last):\nValueError: bad",
            stderr="", error="", figures=[], output_files=[], exec_dir="")
        assert not eng._execution_success(bad); ok("traceback in stdout → failure")
    except Exception as e:
        fail("_execution_success", e)

    print(f"\n{'='*60}")
    print(f"{'All ' + str(passed) + ' tests passed.' if failed == 0 else str(passed) + ' passed, ' + str(failed) + ' FAILED.'}")
    print("=" * 60)
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json as _json

    ap = argparse.ArgumentParser(description="SeismicX code engine CLI")
    ap.add_argument("--test",   action="store_true", help="Run self-tests (no LLM needed)")
    ap.add_argument("--status", action="store_true", help="Print engine/LLM status")
    args = ap.parse_args()

    if args.test:
        sys.exit(0 if _run_tests() else 1)

    if args.status:
        eng = get_code_engine()
        print(_json.dumps({
            "llm_available": eng.is_llm_available(),
            "provider": eng.llm_config.get("provider"),
            "model":    eng.llm_config.get("model"),
            "api_base": eng.llm_config.get("api_base"),
        }, indent=2))
        sys.exit(0)

    ap.print_help()
