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
import subprocess
import sys
import textwrap
import urllib.request
import hashlib
import importlib
import inspect
import ast
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .safe_executor import ExecutionResult, execute_bash, execute_code
from .ce_prompts import (
    _CODEGEN_SYSTEM,
    _DEBUG_SYSTEM,
    _ENGINEERING_PLAN_SYSTEM,
    _PLAN_SYSTEM,
    _VERIFY_SYSTEM,
)
from .ce_utils import (
    CodeRunResult, DebugAttempt, StepResult, WorkflowRunResult,
    _call_llm, _extract_code, _is_bash_code, _extract_diagnosis,
    _extract_plan, _find_file_paths, _profile_file, _format_file_context,
    _pre_sanitize, CodeExecutionCancelled,
)
from .repo_intelligence import SAGE_EDITING_GUIDE, build_repo_intelligence

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
        self._repo_baseline_files: Dict[str, str] = {}
        self._current_engineering_plan_path: Optional[str] = None

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

    @staticmethod
    def _raise_if_cancelled(cancel_event=None):
        if cancel_event is not None and cancel_event.is_set():
            raise CodeExecutionCancelled("cancelled")

    # ── Executors ─────────────────────────────────────────────────────────────

    def _run_code(
        self,
        code: str,
        timeout: int,
        cancel_event=None,
        on_progress: Optional[Callable] = None,
        attempt: int = 0,
    ) -> ExecutionResult:
        """Execute Python or bash code."""
        self._raise_if_cancelled(cancel_event)
        progress_cb = (
            lambda msg: self._emit(on_progress, "executing", attempt, msg)
            if on_progress else None
        )
        if _is_bash_code(code):
            clean = re.sub(r"^#\s*lang:bash\s*\n", "", code, count=1)
            return execute_bash(clean, project_root=self.project_root,
                                timeout=timeout, keep_dir=True,
                                progress_cb=progress_cb,
                                cancel_event=cancel_event)
        return execute_code(_pre_sanitize(code), project_root=self.project_root,
                            timeout=timeout, keep_dir=True,
                            python_executable=self.python_executable,
                            progress_cb=progress_cb,
                            cancel_event=cancel_event)

    def _run_code_in_dir(self, code: str, timeout: int,
                         shared_dir: Optional[str] = None,
                         cancel_event=None,
                         on_progress: Optional[Callable] = None,
                         attempt: int = 0) -> ExecutionResult:
        """Execute code inside a pre-existing shared directory."""
        self._raise_if_cancelled(cancel_event)
        progress_cb = (
            lambda msg: self._emit(on_progress, "executing", attempt, msg)
            if on_progress else None
        )
        if _is_bash_code(code):
            clean = re.sub(r"^#\s*lang:bash\s*\n", "", code, count=1)
            extra_env = (
                {"SAGE_OUTDIR": shared_dir, "SAGE_WORKSPACE_ROOT": self.project_root}
                if shared_dir else None
            )
            return execute_bash(clean, project_root=self.project_root,
                                timeout=timeout, keep_dir=True,
                                extra_env=extra_env,
                                progress_cb=progress_cb,
                                cancel_event=cancel_event)
        clean = _pre_sanitize(code)
        extra_env = None
        if shared_dir:
            preamble = (f"import os as _wf_os\n"
                        f"_wf_os.makedirs({shared_dir!r}, exist_ok=True)\n"
                        f"_wf_os.environ['SAGE_OUTDIR'] = {shared_dir!r}\n")
            clean     = preamble + clean
            extra_env = {
                "SAGE_OUTDIR": shared_dir,
                "SAGE_WORKSPACE_ROOT": self.project_root,
            }
        return execute_code(clean, project_root=self.project_root,
                            timeout=timeout, keep_dir=True,
                            extra_env=extra_env,
                            python_executable=self.python_executable,
                            progress_cb=progress_cb,
                            cancel_event=cancel_event)

    @staticmethod
    def _placeholder_path_reason(code: str) -> str:
        """Return a reason when generated code contains fake input paths."""
        patterns = [
            r"/Users/(?:your_username|username|yourname|you)/",
            r"/path/to/",
            r"/data/data\.sac\b",
            r"/data/wave\.mseed\b",
            r"your_(?:file|path|data)",
            r"replace with your actual",
        ]
        for pat in patterns:
            if re.search(pat, code, re.I):
                return f"generated code contains placeholder path matching `{pat}`"
        return ""

    # ── Repository context helpers ───────────────────────────────────────────

    @staticmethod
    def _primary_user_request(text: str) -> str:
        """Strip appended soft context so routing/validation sees the actual request."""
        value = text or ""
        markers = [
            "\n\n===== Long-term user profile",
            "\n===== Long-term user profile",
            "\n\n===== Project context",
            "\n\n## Repository Context",
            "\n\n## Data/file context",
        ]
        cut = len(value)
        for marker in markers:
            idx = value.find(marker)
            if idx >= 0:
                cut = min(cut, idx)
        return value[:cut].strip()

    @staticmethod
    def _looks_like_repo_task(text: str) -> bool:
        """Heuristic for repository-editing requests."""
        original = CodeEngine._primary_user_request(text)
        t = original.lower()
        file_hints = [".py", ".js", ".ts", ".html", ".css", ".md", ".json", ".toml"]
        if any(h in t for h in file_hints):
            return True

        repo_nouns = [
            "sage", "codeengine", "coding agent", "router", "route", "api",
            "frontend", "backend", "readme", "gitignore", "config", "web_app",
            "seismo_code", "seismo_skill", "tests", "docs", "repository", "repo",
            "codebase", "module", "function", "gui", "ui",
            "仓库", "代码库", "项目", "路由", "接口", "前端", "后端", "配置",
            "模块", "函数", "测试", "文档", "结构文档", "技能", "界面", "skill",
        ]
        actions = [
            "bug", "fix", "refactor", "implement", "integrate", "update", "delete",
            "remove", "add", "support", "optimize", "modify", "change",
            "修复", "报错", "优化", "实现", "集成", "更新", "删除", "添加",
            "支持", "修改", "改成", "改为", "加入", "加上", "删掉", "清理",
            "放在", "写到",
        ]
        locating = bool(re.search(r"where|locate|find|定位|在哪里|哪个文件|位置", original, re.I))
        has_repo_noun = any(k in t for k in repo_nouns)
        has_action = any(k in t for k in actions)
        return has_repo_noun and (has_action or locating)

    def _repo_file_list(self, limit: int = 260) -> List[str]:
        """Return tracked/worktree files for repository context."""
        try:
            proc = subprocess.run(
                ["rg", "--files"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            files = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        except Exception:
            files = []
        skip_parts = {
            ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
            "third_party/aider", "web_app/outputs", "web_app/uploads", "seismo_rag",
            ".aider.tags.cache.v4",
        }
        filtered = []
        for fp in files:
            if any(part in fp for part in skip_parts):
                continue
            filtered.append(fp)
        return filtered[:limit]

    def _repo_worktree_files(self) -> set[str]:
        """Return modified/untracked worktree files."""
        try:
            proc = subprocess.run(
                ["git", "status", "--short", "--untracked-files=all"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            files = set()
            for line in proc.stdout.splitlines():
                rel = line[3:].strip() if len(line) > 3 else ""
                if " -> " in rel:
                    rel = rel.split(" -> ", 1)[1].strip()
                if rel:
                    files.add(rel)
            return files
        except Exception:
            return set()

    def _repo_file_digest(self, rel: str) -> str:
        path = Path(self.project_root) / rel
        if not path.is_file():
            return ""
        try:
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""

    def _repo_snapshot(self) -> Dict[str, str]:
        return {rel: self._repo_file_digest(rel) for rel in self._repo_worktree_files()}

    @staticmethod
    def _repo_terms(request: str, limit: int = 10) -> List[str]:
        """Extract search terms for codebase discovery."""
        text = request or ""
        quoted = re.findall(r"`([^`]{2,80})`|['\"]([A-Za-z_][\w.:-]{2,80})['\"]", text)
        terms = [a or b for a, b in quoted if (a or b)]
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", text):
            if token.lower() in {
                "the", "and", "for", "with", "that", "this", "code", "file",
                "function", "class", "test", "实现", "修复", "代码", "文件", "函数",
            }:
                continue
            if token not in terms:
                terms.append(token)
        return terms[:limit]

    def _repo_rg_hits(self, request: str, max_hits: int = 80) -> List[str]:
        """Run targeted rg searches and return compact file:line hits."""
        hits: List[str] = []
        for term in self._repo_terms(request):
            try:
                proc = subprocess.run(
                    ["rg", "-n", "--glob", "!third_party/aider/**", "--glob", "!seismo_rag/**", term],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                continue
            for line in proc.stdout.splitlines():
                if line and line not in hits:
                    hits.append(line[:260])
                    if len(hits) >= max_hits:
                        return hits
        return hits

    def _repo_symbol_index(self, max_lines: int = 160) -> List[str]:
        """Collect function/class/route symbols for fast location awareness."""
        patterns = [
            r"^\s*(def|class)\s+[A-Za-z_][A-Za-z0-9_]*",
            r"^\s*@(?:bp|app)\.route\(",
            r"^\s*(async\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*",
            r"^\s*(export\s+)?(const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=",
        ]
        symbols: List[str] = []
        for pat in patterns:
            try:
                proc = subprocess.run(
                    ["rg", "-n", "--glob", "!third_party/aider/**", "--glob", "!seismo_rag/**", pat],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=6,
                    check=False,
                )
            except Exception:
                continue
            for line in proc.stdout.splitlines():
                if line and line not in symbols:
                    symbols.append(line[:220])
                    if len(symbols) >= max_lines:
                        return symbols
        return symbols

    @staticmethod
    def _score_repo_file(path: str, request: str) -> int:
        text = (request or "").lower()
        p = path.lower()
        score = 0
        if Path(path).name.lower() in text:
            score += 8
        for part in p.replace("/", " ").replace("_", " ").replace("-", " ").split():
            if len(part) >= 3 and part in text:
                score += 2
        for hint, bonus in [
            ("config", 3), ("llm", 3), ("code", 3), ("agent", 3), ("chat", 3),
            ("skill", 3), ("knowledge", 3), ("readme", 4), ("route", 3),
        ]:
            if hint in text and hint in p:
                score += bonus
        return score

    def _build_repo_context(self, request: str, max_files: int = 10, max_chars: int = 18000) -> str:
        """Built-in compact repository map plus relevant snippets."""
        files = self._repo_file_list()
        if not files:
            return ""
        repo_intel = build_repo_intelligence(self.project_root, request, files)
        rg_hits = self._repo_rg_hits(request)
        hit_files = []
        for hit in rg_hits:
            rel = hit.split(":", 1)[0]
            if rel and rel not in hit_files:
                hit_files.append(rel)
        ranked = sorted(files, key=lambda f: self._score_repo_file(f, request), reverse=True)
        selected = [f for f in repo_intel.ranked_files if f in files][:max_files]
        selected += [f for f in hit_files if f not in selected]
        selected += [f for f in ranked if self._score_repo_file(f, request) > 0 and f not in selected]
        selected = selected[:max_files]
        if not selected:
            selected = ranked[:min(6, len(ranked))]

        parts = [
            "## Repository Context",
            "Project root: " + self.project_root,
            "Relevant files selected from `rg --files` (not exhaustive):",
            "\n".join(f"- {f}" for f in ranked[:80]),
            "\n## Targeted `rg` hits",
            "\n".join(f"- {h}" for h in rg_hits[:80]) or "(no direct text hits)",
            "\n## Symbol / route index",
            "\n".join(f"- {s}" for s in self._repo_symbol_index()[:120]),
        ]
        if repo_intel.available and repo_intel.repo_map:
            parts.extend([
                "\n## SAGE Repo Map",
                repo_intel.repo_map,
                "\n## SAGE-ranked files",
                "\n".join(f"- {f}" for f in repo_intel.ranked_files[:80]) or "(none)",
            ])
        else:
            parts.extend([
                "\n## SAGE Repo Map",
                f"(unavailable; falling back to rg/symbol context: {repo_intel.error or 'unknown error'})",
            ])
        parts.append("\n## Relevant File Snippets")
        used = sum(len(p) for p in parts)
        for rel in selected:
            path = Path(self.project_root) / rel
            if not path.is_file() or path.stat().st_size > 220_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            snippet = "\n".join(text.splitlines()[:140])
            block = f"\n### {rel}\n```text\n{snippet}\n```"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        parts.append(
            "\n## Repository Editing Rules\n"
            "- If the task is to change the active codebase, generate a Python script that edits files under the project root shown above.\n"
            "- Use the repo map and ranked files first, then targeted `rg` hits, to choose files.\n"
            "- You MAY edit multiple files and create new modules/tests when the task requires it.\n"
            "- For non-trivial behavior changes, add, insert, update, or delete focused unit tests under `tests/`.\n"
            "- Deleting test code is allowed only when the test is obsolete, asserts the wrong behavior, or is replaced by equivalent/better coverage; print the reason.\n"
            "- Before editing, print `[SAGE_AGENT] located <path>: <reason>` for the files you selected from the repo map/rg hits.\n"
            "- After editing, print `[SAGE_CHANGED] <path>` for every changed file.\n"
            "- Prefer structured file edits with pathlib and small helper functions; do not rewrite unrelated files.\n"
            "- Run validation inside the script with subprocess using cwd=PROJECT_ROOT: py_compile for changed Python files and targeted pytest for changed or related tests.\n"
            "- For broad project-level requests or explicit full-test/build requests, run the relevant full suite too (`pytest`, `npm test`, `npm run build`, or the project's documented command) and print the command/result.\n"
            "- Python behavior changes must add/update focused unit tests or run existing focused tests found by filename/API relation.\n"
            "- Print changed file paths and include `[SAGE_TEST]` checks such as py_compile, pytest, syntax checks, or targeted API checks.\n"
            "- When locating behavior, use the provided `rg` hits and symbol index before editing.\n"
            "- Do not modify vendored or dependency directories such as `third_party/aider`, `node_modules`, or `.venv` unless the user explicitly asks."
        )
        parts.append(SAGE_EDITING_GUIDE)
        return "\n\n".join(parts)

    def _git_diff_summary(self) -> str:
        try:
            proc = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            return (proc.stdout or "").strip()
        except Exception:
            return ""

    def _repo_new_changed_files(self) -> List[str]:
        current = {rel: self._repo_file_digest(rel) for rel in self._repo_worktree_files()}
        changed = [
            rel for rel, digest in current.items()
            if self._repo_baseline_files.get(rel) != digest
        ]
        return sorted(changed)

    @staticmethod
    def _repo_behavior_change_request(text: str) -> bool:
        primary = CodeEngine._primary_user_request(text)
        return bool(re.search(
            r"\b(fix|implement|add|update|refactor|change|support|create|delete|remove)\b"
            r"|修复|实现|添加|更新|重构|支持|创建|删除|改成|改为",
            primary or "",
            re.I,
        ))

    @staticmethod
    def _repo_full_validation_request(text: str) -> bool:
        primary = CodeEngine._primary_user_request(text)
        return bool(re.search(
            r"full\s*(test|suite|build)|entire\s*(test|suite|program)|all\s*tests|"
            r"全量测试|完整测试|全部测试|全量构建|完整构建|构建全量|全量程序|"
            r"跑全量|运行全部|测试全部",
            primary or "",
            re.I,
        ))

    def _run_full_project_validation(self, timeout: int = 300) -> tuple[bool, str]:
        """Run broad project-level tests/builds when explicitly requested."""
        root = Path(self.project_root)
        commands: List[List[str]] = []
        if (root / "tests").is_dir():
            commands.append([sys.executable, "-m", "pytest"])
        if (root / "package.json").is_file():
            commands.append(["npm", "test"])
            commands.append(["npm", "run", "build"])
        if not commands:
            return True, "[SAGE_TEST] no standard full test/build command found"

        messages = []
        for cmd in commands:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except FileNotFoundError:
                messages.append(f"[SAGE_TEST] skipped missing command: {' '.join(cmd)}")
                continue
            except subprocess.TimeoutExpired as exc:
                return False, f"full validation timed out: {' '.join(cmd)}\n{str(exc)[-2000:]}"
            if proc.returncode != 0:
                return False, (
                    f"full validation failed: {' '.join(cmd)}\n"
                    + (proc.stdout + "\n" + proc.stderr)[-6000:]
                )
            messages.append(f"[SAGE_TEST] full validation passed: {' '.join(cmd)}")
        return True, "\n".join(messages)

    def _repo_related_test_files(self, changed_files: List[str], request: str, limit: int = 8) -> List[str]:
        """Find existing focused tests likely affected by changed application files."""
        tests_dir = Path(self.project_root) / "tests"
        if not tests_dir.is_dir():
            return []

        candidates: List[str] = []
        stems = set()
        req = request or ""
        for rel in changed_files or []:
            path = Path(rel)
            if rel.startswith("tests/") or "/tests/" in rel or path.suffix != ".py":
                continue
            stems.add(path.stem.lower())
            for part in path.parts:
                if part and part not in {".", ".."}:
                    stems.add(part.lower())
        for token in self._repo_terms(req, limit=16):
            if len(token) >= 4:
                stems.add(token.lower())

        for test_path in sorted(tests_dir.rglob("test_*.py")):
            rel = test_path.relative_to(self.project_root).as_posix()
            text = rel.lower().replace("/", "_")
            if any(stem and stem in text for stem in stems):
                candidates.append(rel)
                if len(candidates) >= limit:
                    return candidates

        # Common SAGE routing/code-engine conventions.
        joined = " ".join((changed_files or []) + [req]).lower()
        pattern_hints = []
        if "chat" in joined:
            pattern_hints.append("test_chat")
        if "code_engine" in joined or "seismo_code" in joined or "coding agent" in joined:
            pattern_hints.append("test_code_engine")
        if "route" in joined or "api" in joined:
            pattern_hints.append("test_web")
        for test_path in sorted(tests_dir.rglob("test_*.py")):
            rel = test_path.relative_to(self.project_root).as_posix()
            name = test_path.name.lower()
            if any(hint in name for hint in pattern_hints) and rel not in candidates:
                candidates.append(rel)
                if len(candidates) >= limit:
                    break
        return candidates

    # ── Local API context helpers ───────────────────────────────────────────

    @staticmethod
    def _looks_like_local_api_task(text: str) -> bool:
        """Heuristic for tasks likely to use SAGE-local scientific modules."""
        return bool(re.search(
            r"\bb[-_\s]?value\b|震级|b值|完整性震级|Gutenberg|Richter|FMD|"
            r"catalog|目录|seismo_stats|plot_gr|plot_all|calc_bvalue|calc_mc|"
            r"gui|desktop|mouse|click|button|window|screenshot|keyboard|hotkey|"
            r"type text|scroll|drag|pyautogui|xdotool|鼠标|点击|按钮|窗口|界面|"
            r"屏幕|截图|键盘|快捷键|输入|滚动|拖拽",
            text or "",
            re.I,
        ))

    @staticmethod
    def _looks_like_gui_task(text: str) -> bool:
        """Heuristic for requests that need desktop GUI automation helpers."""
        return bool(re.search(
            r"gui|desktop|mouse|click|button|window|screenshot|keyboard|hotkey|"
            r"type text|scroll|drag|pyautogui|xdotool|鼠标|点击|按钮|窗口|界面|"
            r"屏幕|截图|键盘|快捷键|输入|滚动|拖拽",
            text or "",
            re.I,
        ))

    def _build_local_api_context(self, request: str, max_chars: int = 9000) -> str:
        """
        Build a compact, introspected reference for SAGE-local APIs.

        This is for normal analysis jobs, not only repository-editing tasks. It
        prevents the LLM from guessing function names, return attributes, or
        keyword arguments for modules that are present in this checkout.
        """
        if not self._looks_like_local_api_task(request):
            return ""

        modules = []
        if re.search(
            r"\bb[-_\s]?value\b|震级|b值|完整性震级|Gutenberg|Richter|FMD|"
            r"catalog|目录|seismo_stats|plot_gr|plot_all|calc_bvalue|calc_mc",
            request or "",
            re.I,
        ):
            modules.extend([
                "seismo_stats.bvalue",
                "seismo_stats.plotting",
                "seismo_stats.catalog_loader",
            ])
        if self._looks_like_gui_task(request):
            modules.append("seismo_code.gui_automation")
        parts = [
            "## Local API reference",
            "Use these exact signatures and return attributes for SAGE-local modules.",
            "Do not invent similarly named functions or keyword arguments.",
        ]
        used = sum(len(p) for p in parts)

        for mod_name in modules:
            try:
                module = importlib.import_module(mod_name)
            except Exception as exc:
                block = f"\n### {mod_name}\n(unavailable: {exc})"
                if used + len(block) <= max_chars:
                    parts.append(block)
                    used += len(block)
                continue

            lines = [f"\n### {mod_name}"]
            public = getattr(module, "__all__", None)
            names = public if public else [
                name for name in dir(module)
                if not name.startswith("_")
            ]
            for name in names:
                obj = getattr(module, name, None)
                if inspect.isfunction(obj) or inspect.isclass(obj):
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "(...)"
                    lines.append(f"- `{name}{sig}`")
                    if name == "BvalueResult" and hasattr(obj, "__dataclass_fields__"):
                        attrs = ", ".join(obj.__dataclass_fields__.keys())
                        lines.append(f"  returns attributes: {attrs}")
                    doc = inspect.getdoc(obj) or ""
                    if doc:
                        first = doc.splitlines()[0].strip()
                        if first:
                            lines.append(f"  {first[:180]}")
            block = "\n".join(lines)
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)

        parts.append(
            "\n## Local API usage notes\n"
            "- For b-value analysis, `calc_bvalue_mle(...)` returns a `BvalueResult` object; use `.b_value`, `.b_uncertainty`, `.mc`, `.n_events`, and `.summary()`.\n"
            "- For a Gutenberg-Richter plot from a `BvalueResult`, use `plot_gr(result, output_path)`. `plot_all(result, catalog, output_prefix)` is for full catalogs, not raw magnitude arrays.\n"
            "- If the user asks for uniformly random magnitudes in [0, 7], keep that uniform assumption; if no completeness threshold is specified, call `calc_bvalue_mle(magnitudes, mc=0.0)` and state that the b-value is a calculation under a non-Gutenberg-Richter synthetic distribution."
        )
        if self._looks_like_gui_task(request):
            parts.append(
                "\n## GUI automation notes\n"
                "- Use `seismo_code.gui_automation.backend_status()` to report available backends before acting.\n"
                "- Use `screenshot('screen.png')` before coordinate clicks when the target position is uncertain.\n"
                "- Use `click(x, y)`, `drag(from_x, from_y, to_x, to_y)`, `type_text(text)`, `hotkey('ctrl', 's')`, and `scroll(clicks)` for explicit GUI control.\n"
                "- Do not pretend OCR/text-clicking is available; `click_text(...)` raises a clear error unless a future OCR backend is added.\n"
                "- For browser pages, prefer browser automation/Playwright APIs when available; use pixel GUI control only when the user explicitly asks for desktop GUI operation."
            )
        return "\n\n".join(parts)

    def _build_engineering_plan(
        self,
        request: str,
        *,
        file_contexts: Optional[List[str]] = None,
        repo_ctx: str = "",
        skill_ctx: str = "",
        rag_ctx: str = "",
        local_api_ctx: str = "",
        cancel_event=None,
    ) -> str:
        """Ask the LLM for a broad-to-detailed engineering plan with API/test details."""
        self._raise_if_cancelled(cancel_event)
        context_parts = []
        if file_contexts:
            context_parts.append("## Data/file context\n" + "\n\n".join(file_contexts))
        if repo_ctx:
            context_parts.append("## Repository context\n" + repo_ctx[:12000])
        if skill_ctx:
            context_parts.append("## Skill docs\n" + skill_ctx[:10000])
        if rag_ctx:
            context_parts.append("## Knowledge base\n" + rag_ctx[:6000])
        if local_api_ctx:
            context_parts.append("## Local API reference\n" + local_api_ctx[:9000])

        prompt = (
            f"## User request\n{request}\n\n"
            + ("\n\n".join(context_parts) if context_parts else "(no extra context)")
        )
        try:
            raw = _call_llm(
                [
                    {"role": "system", "content": _ENGINEERING_PLAN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                self.llm_config,
                max_tokens=1400,
                cancel_event=cancel_event,
            ).strip()
        except Exception:
            raw = ""

        if "## Engineering Plan" in raw and "### API Details" in raw:
            return raw[:9000]

        fallback = [
            "## Engineering Plan",
            "### Route",
            "- Inspect provided context, identify the smallest implementation surface, then edit and validate.",
            "### Files",
        ]
        if repo_ctx:
            fallback.append("- Use the SAGE Repo Map and targeted rg hits to select implementation and test files.")
        elif file_contexts:
            fallback.append("- Use the profiled data file schema and exact column/API names.")
        else:
            fallback.append("- Create a standalone script with explicit functions and self-checks.")
        fallback.extend([
            "### API Details",
            "- Use exact signatures from Local API reference and SKILL docs; do not invent imports or keyword arguments.",
            "### Unit Tests",
            "- Add/update focused unit tests for changed functions/APIs, or include `[SAGE_TEST]` self-checks for standalone scripts.",
            "### Validation",
            "- Run py_compile for changed Python files and targeted pytest for changed or related tests.",
        ])
        return "\n".join(fallback)

    def _persist_engineering_plan(
        self,
        plan_text: str,
        *,
        exec_dir: Optional[str] = None,
        attempt: int = 0,
        reason: str = "initial",
    ) -> str:
        """Write the engineering plan/revision into the run directory for later debug rounds."""
        if not plan_text:
            return ""
        try:
            target_dir = Path(exec_dir or self._last_exec_dir or self.project_root) / ".sage_runtime" / "code_plans"
            if exec_dir:
                target_dir = Path(exec_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            name = "engineering_plan.md" if attempt <= 0 else f"engineering_plan_debug_round_{attempt}.md"
            path = target_dir / name
            path.write_text(
                f"<!-- SAGE engineering plan: {reason}; attempt={attempt} -->\n\n"
                + plan_text.strip()
                + "\n",
                encoding="utf-8",
            )
            if attempt <= 0:
                self._current_engineering_plan_path = str(path)
            return str(path)
        except Exception:
            return ""

    def _load_engineering_plan_for_debug(self, fallback: str = "", max_chars: int = 9000) -> str:
        """Read the persisted plan so debug rounds use the same design contract."""
        candidates = []
        if self._current_engineering_plan_path:
            candidates.append(Path(self._current_engineering_plan_path))
        if self._last_exec_dir:
            candidates.append(Path(self._last_exec_dir) / "engineering_plan.md")
        for path in candidates:
            try:
                if path.is_file():
                    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
            except Exception:
                pass
        return (fallback or "")[:max_chars]

    def _run_repo_validation(
        self,
        request: str,
        changed_files: List[str],
        timeout: int = 90,
    ) -> tuple[bool, str]:
        """Run deterministic validation for repository edits."""
        wants_full = self._repo_full_validation_request(request)
        if not changed_files:
            if wants_full:
                return self._run_full_project_validation(timeout=max(timeout, 300))
            if self._repo_behavior_change_request(request):
                return False, "repo coding task made no codebase changes"
            return True, "[SAGE_TEST] repository inspection completed without edits"

        messages: List[str] = []
        py_files = [
            f for f in changed_files
            if f.endswith(".py") and Path(self.project_root, f).is_file()
        ]
        if py_files:
            cmd = [sys.executable, "-m", "py_compile", *py_files]
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if proc.returncode != 0:
                return False, "py_compile failed:\n" + (proc.stderr or proc.stdout)[-3000:]
            messages.append(f"[SAGE_TEST] py_compile passed for {len(py_files)} file(s)")

        test_files = [
            f for f in changed_files
            if (f.startswith("tests/") or "/tests/" in f) and f.endswith(".py")
            and Path(self.project_root, f).is_file()
        ]
        related_test_files = [
            f for f in self._repo_related_test_files(changed_files, request)
            if f not in test_files
        ]
        tests_to_run = test_files + related_test_files
        if tests_to_run:
            cmd = [sys.executable, "-m", "pytest", *tests_to_run]
            proc = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=max(timeout, 180),
                check=False,
            )
            if proc.returncode != 0:
                return False, "pytest failed:\n" + (proc.stdout + "\n" + proc.stderr)[-5000:]
            kind = "changed" if not related_test_files else "changed/related"
            messages.append(f"[SAGE_TEST] pytest passed for {len(tests_to_run)} {kind} test file(s)")
        elif self._repo_behavior_change_request(request) and py_files:
            app_py = [
                f for f in py_files
                if not (f.startswith("tests/") or "/tests/" in f)
            ]
            if app_py:
                return (
                    False,
                    "Python behavior changed but no focused tests were changed or found. "
                    "Add/update unit tests under tests/ for the modified function/API, then rerun targeted pytest."
                )
            messages.append("[SAGE_TEST] repository Python changes are test-only")

        if not messages:
            messages.append("[SAGE_TEST] repository files changed: " + ", ".join(changed_files[:12]))
        if wants_full:
            ok, note = self._run_full_project_validation(timeout=max(timeout, 300))
            if not ok:
                return False, note
            messages.append(note)
        return True, "\n".join(messages)

    # ── Output checkers ───────────────────────────────────────────────────────

    def _has_failure_signal(self, text: str) -> bool:
        """Detect explicit self-check/tool failure messages in stdout/stderr."""
        if not text:
            return False
        failure_patterns = [
            r"^\s*(?:\[[^\]]+\]\s*)?(?:FAIL|FAILED|FATAL|ERROR)\b\s*[:：-]?",
            r"^\s*(?:✗|❌)\s+",
            r"\bwas not generated\b",
            r"\bnot generated\b",
            r"\bmissing (?:expected )?(?:output|file|figure|image)\b",
            r"\b(?:output|figure|image) (?:file )?(?:missing|not found)\b",
            r"\bno (?:output|figure|image) (?:was )?(?:generated|produced|created)\b",
        ]
        return any(re.search(pat, text, re.I | re.M) for pat in failure_patterns)

    def _execution_success(self, exec_res: ExecutionResult) -> bool:
        """True when process exited cleanly with no traceback in output."""
        if not exec_res or not exec_res.success:
            return False
        combined = "\n".join([exec_res.stdout or "", exec_res.stderr or ""]).strip()
        if not combined:
            return True
        if self._has_failure_signal(combined):
            return False
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

    def _mini_test_ok(
        self,
        original_request: str,
        code: str,
        exec_res: ExecutionResult,
    ) -> tuple[bool, str]:
        """
        Deterministic post-run smoke tests.

        This catches a common failure mode: the generated script exits 0 but
        silently produced no useful result, empty figures, unreadable images, or
        no visible self-checks. It is intentionally conservative; failures are
        fed back into the normal debugger as an output-check error.
        """
        stdout = (exec_res.stdout or "").strip() if exec_res else ""
        stderr = (exec_res.stderr or "").strip() if exec_res else ""
        combined = "\n".join([stdout, stderr])
        if self._has_failure_signal(combined):
            return False, "stdout/stderr contains an explicit failure self-check"
        if not exec_res or not self._execution_success(exec_res):
            return False, "execution did not succeed"

        if re.search(
            r"\b(no such file|file not found|empty dataframe|"
            r"(?:failed|error)\s*[:：])",
            combined,
            re.I,
        ):
            return False, "stdout/stderr contains an error-like message"

        wants_output = bool(re.search(
            r"plot|figure|图|绘制|visuali|chart|map|waveform|spectrogram|psd|"
            r"save|write|output|export|保存|输出|写入|统计|calculate|compute",
            original_request,
            re.I,
        ))
        produced = list(exec_res.figures or []) + list(exec_res.output_files or [])
        if wants_output and not produced and not stdout:
            return False, "request expected output but script produced no files and no stdout"

        for fp in produced:
            p = Path(fp)
            if not p.exists():
                return False, f"declared output missing: {fp}"
            if p.is_file() and p.stat().st_size == 0:
                return False, f"empty output file: {fp}"

        for fp in exec_res.figures or []:
            p = Path(fp)
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    from PIL import Image
                    with Image.open(p) as img:
                        img.verify()
                except ImportError:
                    try:
                        import matplotlib.image as _mpimg
                        _mpimg.imread(str(p))
                    except Exception as exc:
                        return False, f"unreadable image output {p.name}: {exc}"
                except Exception as exc:
                    return False, f"unreadable image output {p.name}: {exc}"

        if wants_output and "[SAGE_TEST]" not in stdout:
            return False, "missing [SAGE_TEST] self-check output"

        phase_pick_req = bool(re.search(
            r"拾取|震相|到时|phase\s*pick|arrival|pick",
            original_request,
            re.I,
        ))
        pick_visualization_req = bool(
            phase_pick_req
            and re.search(r"绘制|画|叠加|标注|visuali[sz]e|plot|draw|overlay|annotat", original_request, re.I)
            and re.search(r"结果|已有|existing|result|marker|pick", original_request, re.I)
        )
        explicit_classical_pick_req = bool(re.search(
            r"STA\s*/?\s*LTA|stalta|classic_sta_lta|recursive_sta_lta|"
            r"classical\s+trigger|classic\s+trigger|传统触发|经典触发|经典方法",
            original_request,
            re.I,
        ))
        sta_lta_code = bool(re.search(
            r"trigger_onset\s*\(|classic_sta_lta|recursive_sta_lta|"
            r"\bsta_len\b|\blta_len\b|STA\s*/?\s*LTA",
            code,
            re.I,
        ))
        if phase_pick_req and sta_lta_code and not explicit_classical_pick_req:
            return (
                False,
                "generic phase-picking code fell back to STA/LTA; use PNSNPicker from pnsn_phase_detection unless the user explicitly requests STA/LTA",
            )
        active_phase_pick_req = phase_pick_req and not pick_visualization_req
        pnsn_code = bool(re.search(
            r"PNSNPicker|pnsn_phase_detection|pnsn[/._-]?picker|"
            r"pnsn\.v\d|pick_stream\s*\(|pick_directory\s*\(",
            code,
            re.I,
        ))
        if active_phase_pick_req and not explicit_classical_pick_req and not pnsn_code:
            return (
                False,
                "phase-picking code did not use the PNSN picker API/model; import PNSNPicker from seismo_skill.skills.pnsn_phase_detection.pnsn",
            )
        if phase_pick_req and re.search(r"绘制|画|叠加|标注|visuali[sz]e|plot|draw|overlay|annotat", original_request, re.I):
            wrong_pnsn_conversion = bool(
                pnsn_code
                and re.search(r"for\s+\w+\s+in\s+picks\s*:", code)
                and re.search(
                    r"\.get\(\s*['\"](?:phase_name|absolute_time)['\"]",
                    code,
                )
                and not re.search(
                    r"\.get\(\s*['\"](?:phase|time_abs|time_rel_s)['\"]",
                    code,
                )
            )
            if wrong_pnsn_conversion:
                return (
                    False,
                    "PNSNPicker returns phase/time_abs/time_rel_s dictionaries; pass raw picks to plot_stream or preserve those keys instead of converting from phase_name/absolute_time",
                )
            pick_count_matches = re.findall(
                r"(?:PNSN\s*)?(?:拾取到|picks?\s*[:=])\s*(\d+)|PNSN picks\s*[:：]\s*(\d+)",
                stdout,
                flags=re.I,
            )
            found_pick_count = max(
                [int(a or b) for a, b in pick_count_matches if (a or b).isdigit()] or [0]
            )
            zero_plot_picks = bool(re.search(
                r"用于绘图的\s*picks\s*数量\s*[:：]\s*0|plot(?:ting)?\s+picks?\s*[:=]\s*0",
                stdout,
                re.I,
            ))
            if found_pick_count > 0 and zero_plot_picks:
                return (
                    False,
                    "PNSN picks were found but zero picks were passed to plotting; pass PNSN pick dictionaries through or normalize time_abs/time_rel_s before plot_stream",
                )

        if phase_pick_req and re.search(r"trigger_onset\s*\(", code):
            first_trigger_patterns = [
                r"triggers\s*\[\s*0\s*\]\s*\[\s*0\s*\]",
                r"triggers\s*\[\s*0\s*\]\s*\.\s*0",
                r"triggers\s*\[\s*0\s*\]",
            ]
            if any(re.search(pat, code) for pat in first_trigger_patterns):
                return (
                    False,
                    "phase picker uses the first STA/LTA trigger; ignore edge triggers and choose validated P/S candidates",
                )

        random_mag_req = bool(re.search(
            r"(?:随机(?:生成)?|random).{0,40}(?:10000|10,000).{0,40}(?:0\s*[-到至~]\s*7|between\s+0\s+and\s+7)"
            r"|(?:0\s*[-到至~]\s*7|between\s+0\s+and\s+7).{0,40}(?:随机(?:生成)?|random).{0,40}(?:震级|magnitude)",
            original_request,
            re.I | re.S,
        ))
        if random_mag_req and re.search(r"exponential|Gutenberg|Richter|true_b|beta\s*=|np\.random\.exponential", code, re.I):
            return (
                False,
                "user requested random magnitudes between 0 and 7; do not replace that assumption with a Gutenberg-Richter/exponential distribution",
            )

        if "plot_all(" in code and re.search(r"b值|b[-_\s]?value|FMD|震级|magnitude", original_request, re.I):
            wrong_plot_all = bool(re.search(
                r"plot_all\s*\(\s*(?:magnitudes|mags|np\.|[A-Za-z_][A-Za-z0-9_]*\s*,\s*(?:Mc|mc))",
                code,
            ) or re.search(r"plot_all\s*\([^)]*\b(?:b_value|mc|outfile)\s*=", code, re.S))
            if wrong_plot_all:
                return (
                    False,
                    "seismo_stats.plotting.plot_all expects (BvalueResult|None, CatalogData, output_prefix); for b-value/FMD plots use plot_gr(result, output_path)",
                )

        return True, ""

    # ── Error context builder ─────────────────────────────────────────────────

    def _build_error_context(self, code: str, exec_res: ExecutionResult) -> str:
        parts  = []
        stderr = (exec_res.stderr or "").strip()
        stdout = exec_res.stdout.strip()

        is_bash = _is_bash_code(code) or bool(re.search(
            r"(command not found|exit status \d|CalledProcessError|"
            r"bash:|/bin/sh:|/bin/bash:)",
            stderr + stdout, re.I))
        is_py = bool(re.search(
            r"(Traceback \(most recent call last\)|Error:|Exception:|"
            r"SyntaxError|IndentationError|NameError|TypeError|ValueError)", stderr))

        if is_bash and not is_py:
            parts.append("=== ERROR TYPE: Bash/CLI script failure ===")
        elif is_py:
            parts.append("=== ERROR TYPE: Python runtime error ===")

        if stdout:
            parts.append("=== Partial stdout (last 1500 chars) ===\n" + stdout[-1500:])
        if stderr:
            parts.append("=== Traceback / stderr ===\n" + stderr[-3000:])
        if exec_res.error:
            parts.append("=== Error summary ===\n" + exec_res.error)
        if exec_res.exec_dir:
            files = []
            try:
                for p in sorted(Path(exec_res.exec_dir).iterdir()):
                    if p.is_file():
                        files.append(f"{p.name} ({p.stat().st_size} bytes)")
            except Exception:
                pass
            parts.append(
                "=== Execution directory ===\n"
                f"{exec_res.exec_dir}\n"
                + ("Files:\n" + "\n".join(files[:40]) if files else "No output files detected.")
            )
        numbered = "\n".join(
            f"{i:04d}: {line}"
            for i, line in enumerate(code.splitlines(), 1)
        )
        parts.append("=== Failing code with line numbers ===\n" + numbered[-6000:])
        if is_bash:
            key_err = re.findall(r"(?:error|Error|warning|Warning|failed|Failed).*", stderr, re.I)
            if key_err:
                parts.append("=== Bash/CLI key error lines ===\n" + "\n".join(key_err[-5:]))

        return "\n\n".join(parts) if parts else "No error details captured."

    @staticmethod
    def _response_has_fenced_code(text: str) -> bool:
        """True when an LLM response contains a fenced code block."""
        return bool(re.search(r"```(?:python|py|bash|sh)?\s*.*?(?:```|\Z)", text or "", re.DOTALL | re.I))

    @staticmethod
    def _python_syntax_error(code: str) -> str:
        """Return a syntax error message for Python code, or empty string when valid."""
        if _is_bash_code(code):
            return ""
        try:
            ast.parse(re.sub(r"^#\s*lang:python\s*\n", "", code, count=1))
            return ""
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg} at line {exc.lineno}"

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
        local_api_ctx: str = "",    # ← local introspected API docs
        exec_dir: Optional[str] = None,  # ← run fixed code in this dir (workflow)
        engineering_plan: str = "",
        cancel_event=None,
    ) -> tuple[str, ExecutionResult, str]:
        """
        Ask the LLM debugger to fix failing code, then execute the fix.

        skill_ctx is forwarded from code generation so the debugger sees
        the same API documentation it should use when rewriting.
        exec_dir avoids double-execution: the fix is run here directly in
        the workflow's shared directory rather than re-run outside.

        Returns (fixed_code, new_exec_result, diagnosis).
        """
        self._raise_if_cancelled(cancel_event)
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
        if local_api_ctx:
            debug_system += (
                "\n\n## Local API reference\n"
                + local_api_ctx
                + "\n\nUse these exact signatures and attributes when fixing imports, calls, and plots.")
        persisted_plan = self._load_engineering_plan_for_debug(engineering_plan)
        if persisted_plan:
            debug_system += (
                "\n\n## Persisted engineering plan\n"
                + persisted_plan
                + "\n\nDebug against this design/API/test plan. If the fix changes the plan, "
                "print the updated detail and keep tests aligned."
            )

        failed_lang = "bash" if _is_bash_code(failed_code) else "python"
        debug_messages = [
            {"role": "system", "content": debug_system},
            {"role": "user", "content": (
                f"## Original request\n{original_request}"
                f"{file_ctx_str}\n\n"
                f"## Failing code\n```{failed_lang}\n{failed_code}\n```\n\n"
                f"## Error output\n{error_ctx}\n\n"
                "Fix the code. Output exactly two parts: one [DIAGNOSIS] line, then one complete corrected fenced code block. "
                "Do not output prose, Markdown explanations, or diagnosis text inside the code block. "
                "The code block must be executable and syntactically complete."
            )},
        ]

        self._emit(on_progress, "debugging", attempt, f"Analyzing error (attempt {attempt})…")
        try:
            raw = _call_llm(
                debug_messages, self.llm_config, max_tokens=4096,
                cancel_event=cancel_event)
        except ConnectionError as e:
            return failed_code, exec_res, str(e)

        self._raise_if_cancelled(cancel_event)
        diagnosis  = _extract_diagnosis(raw)
        if not self._response_has_fenced_code(raw):
            from dataclasses import replace as _dc_replace
            guarded = _dc_replace(
                exec_res,
                success=False,
                stderr=(
                    (exec_res.stderr or "")
                    + "\n[DEBUG OUTPUT INVALID] debugger returned prose instead of a fenced code block"
                ),
                error="Debugger returned prose instead of executable code",
            )
            return failed_code, guarded, (
                diagnosis
                + " [Rejected: debugger response did not contain a fenced code block.]"
            )
        fixed_code = _extract_code(raw)
        syntax_error = self._python_syntax_error(fixed_code)
        if syntax_error:
            from dataclasses import replace as _dc_replace
            guarded = _dc_replace(
                exec_res,
                success=False,
                stderr=(exec_res.stderr or "") + f"\n[DEBUG OUTPUT INVALID] {syntax_error}",
                error=f"Debugger returned syntactically invalid code: {syntax_error}",
            )
            return failed_code, guarded, (
                diagnosis
                + f" [Rejected: debugger returned syntactically invalid code: {syntax_error}.]"
            )

        self._emit(on_progress, "executing", attempt, f"Running fixed code (attempt {attempt})…")
        # Execute in shared dir (workflow) or fresh temp dir (single-request)
        if exec_dir:
            new_exec = self._run_code_in_dir(
                fixed_code, timeout, exec_dir, cancel_event=cancel_event,
                on_progress=on_progress, attempt=attempt)
        else:
            new_exec = self._run_code(
                fixed_code, timeout, cancel_event=cancel_event,
                on_progress=on_progress, attempt=attempt)

        try:
            plan_revision = self._build_engineering_plan(
                original_request
                + "\n\nDebug diagnosis: "
                + diagnosis
                + "\n\nLatest error context:\n"
                + error_ctx[:3000],
                file_contexts=file_contexts,
                skill_ctx=skill_ctx,
                rag_ctx=extra_rag_ctx,
                local_api_ctx=local_api_ctx,
                cancel_event=cancel_event,
            )
            self._persist_engineering_plan(
                plan_revision,
                exec_dir=exec_dir or new_exec.exec_dir,
                attempt=attempt,
                reason=f"debug round {attempt}: {diagnosis[:120]}",
            )
        except Exception:
            pass

        return fixed_code, new_exec, diagnosis

    # ── Verify output ─────────────────────────────────────────────────────────

    def _verify_output(self, original_request: str,
                       exec_res: ExecutionResult,
                       cancel_event=None) -> tuple[bool, str]:
        """Quick LLM sanity-check: did the output fulfil the request?"""
        self._raise_if_cancelled(cancel_event)
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
            verdict = _call_llm(
                msgs, self.llm_config, max_tokens=80,
                cancel_event=cancel_event).strip()
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
        output_dir: Optional[str] = None,
        cancel_event=None,
    ) -> CodeRunResult:
        """Generate, execute, debug, and optionally verify Python or bash code."""
        try:
            self._raise_if_cancelled(cancel_event)
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            repo_task = self._looks_like_repo_task(user_request)
            if repo_task:
                self._repo_baseline_files = self._repo_snapshot()

            # 1. Profile files mentioned in the request
            file_contexts: List[str] = []
            all_text = user_request + (f"\n{data_hint}" if data_hint else "")
            for fp in _find_file_paths(all_text)[:3]:
                self._raise_if_cancelled(cancel_event)
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
            repo_ctx = ""
            if repo_task:
                self._emit(on_progress, "analyzing", 0, "Building repository context…")
                repo_ctx = self._build_repo_context(user_request)
                if repo_ctx:
                    msg += "\n\n" + repo_ctx

            # 3. Build skill + RAG context (queried once; forwarded to debug loop)
            try:
                self._raise_if_cancelled(cancel_event)
                skill_ctx, rag_ctx = _build_ctx(
                    user_request, max_skill_chars=18000, max_rag_chars=7000, top_k=7)
            except CodeExecutionCancelled:
                raise
            except Exception:
                skill_ctx, rag_ctx = "", ""

            local_api_ctx = self._build_local_api_context(user_request)

            self._emit(on_progress, "planning", 0, "Building engineering plan…")
            engineering_plan = self._build_engineering_plan(
                user_request,
                file_contexts=file_contexts,
                repo_ctx=repo_ctx,
                skill_ctx=skill_ctx,
                rag_ctx=rag_ctx,
                local_api_ctx=local_api_ctx,
                cancel_event=cancel_event,
            )
            if engineering_plan:
                msg += "\n\n## Engineering Plan to follow\n" + engineering_plan
                preview = " → ".join(
                    ln.lstrip("- ").strip()
                    for ln in engineering_plan.splitlines()
                    if ln.strip().startswith("- ")
                )[:500]
                if preview:
                    self._emit(on_progress, "planning", 0, "Engineering plan: " + preview)

            self._history.append({"role": "user", "content": msg})

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
            if repo_ctx:
                system += (
                    "\n\n## Built-in Coding Agent Mode\n"
                    "You are working as SAGE's built-in repository coding agent. "
                    "Use the repository context to make minimal, coherent codebase edits across as many files as needed. "
                    "Use repo-aware editing discipline: inspect relevant files, edit only what is needed, "
                    "add/update focused tests for behavioral changes, run py_compile/pytest checks, and print a concise diff summary. "
                    "The generated script should be an edit-and-test driver, not the final application code."
                )
            if local_api_ctx:
                system += "\n\n" + local_api_ctx
            if engineering_plan:
                system += (
                    "\n\n## Required Engineering Plan\n"
                    + engineering_plan
                    + "\n\nFollow this plan from coarse route to file/API details to unit tests. "
                    "If implementation discovers a better detail, update the code and tests coherently and print the reason."
                )

            messages = [{"role": "system", "content": system}] + \
                       [m for m in self._history if m["role"] != "system"]

            # 4. Plan (non-fatal)
            plan: List[str] = []
            self._emit(on_progress, "planning", 0, "Planning…")
            try:
                plan_context = "\n".join(file_contexts)
                if repo_ctx:
                    plan_context += "\n\n" + repo_ctx[:10000]
                plan = _extract_plan(_call_llm(
                    [{"role": "system", "content": _PLAN_SYSTEM},
                     {"role": "user", "content":
                      f"Request: {user_request}\n\n{plan_context}"
                      + "\n\nList the execution steps. For repository coding tasks, include search/localization, edits, tests, and validation."}],
                    self.llm_config, max_tokens=400,
                    cancel_event=cancel_event))
            except CodeExecutionCancelled:
                raise
            except Exception:
                pass
            if plan:
                self._emit(on_progress, "planning", 0, "Plan: " + " → ".join(plan))

            # 5. Generate code
            self._emit(on_progress, "generating", 0, "Generating code…")
            try:
                code = _extract_code(_call_llm(
                    messages, self.llm_config, cancel_event=cancel_event))
            except ConnectionError as e:
                return CodeRunResult(success=False, response=str(e), code="", exec_result=None)

            self._raise_if_cancelled(cancel_event)
            placeholder_reason = self._placeholder_path_reason(code)
            if placeholder_reason:
                response = (
                    "未执行代码：模型生成了示例/占位数据路径，而不是实际存在的输入文件。\n"
                    "请在问题中提供真实的 SAC/MSEED/CSV 文件或目录路径后重试。\n"
                    f"原因: {placeholder_reason}"
                )
                self._emit(on_progress, "done", 0, response)
                return CodeRunResult(
                    success=False,
                    response=response,
                    code=code,
                    exec_result=None,
                    attempts=0,
                    debug_trace=[],
                    plan=plan,
                )

            # 6. First execution
            self._emit(on_progress, "executing", 0, "Executing code…")
            if output_dir:
                exec_res = self._run_code_in_dir(
                    code, timeout, shared_dir=output_dir, cancel_event=cancel_event,
                    on_progress=on_progress, attempt=0)
            else:
                exec_res = self._run_code(
                    code, timeout, cancel_event=cancel_event,
                    on_progress=on_progress, attempt=0)
            if 'engineering_plan' in locals() and engineering_plan:
                plan_path = self._persist_engineering_plan(
                    engineering_plan,
                    exec_dir=output_dir or exec_res.exec_dir,
                    attempt=0,
                    reason="initial generation",
                )
                if plan_path:
                    self._emit(on_progress, "planning", 0, f"Engineering plan saved: {plan_path}")
                    from dataclasses import replace as _dc_replace
                    exec_res = _dc_replace(
                        exec_res,
                        output_files=list(dict.fromkeys((exec_res.output_files or []) + [plan_path])),
                    )
            self._raise_if_cancelled(cancel_event)
        except CodeExecutionCancelled:
            self._emit(on_progress, "cancelled", 0, "Execution cancelled.")
            return CodeRunResult(
                success=False,
                response="已取消执行。",
                code="",
                exec_result=None,
                attempts=0,
                debug_trace=[],
                plan=[],
            )
        debug_trace: List[DebugAttempt] = []
        attempt = 0

        # 7. Debug loop — handles runtime errors AND mini-test/output-check failures.
        while attempt < max_debug_rounds:
            self._raise_if_cancelled(cancel_event)
            if self._execution_success(exec_res):
                test_ok, test_reason = self._mini_test_ok(user_request, code, exec_res)
                if test_ok and repo_task:
                    changed = self._repo_new_changed_files()
                    test_ok, test_reason = self._run_repo_validation(user_request, changed)
                    if test_ok:
                        from dataclasses import replace as _dc_replace
                        exec_res = _dc_replace(
                            exec_res,
                            stdout=(exec_res.stdout or "") + "\n" + test_reason,
                        )
                if test_ok:
                    break
                from dataclasses import replace as _dc_replace
                exec_res = _dc_replace(
                    exec_res,
                    success=False,
                    stderr=(exec_res.stderr or "") + f"\n[MINI TEST FAILED] {test_reason}",
                    error=f"Mini test failed: {test_reason}",
                )

            attempt += 1
            err_summary = f"{exec_res.stdout}\n{exec_res.stderr}\n{exec_res.error}".strip()
            debug_trace.append(DebugAttempt(
                attempt=attempt, diagnosis="", code=code,
                error=err_summary, stdout=exec_res.stdout, success=False))

            try:
                self._raise_if_cancelled(cancel_event)
                _, dbg_rag = _build_ctx(f"{user_request} {err_summary[:400]}",
                                        max_skill_chars=4000, max_rag_chars=5000, top_k=5)
            except CodeExecutionCancelled:
                raise
            except Exception:
                dbg_rag = ""

            code, exec_res, diagnosis = self._debug_and_fix(
                original_request=user_request, failed_code=code,
                exec_res=exec_res, attempt=attempt, timeout=timeout,
                on_progress=on_progress, file_contexts=file_contexts,
                skill_ctx=skill_ctx, extra_rag_ctx=dbg_rag,
                local_api_ctx=local_api_ctx if 'local_api_ctx' in locals() else "",
                exec_dir=output_dir,
                engineering_plan=engineering_plan if 'engineering_plan' in locals() else "",
                cancel_event=cancel_event,
            )
            self._raise_if_cancelled(cancel_event)
            debug_trace[-1].diagnosis = diagnosis

            if self._execution_success(exec_res):
                test_ok, test_reason = self._mini_test_ok(user_request, code, exec_res)
                if test_ok and repo_task:
                    changed = self._repo_new_changed_files()
                    test_ok, test_reason = self._run_repo_validation(user_request, changed)
                    if test_ok:
                        from dataclasses import replace as _dc_replace
                        exec_res = _dc_replace(
                            exec_res,
                            stdout=(exec_res.stdout or "") + "\n" + test_reason,
                        )
                if test_ok:
                    debug_trace.append(DebugAttempt(
                        attempt=attempt, diagnosis=f"Fixed: {diagnosis}",
                        code=code, error="", stdout=exec_res.stdout, success=True))
                    self._emit(on_progress, "executing", attempt,
                               f"✓ Fixed after {attempt} debug round(s)")
                    break
                from dataclasses import replace as _dc_replace
                exec_res = _dc_replace(
                    exec_res,
                    success=False,
                    stderr=(exec_res.stderr or "") + f"\n[MINI TEST FAILED] {test_reason}",
                    error=f"Mini test failed: {test_reason}",
                )
            self._emit(on_progress, "debugging", attempt,
                       f"Attempt {attempt} still failing, retrying…")

        # 8. Update conversation history
        final_success = self._execution_success(exec_res)
        if final_success:
            final_success, mini_note = self._mini_test_ok(user_request, code, exec_res)
            if final_success and 'repo_task' in locals() and repo_task:
                changed = self._repo_new_changed_files()
                final_success, mini_note = self._run_repo_validation(user_request, changed)
                if final_success:
                    from dataclasses import replace as _dc_replace
                    exec_res = _dc_replace(
                        exec_res,
                        stdout=(exec_res.stdout or "") + "\n" + mini_note,
                    )
            if not final_success:
                from dataclasses import replace as _dc_replace
                exec_res = _dc_replace(
                    exec_res,
                    success=False,
                    stderr=(exec_res.stderr or "") + f"\n[MINI TEST FAILED] {mini_note}",
                    error=f"Mini test failed: {mini_note}",
                )
        summary = "Execution " + ("succeeded." if final_success else "failed.")
        if exec_res and exec_res.figures:
            summary += "\nFigures: " + str([Path(f).name for f in exec_res.figures])
        if exec_res and exec_res.output_files:
            summary += "\nFiles: "   + str([Path(f).name for f in exec_res.output_files])
        if exec_res and exec_res.stdout.strip():
            clean = "\n".join(l for l in exec_res.stdout.splitlines()
                              if not l.startswith("[FIGURE]")).strip()
            if clean:
                summary += f"\nOutput (truncated):\n{clean[:400]}"
        diff_stat = self._git_diff_summary() if ('repo_task' in locals() and repo_task) else ""
        if diff_stat:
            summary += "\nGit diff stat:\n" + diff_stat[:1200]
        if exec_res and not final_success:
            err = (exec_res.stderr or exec_res.error or "").strip()
            if err:
                summary += f"\nError:\n{err[:300]}"
        lang = "bash" if _is_bash_code(code) else "python"
        self._history.append({
            "role": "assistant",
            "content": f"```{lang}\n{code}\n```\n\n[Result] {summary}",
        })
        if exec_res:
            self._last_exec_dir = exec_res.exec_dir
            try:
                plan_files = []
                if exec_res.exec_dir:
                    plan_files = [
                        str(p)
                        for p in sorted(Path(exec_res.exec_dir).glob("engineering_plan*.md"))
                        if p.is_file()
                    ]
                if plan_files:
                    from dataclasses import replace as _dc_replace
                    exec_res = _dc_replace(
                        exec_res,
                        output_files=list(dict.fromkeys((exec_res.output_files or []) + plan_files)),
                    )
            except Exception:
                pass

        # 9. Verify (optional)
        verify_pass, verify_note = None, ""
        if run_verify and exec_res and exec_res.success:
            self._emit(on_progress, "verifying", attempt, "Verifying output…")
            verify_pass, verify_note = self._verify_output(
                user_request, exec_res, cancel_event=cancel_event)

        # 10. Save final script
        script_path = ""
        if code:
            try:
                import os, tempfile
                sd = (exec_res.exec_dir if (exec_res and exec_res.exec_dir)
                      else tempfile.mkdtemp(prefix="sage_script_"))
                is_bash = _is_bash_code(code)
                script_path = os.path.join(sd, "analysis.sh" if is_bash else "analysis.py")
                with open(script_path, "w", encoding="utf-8") as f:
                    clean_code = re.sub(r"^#\s*lang:(?:python|bash)\s*\n", "", code, count=1)
                    f.write(f"# Generated by SeismicX — {user_request[:80]}\n"
                            f"# Attempts: {attempt+1}\n\n" + clean_code)
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
                max_skill_chars=12000, max_rag_chars=5000, top_k=5)
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
        cancel_event=None,
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
            self._raise_if_cancelled(cancel_event)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from seismo_skill.workflow_runner import load_workflow
            workflow = load_workflow(workflow_name)
        except CodeExecutionCancelled:
            return WorkflowRunResult(
                workflow_name=workflow_name, workflow_title="",
                success=False, steps_total=0, steps_done=0,
                step_results=[], all_figures=[], all_output_files=[],
                response="已取消执行。")
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
            self._raise_if_cancelled(cancel_event)
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
                code = _extract_code(_call_llm(
                    messages, self.llm_config, cancel_event=cancel_event))
            except CodeExecutionCancelled:
                raise
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
            exec_res  = self._run_code_in_dir(
                code, timeout, shared_dir, cancel_event=cancel_event,
                on_progress=on_progress, attempt=0)
            self._raise_if_cancelled(cancel_event)
            attempt   = 0
            diagnosis = ""
            if shared_dir is None and exec_res.exec_dir:
                shared_dir = exec_res.exec_dir

            # Debug loop
            # The loop handles both runtime failures AND semantic output failures
            # (_step_output_ok). skill_ctx is forwarded; exec_dir avoids double-execution.
            while attempt < max_debug_rounds:
                self._raise_if_cancelled(cancel_event)
                exec_ok = self._execution_success(exec_res)
                if exec_ok:
                    out_ok, out_reason = self._step_output_ok(desc, exec_res)
                    if out_ok:
                        out_ok, out_reason = self._mini_test_ok(
                            f"{desc}\n{user_request}", code, exec_res)
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
                    self._raise_if_cancelled(cancel_event)
                    _, dbg_rag = _build_ctx(
                        f"{skill_nm} {desc} {err_summary[:300]}",
                        max_skill_chars=3000, max_rag_chars=4000, top_k=4)
                except CodeExecutionCancelled:
                    raise
                except Exception:
                    dbg_rag = ""

                # _debug_and_fix runs the fix in shared_dir — no separate re-run needed
                code, exec_res, diagnosis = self._debug_and_fix(
                    original_request=f"{desc}\n{user_request}",
                    failed_code=code, exec_res=exec_res,
                    attempt=attempt, timeout=timeout, on_progress=on_progress,
                    file_contexts=[f"Available: {f}" for f in available_files[:5]],
                    skill_ctx=skill_ctx,     # ← forwarded skill docs
                    extra_rag_ctx=dbg_rag,
                    exec_dir=shared_dir,     # ← no double-execution
                    cancel_event=cancel_event,
                )
                self._raise_if_cancelled(cancel_event)
                if shared_dir is None and exec_res.exec_dir:
                    shared_dir = exec_res.exec_dir

            step_success = self._execution_success(exec_res)
            if step_success:
                step_success, final_step_reason = self._mini_test_ok(
                    f"{desc}\n{user_request}", code, exec_res)
                if not step_success:
                    from dataclasses import replace as _dc_replace
                    exec_res = _dc_replace(
                        exec_res,
                        success=False,
                        stderr=(exec_res.stderr or "") + f"\n[MINI TEST FAILED] {final_step_reason}",
                        error=f"Mini test failed: {final_step_reason}",
                    )
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
                lang = "bash" if _is_bash_code(code) else "python"
                wf_history.append({
                    "role": "assistant",
                    "content": f"```{lang}\n{code}\n```\n\n[Step result] {step_summary}"})

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
        assert _is_bash_code("#!/usr/bin/env bash\necho hi");      ok("env shebang")
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
        assert isinstance(eng.is_llm_available(), bool); ok("instantiation + is_llm_available()")
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

    # 9. _mini_test_ok heuristics
    print("\n[9] _mini_test_ok")
    try:
        from .safe_executor import ExecutionResult
        eng = CodeEngine(llm_config={})
        ok_res = ExecutionResult(
            success=True,
            stdout="[SAGE_TEST] output checked",
            stderr="", error="", figures=[], output_files=[], exec_dir="")
        passed_test, reason = eng._mini_test_ok("compute statistics", "print('x')", ok_res)
        assert passed_test, reason
        ok("mini test accepts explicit self-check")
        no_check = ExecutionResult(
            success=True, stdout="done", stderr="", error="",
            figures=[], output_files=[], exec_dir="")
        passed_test, reason = eng._mini_test_ok("compute statistics", "print('x')", no_check)
        assert not passed_test and "SAGE_TEST" in reason
        ok("mini test rejects missing self-check for output task")
    except Exception as e:
        fail("_mini_test_ok", e)

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
