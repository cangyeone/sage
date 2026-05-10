"""Adapters for integrated repository-level coding backends such as Aider and OpenHands."""

from __future__ import annotations

import os
import importlib.util
import inspect
import io
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .ce_utils import CodeRunResult
from .safe_executor import ExecutionResult


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_AIDER_ROOT = _PROJECT_ROOT / "third_party" / "aider"
_VENDORED_AIDER_PACKAGE = _VENDORED_AIDER_ROOT / "aider"


def _split_args(value: str) -> List[str]:
    if not value:
        return []
    try:
        return shlex.split(value)
    except Exception:
        return [x for x in value.split() if x]


def _resolve_command(config: Dict, backend: str) -> str:
    configured = str(config.get("command") or "").strip()
    if configured:
        return configured
    return shutil.which(backend) or backend


def _has_vendored_aider() -> bool:
    return (_VENDORED_AIDER_PACKAGE / "__init__.py").exists()


def _ensure_vendored_aider_on_path() -> bool:
    if not _has_vendored_aider():
        return False
    root = str(_VENDORED_AIDER_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return True


def _module_available(name: str) -> bool:
    if name == "aider" and _ensure_vendored_aider_on_path():
        return importlib.util.find_spec(name) is not None
    return importlib.util.find_spec(name) is not None


def _python_module_command(module: str) -> List[str]:
    return [shutil.which("python") or "python", "-m", module]


def _derive_aider_model(config: Dict, llm_config: Optional[Dict] = None) -> str:
    configured = str(config.get("model") or "").strip()
    if configured:
        return configured
    llm_config = llm_config or {}
    provider = (llm_config.get("provider") or "").lower()
    model = str(llm_config.get("model") or "").strip()
    if not model:
        return ""
    if provider == "ollama":
        return f"ollama/{model}"
    if provider == "deepseek":
        return f"deepseek/{model}"
    if provider in {"siliconflow", "custom", "moonshot", "dashscope", "zhipu"}:
        return f"openai/{model}"
    if provider == "anthropic":
        return f"anthropic/{model}"
    return model


def _llm_env(llm_config: Dict, enabled: bool) -> Dict[str, str]:
    if not enabled:
        return {}
    provider = (llm_config.get("provider") or "").lower()
    api_key = llm_config.get("api_key") or ""
    api_base = llm_config.get("api_base") or ""
    env: Dict[str, str] = {}
    if api_key:
        if provider == "openai":
            env["OPENAI_API_KEY"] = api_key
        elif provider == "deepseek":
            env["DEEPSEEK_API_KEY"] = api_key
        elif provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = api_key
        elif provider == "moonshot":
            env["MOONSHOT_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        elif provider == "dashscope":
            env["DASHSCOPE_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        elif provider == "zhipu":
            env["ZHIPUAI_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        elif provider == "siliconflow":
            env["SILICONFLOW_API_KEY"] = api_key
            env["OPENAI_API_KEY"] = api_key
        else:
            env["OPENAI_API_KEY"] = api_key
    if api_base and provider not in {"ollama", "anthropic"}:
        env["OPENAI_API_BASE"] = api_base
        env["OPENAI_BASE_URL"] = api_base
    return env


def _patch_litellm_compat() -> None:
    """Patch small LiteLLM API differences expected by vendored Aider."""
    try:
        import litellm  # type: ignore
    except Exception:
        return
    if not hasattr(litellm, "PermissionDeniedError"):
        fallback = getattr(litellm, "AuthenticationError", None) or getattr(litellm, "OpenAIError", Exception)
        setattr(litellm, "PermissionDeniedError", fallback)


def _git_changed_files(workspace: Path) -> List[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _git_diff_stat(workspace: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "diff", "--stat"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        return proc.stdout.strip()
    except Exception:
        return ""


def _aider_command(task_file: Path, config: Dict) -> List[str]:
    cmd = [
        _resolve_command(config, "aider"),
        "--message-file",
        str(task_file),
        "--no-auto-commits",
    ]
    if config.get("auto_approve", True):
        cmd.append("--yes")
    model = str(config.get("model") or "").strip()
    if model:
        cmd.extend(["--model", model])
    cmd.extend(_split_args(str(config.get("extra_args") or "")))
    return cmd


def _aider_module_command(task_file: Path, config: Dict, llm_config: Optional[Dict] = None) -> List[str]:
    cmd = [
        *_python_module_command("aider"),
        "--message-file",
        str(task_file),
        "--no-auto-commits",
    ]
    if config.get("auto_approve", True):
        cmd.append("--yes")
    model = _derive_aider_model(config, llm_config)
    if model:
        cmd.extend(["--model", model])
    cmd.extend(_split_args(str(config.get("extra_args") or "")))
    return cmd


def _openhands_command(task_file: Path, config: Dict) -> List[str]:
    cmd = [
        _resolve_command(config, "openhands"),
        "--headless",
        "--file",
        str(task_file),
    ]
    if config.get("auto_approve", True):
        cmd.append("--always-approve")
    model = str(config.get("model") or "").strip()
    if model:
        cmd.extend(["--model", model])
    cmd.extend(_split_args(str(config.get("extra_args") or "")))
    return cmd


def external_agent_status(config: Dict) -> Dict:
    backend = (config.get("backend") or "builtin").strip()
    if backend == "builtin":
        return {"available": True, "backend": backend, "command": ""}
    if backend == "aider" and _ensure_vendored_aider_on_path():
        return {
            "available": True,
            "backend": backend,
            "command": f"python-api:vendored-aider@{_VENDORED_AIDER_ROOT}",
            "mode": "api",
            "source": "vendored",
        }
    if backend == "aider" and _module_available("aider"):
        return {
            "available": True,
            "backend": backend,
            "command": "python-api:aider",
            "mode": "api",
            "source": "installed",
        }
    command = _resolve_command(config, backend)
    if os.path.isabs(command):
        executable = command if os.path.isfile(command) and os.access(command, os.X_OK) else ""
    else:
        executable = shutil.which(command)
    return {
        "available": bool(executable),
        "backend": backend,
        "command": executable or command,
        "mode": "cli",
        "source": "path",
    }


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(old))


@contextmanager
def _patched_env(values: Dict[str, str]):
    old_values = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _call_with_supported_kwargs(func, **kwargs):
    try:
        params = inspect.signature(func).parameters
    except Exception:
        return func(**kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return func(**accepted)


def _run_aider_api(
    task: str,
    *,
    workspace_path: Path,
    llm_config: Dict,
    config: Dict,
) -> Tuple[int, str, str, str]:
    """Run Aider through its Python scripting API.

    Aider documents this API as scripting-friendly but not guaranteed stable, so
    the integration uses conservative argument introspection and the caller keeps
    a CLI fallback available.
    """
    _ensure_vendored_aider_on_path()
    _patch_litellm_compat()
    from aider.coders import Coder
    from aider.io import InputOutput
    from aider.models import Model

    model_name = _derive_aider_model(config, llm_config)
    if not model_name:
        raise RuntimeError("Aider backend needs a model; configure one in Config -> Coding Agent or in the active LLM config.")

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    code_repr = (
        f"python-api:vendored-aider model={model_name}"
        if _has_vendored_aider()
        else f"python-api:aider model={model_name}"
    )
    env_values = _llm_env(llm_config, bool(config.get("use_current_llm_env", True)))
    with _patched_env(env_values), _pushd(workspace_path), redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        main_model = Model(model_name)
        io_obj = _call_with_supported_kwargs(
            InputOutput,
            yes=bool(config.get("auto_approve", True)),
        )
        coder = _call_with_supported_kwargs(
            Coder.create,
            main_model=main_model,
            fnames=[],
            io=io_obj,
            auto_commits=False,
            dirty_commits=False,
        )
        result = coder.run(task)
        if result is not None:
            stdout_buf.write("\n")
            stdout_buf.write(str(result))
    return 0, stdout_buf.getvalue(), stderr_buf.getvalue(), code_repr


def _run_cli(
    cmd: List[str],
    *,
    workspace_path: Path,
    env: Dict[str, str],
    timeout: int,
) -> Tuple[int, str, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(workspace_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or "", " ".join(shlex.quote(x) for x in cmd)


def run_external_coding_agent(
    task: str,
    *,
    workspace: str,
    llm_config: Dict,
    config: Dict,
    progress_cb: Optional[Callable[[Dict], None]] = None,
    cancel_event: Optional[object] = None,
) -> CodeRunResult:
    """Run an integrated coding backend CLI and adapt its result to CodeRunResult."""
    backend = (config.get("backend") or "builtin").strip()
    workspace_path = Path(workspace).expanduser().resolve()
    timeout = int(config.get("timeout_s") or 900)

    status = external_agent_status(config)
    if backend not in {"aider", "openhands"}:
        return CodeRunResult(
            success=False,
            response=f"Coding backend is not handled by the CLI adapter: {backend}",
            code="",
            exec_result=None,
        )
    if not status.get("available"):
        install_hint = (
            "Install the project dependency `aider-chat` or run `python -m pip install aider-chat`."
            if backend == "aider"
            else "Install OpenHands CLI, then run `openhands --help` to verify it."
        )
        return CodeRunResult(
            success=False,
            response=f"{backend} command not found. Install/configure it first: {install_hint}",
            code="",
            exec_result=ExecutionResult(
                success=False,
                error=f"{backend} command not found",
                stdout="",
                stderr="",
            ),
        )

    workspace_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sage_coding_backend_") as tmp:
        task_file = Path(tmp) / "task.md"
        task_file.write_text(task, encoding="utf-8")

        env = os.environ.copy()
        env.update(_llm_env(llm_config, bool(config.get("use_current_llm_env", True))))
        if backend == "aider" and _has_vendored_aider():
            env["PYTHONPATH"] = (
                str(_VENDORED_AIDER_ROOT)
                + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            )
        if progress_cb:
            mode = status.get("mode") or ("api" if backend == "aider" and _module_available("aider") else "cli")
            progress_cb({
                "phase": "coding_backend",
                "message": f"Running coding backend {backend} ({mode})",
            })
        started = time.time()
        try:
            if backend == "aider" and status.get("mode") == "api":
                returncode, stdout, stderr, code_repr = _run_aider_api(
                    task,
                    workspace_path=workspace_path,
                    llm_config=llm_config,
                    config=config,
                )
            else:
                if backend == "aider":
                    cmd = (
                        _aider_module_command(task_file, config, llm_config)
                        if _module_available("aider")
                        else _aider_command(task_file, config)
                    )
                else:
                    cmd = _openhands_command(task_file, config)
                returncode, stdout, stderr, code_repr = _run_cli(
                    cmd,
                    workspace_path=workspace_path,
                    env=env,
                    timeout=timeout,
                )
            success = returncode == 0
            changed = _git_changed_files(workspace_path)
            diff_stat = _git_diff_stat(workspace_path)
            response_parts = [
                f"Coding backend `{backend}` finished with exit code {returncode}.",
                f"Elapsed: {time.time() - started:.1f}s.",
            ]
            if changed:
                response_parts.append("Changed files:\n" + "\n".join(f"- {f}" for f in changed[:80]))
            if diff_stat:
                response_parts.append("Diff stat:\n" + diff_stat)
            if stdout.strip():
                response_parts.append("STDOUT preview:\n" + stdout[-5000:])
            if stderr.strip():
                response_parts.append("STDERR preview:\n" + stderr[-3000:])
            return CodeRunResult(
                success=success,
                response="\n\n".join(response_parts),
                code=code_repr,
                exec_result=ExecutionResult(
                    success=success,
                    stdout=stdout,
                    stderr=stderr,
                    error="" if success else (stderr.strip().splitlines()[-1] if stderr.strip() else f"exit {returncode}"),
                    output_files=[str(workspace_path / f) for f in changed if (workspace_path / f).is_file()],
                    exec_dir=str(workspace_path),
                ),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return CodeRunResult(
                success=False,
                response=f"`{backend}` timed out after {timeout}s.",
                code=f"{backend} backend",
                exec_result=ExecutionResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr,
                    error=f"timeout after {timeout}s",
                    exec_dir=str(workspace_path),
                ),
            )
        except FileNotFoundError as exc:
            return CodeRunResult(
                success=False,
                response=f"`{backend}` command not found: {exc}",
                code=" ".join(shlex.quote(x) for x in cmd),
                exec_result=ExecutionResult(
                    success=False,
                    stdout="",
                    stderr=str(exc),
                    error=str(exc),
                    exec_dir=str(workspace_path),
                ),
            )
