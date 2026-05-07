"""
safe_executor.py — 安全地在子进程中执行 LLM 生成的 Python 代码。

工作原理
--------
1. 将代码写入临时目录
2. 在子进程中执行，有超时保护
3. 捕获 stdout / stderr
4. 收集生成的图像文件（PNG/PDF）
5. 返回 ExecutionResult 结构

安全说明
--------
• 代码在独立子进程中运行，主进程不受崩溃影响
• 限制执行超时（默认 60 秒）
• 禁止子进程继续 fork（可选）
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    error: str = ""           # Short error summary
    figures: List[str] = field(default_factory=list)   # Absolute paths of generated images
    output_files: List[str] = field(default_factory=list)  # Other generated files
    variables: Dict[str, Any] = field(default_factory=dict)  # Exported variables (via SAGE_EXPORT)
    exec_dir: str = ""        # The temp directory used (kept for inspection)

    def short_summary(self) -> str:
        lines = []
        if self.success:
            lines.append("✓ 代码执行成功")
        else:
            lines.append(f"✗ 执行失败: {self.error}")
        if self.stdout.strip():
            lines.append("输出:\n" + textwrap.indent(self.stdout.strip(), "  "))
        if self.figures:
            lines.append(f"生成图像 ({len(self.figures)} 个):")
            for f in self.figures:
                lines.append(f"  • {f}")
        if self.output_files:
            lines.append(f"生成文件 ({len(self.output_files)} 个):")
            for f in self.output_files:
                lines.append(f"  • {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

# Preamble injected at the top of every executed script
_PREAMBLE = """
import os, sys, warnings
import numpy as np
warnings.filterwarnings('ignore')

# Add project root to path so seismo_* modules are importable
_proj = os.environ.get('SAGE_PROJECT_ROOT', '.')
if _proj not in sys.path:
    sys.path.insert(0, _proj)

# Import the built-in seismology toolkit
try:
    from seismo_code.toolkit import *   # noqa: F401,F403
except ImportError:
    pass

# matplotlib non-interactive backend
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Directory for this run — all relative file saves land here
_OUTDIR = os.environ.get('SAGE_OUTDIR', '.')

def _savefig(name):
    \"\"\"Save current matplotlib figure to the output directory.\"\"\"
    import matplotlib.pyplot as _plt
    path = os.path.join(_OUTDIR, name)
    _plt.savefig(path, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f'[FIGURE] {path}')
    return path

# Make savefig available as a helper
savefig = _savefig
"""


def _cancel_requested(cancel_event: Optional[Any]) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _terminate_process(proc: subprocess.Popen):
    """Terminate a child process and its process group when available."""
    def _signal(sig):
        if proc.poll() is not None:
            return
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), sig)
        elif sig == signal.SIGTERM:
            proc.terminate()
        else:
            proc.kill()

    try:
        _signal(signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    try:
        proc.wait(timeout=0.3)
    except Exception:
        try:
            _signal(signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=0.7)
        except Exception:
            pass


def _communicate_with_cancel(
    proc: subprocess.Popen,
    timeout: int,
    cancel_event: Optional[Any] = None,
) -> tuple[bool, str, str, int, str]:
    """Wait for process completion while honoring cancellation."""
    deadline = time.monotonic() + timeout
    while True:
        if _cancel_requested(cancel_event):
            _terminate_process(proc)
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return False, stdout or "", stderr or "", proc.returncode or -15, "Execution cancelled"

        if time.monotonic() >= deadline:
            _terminate_process(proc)
            try:
                stdout, stderr = proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            return False, stdout or "", stderr or "", proc.returncode or -9, f"执行超时（>{timeout}s）"

        if proc.poll() is not None:
            stdout, stderr = proc.communicate(timeout=2)
            return proc.returncode == 0, stdout or "", stderr or "", proc.returncode or 0, ""

        time.sleep(0.05)


def execute_code(
    code: str,
    project_root: Optional[str] = None,
    timeout: int = 60,
    keep_dir: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
    python_executable: Optional[str] = None,
    cancel_event: Optional[Any] = None,
) -> ExecutionResult:
    """
    Execute Python code in an isolated subprocess.

    Parameters
    ----------
    code : str
        Python source code to execute.
    project_root : str, optional
        Path to the SAGE project root (added to PYTHONPATH).
    timeout : int
        Maximum execution time in seconds. Default 60.
    keep_dir : bool
        If True, do not delete the temp directory after execution.
    extra_env : dict, optional
        Additional environment variables.

    Returns
    -------
    ExecutionResult
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent)

    # Create temp directory for this run
    tmp = tempfile.mkdtemp(prefix="sage_exec_")
    script_path = os.path.join(tmp, "run.py")

    # Assemble full script
    full_code = _PREAMBLE + "\n" + code
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full_code)

    # Build environment
    env = os.environ.copy()
    env["SAGE_PROJECT_ROOT"] = project_root
    env["SAGE_OUTDIR"] = tmp
    env["MPLBACKEND"] = "Agg"
    # Add project root to PYTHONPATH
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{existing_pp}" if existing_pp else project_root
    # Limit BLAS/OpenMP thread counts to 1 — prevents SIGSEGV caused by
    # inheriting a forked thread pool when the parent loaded numpy/PyTorch.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # avoids OMP abort on macOS
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"   # macOS: prevent SIGSEGV on fork
    if extra_env:
        env.update(extra_env)

    # Execute
    try:
        python_cmd = python_executable or sys.executable
        proc = subprocess.Popen(
            [python_cmd, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        success, stdout, stderr, returncode, error = _communicate_with_cancel(
            proc, timeout, cancel_event)
        if not success:
            # Extract the last traceback line as short error
            if not error:
                lines = stderr.strip().splitlines()
                error = lines[-1] if lines else f"Exit code {returncode}"
    except Exception as e:
        success = False
        stdout = ""
        stderr = str(e)
        error = str(e)

    # Collect generated files
    figures = []
    output_files = []

    # Find files mentioned in stdout as [FIGURE] ...
    for line in stdout.splitlines():
        if line.startswith("[FIGURE] "):
            fig_path = line[len("[FIGURE] "):].strip()
            if os.path.isfile(fig_path):
                figures.append(fig_path)

    # Also scan the temp directory for any image/data files not already captured
    if os.path.isdir(tmp):
        for fname in sorted(os.listdir(tmp)):
            fpath = os.path.join(tmp, fname)
            if fname == "run.py":
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in (".png", ".pdf", ".svg") and fpath not in figures:
                figures.append(fpath)
            elif ext not in (".py",) and os.path.isfile(fpath) and fpath not in output_files:
                output_files.append(fpath)

    if not keep_dir and not figures and not output_files:
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    return ExecutionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        error=error,
        figures=figures,
        output_files=output_files,
        exec_dir=tmp,
    )


def execute_bash(
    script: str,
    project_root: Optional[str] = None,
    timeout: int = 180,
    keep_dir: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
    cancel_event: Optional[Any] = None,
) -> ExecutionResult:
    """
    Execute a bash script in an isolated temp directory.

    Designed for shell-native tools.  The script runs with
    SAGE_OUTDIR set to the temp directory so relative file writes land there.
    All PNG / PDF / SVG files produced in that directory are collected as figures.

    Parameters
    ----------
    script : str
        Complete bash script.  A ``#!/bin/bash`` shebang and ``set -e`` are
        prepended automatically if the script doesn't start with ``#!``.
    project_root : str, optional
        Path added to PATH so project-local tools are discoverable.
    timeout : int
        Maximum execution time in seconds.  Default 180 for heavier CLI jobs.
    keep_dir : bool
        Keep temp directory after execution (useful for debugging).
    extra_env : dict, optional
        Additional environment variables forwarded to the script.

    Returns
    -------
    ExecutionResult
    """
    if project_root is None:
        project_root = str(Path(__file__).parent.parent)

    tmp = tempfile.mkdtemp(prefix="sage_bash_")
    script_path = os.path.join(tmp, "run.sh")

    # Prepend shebang only. Do not force `set -e`: many scientific CLI tools
    # use non-zero statuses for recoverable probes before a later fallback succeeds.
    header = ""
    if not script.strip().startswith("#!"):
        header = "#!/bin/bash\n"
    full_script = header + script

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(full_script)
    os.chmod(script_path, 0o755)

    # Build environment — inherit current env, then layer our additions
    env = os.environ.copy()
    env["SAGE_PROJECT_ROOT"] = project_root
    env["SAGE_OUTDIR"] = tmp              # scripts cd here or write relative paths
    env["MPLBACKEND"] = "Agg"
    # Prevent BLAS fork-safety SIGSEGV (same as execute_code)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"   # macOS: prevent SIGSEGV on fork

    # Common scientific CLI tools installed through Homebrew may need their
    # sibling lib directory visible in non-interactive child processes on macOS.
    lib_candidates = []
    for prefix in ("/opt/homebrew", "/usr/local"):
        lib = os.path.join(prefix, "lib")
        if os.path.isdir(lib):
            lib_candidates.append(lib)
    if lib_candidates:
        joined = os.pathsep.join(lib_candidates)
        cur_dyld = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = f"{joined}{os.pathsep}{cur_dyld}" if cur_dyld else joined
        cur_fall = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{joined}{os.pathsep}{cur_fall}" if cur_fall else joined

    if extra_env:
        env.update(extra_env)

    # Execute
    try:
        proc = subprocess.Popen(
            ["bash", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        success, stdout, stderr, returncode, error = _communicate_with_cancel(
            proc, timeout, cancel_event)
        if not success:
            if not error:
                lines = stderr.strip().splitlines()
                error = lines[-1] if lines else f"Exit code {returncode}"
    except Exception as exc:
        success = False
        stdout  = ""
        stderr  = str(exc)
        error   = str(exc)

    # Collect generated files from the execution temp directory and, when the
    # caller overrides SAGE_OUTDIR, from that shared output directory too.
    figures:      List[str] = []
    output_files: List[str] = []

    scan_dirs = [tmp]
    outdir = env.get("SAGE_OUTDIR", tmp)
    if outdir and outdir not in scan_dirs:
        scan_dirs.append(outdir)

    for scan_dir in scan_dirs:
        if os.path.isdir(scan_dir):
            for fname in sorted(os.listdir(scan_dir)):
                fpath = os.path.join(scan_dir, fname)
                if fname in ("run.sh",) or not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1].lower()
                if ext in (".png", ".pdf", ".svg") and fpath not in figures:
                    figures.append(fpath)
                elif ext not in (".sh",) and fpath not in output_files:
                    output_files.append(fpath)

    # Emit [FIGURE] markers to stdout so helpers.serialize_code_result can find them
    for fig in figures:
        stdout += f"\n[FIGURE] {fig}"

    if not keep_dir and not figures and not output_files:
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    return ExecutionResult(
        success=success,
        stdout=stdout,
        stderr=stderr,
        error=error,
        figures=figures,
        output_files=output_files,
        exec_dir=tmp,
    )
