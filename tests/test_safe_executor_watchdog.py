from seismo_code.safe_executor import execute_bash, execute_code


def test_execute_code_emits_watchdog_progress_while_running():
    progress = []
    result = execute_code(
        "import time\nprint('started', flush=True)\ntime.sleep(0.7)\nprint('[SAGE_TEST] done', flush=True)\n",
        timeout=20,
        keep_dir=True,
        progress_cb=progress.append,
        progress_interval=0.2,
    )

    assert result.success
    assert "[SAGE_TEST] done" in result.stdout
    assert any("watchdog: process still running" in item for item in progress)


def test_execute_bash_emits_watchdog_progress_while_running():
    progress = []
    result = execute_bash(
        "echo started\nsleep 0.7\necho '[SAGE_TEST] done'\n",
        timeout=20,
        keep_dir=True,
        progress_cb=progress.append,
        progress_interval=0.2,
    )

    assert result.success
    assert "[SAGE_TEST] done" in result.stdout
    assert any("watchdog: process still running" in item for item in progress)
