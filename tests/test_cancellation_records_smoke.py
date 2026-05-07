"""Tests for cancellation, run records, and deterministic smoke demo."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_records
from examples.sage_smoke_demo import run_smoke_demo
from seismo_code.safe_executor import execute_code


class TestCancellation(unittest.TestCase):
    def test_execute_code_cancel_kills_child(self):
        cancel_event = threading.Event()

        def cancel_soon():
            time.sleep(0.4)
            cancel_event.set()

        threading.Thread(target=cancel_soon, daemon=True).start()
        started = time.monotonic()
        result = execute_code(
            "import time\nwhile True:\n    time.sleep(0.1)\n",
            timeout=10,
            keep_dir=True,
            cancel_event=cancel_event,
        )
        elapsed = time.monotonic() - started

        self.assertFalse(result.success)
        self.assertIn("cancel", result.error.lower())
        self.assertLess(elapsed, 4)


class TestRunRecordsAndSmokeDemo(unittest.TestCase):
    def test_run_records_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = run_records.RUN_RECORD_DIR
            run_records.RUN_RECORD_DIR = Path(tmp)
            run_records.RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
            try:
                run_id = run_records.start_run("unit", request="test")
                run_records.append_event(run_id, "phase", "message", {"x": 1})
                run_records.finish_run(run_id, "succeeded", result={"ok": True})

                rec = run_records.get_run(run_id)
                self.assertEqual(rec["status"], "succeeded")
                self.assertEqual(rec["events"][0]["phase"], "phase")
                self.assertEqual(run_records.list_runs()[0]["run_id"], run_id)
            finally:
                run_records.RUN_RECORD_DIR = old_dir

    def test_smoke_demo_outputs_artifacts_and_record(self):
        with tempfile.TemporaryDirectory() as out_tmp, tempfile.TemporaryDirectory() as rec_tmp:
            old_dir = run_records.RUN_RECORD_DIR
            run_records.RUN_RECORD_DIR = Path(rec_tmp)
            run_records.RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
            try:
                result = run_smoke_demo(Path(out_tmp))

                self.assertTrue(result["ok"])
                self.assertEqual(result["catalog"]["n_picks"], 6)
                for artifact in result["artifacts"]:
                    self.assertTrue(Path(artifact).is_file(), artifact)

                rec = run_records.get_run(result["run_id"])
                self.assertEqual(rec["kind"], "smoke_demo")
                self.assertEqual(rec["status"], "succeeded")
            finally:
                run_records.RUN_RECORD_DIR = old_dir


if __name__ == "__main__":
    unittest.main()
