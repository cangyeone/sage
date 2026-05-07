"""Tests for cancellation, run records, and deterministic smoke demo."""

from __future__ import annotations

import sys
import csv
import shutil
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
import app as web_app
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
                self.assertEqual(result["generation"]["random_seed"], 20260507)
                self.assertIn("station_truth.csv", [Path(p).name for p in result["artifacts"]])
                for artifact in result["artifacts"]:
                    self.assertTrue(Path(artifact).is_file(), artifact)

                truth_path = Path(out_tmp) / "station_truth.csv"
                with truth_path.open(newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 3)
                self.assertEqual(rows[0]["station"], "STA01")
                self.assertEqual(float(rows[0]["true_p_s"]), 6.20)

                rec = run_records.get_run(result["run_id"])
                self.assertEqual(rec["kind"], "smoke_demo")
                self.assertEqual(rec["status"], "succeeded")
                self.assertIn("generation", rec["metadata"])
            finally:
                run_records.RUN_RECORD_DIR = old_dir

    def test_smoke_route_records_and_serves_artifact(self):
        with tempfile.TemporaryDirectory() as rec_tmp:
            old_dir = run_records.RUN_RECORD_DIR
            run_records.RUN_RECORD_DIR = Path(rec_tmp)
            run_records.RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
            client = web_app.app.test_client()
            output_dir = None
            try:
                resp = client.post("/api/smoke_demo/run", json={})
                self.assertEqual(resp.status_code, 200)
                body = resp.get_json()
                self.assertTrue(body["ok"])
                output_dir = Path(body["output_dir"])

                runs = client.get("/api/runs?limit=5").get_json()
                self.assertTrue(runs["ok"])
                self.assertTrue(any(r["run_id"] == body["run_id"] for r in runs["runs"]))

                artifact = client.get(f"/api/runs/{body['run_id']}/artifact/0")
                self.assertEqual(artifact.status_code, 200)
                self.assertGreater(len(artifact.data), 0)
            finally:
                run_records.RUN_RECORD_DIR = old_dir
                if output_dir and output_dir.exists():
                    shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
