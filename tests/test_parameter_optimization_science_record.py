"""Tests for saving parameter optimization runs as science-analysis evidence."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_APP = ROOT / "web_app"
for path in (str(ROOT), str(WEB_APP)):
    if path not in sys.path:
        sys.path.insert(0, path)

from routes.parameter_optimization import _write_science_analysis_record  # noqa: E402
from routes.chat import _science_guess_file_role  # noqa: E402


class TestParameterOptimizationScienceRecord(unittest.TestCase):
    def test_writes_reusable_science_analysis_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            output_dir = project_root / "outputs" / "science_analysis_agent" / "parameter_optimization" / "p" / "j"
            output_dir.mkdir(parents=True)
            history = output_dir / "optimization_history.csv"
            history.write_text("trial,score\n1,0.8\n", encoding="utf-8")
            summary = {
                "success": True,
                "attempts": 1,
                "figures": [],
                "output_files": [str(history)],
                "output_dir": str(output_dir),
            }

            paths = _write_science_analysis_record(
                project_root=project_root,
                project_id="proj_demo",
                job_id="opt_123",
                data={"objective": "maximize validation F1", "workflow": [{"name": "model"}]},
                summary=summary,
                progress=[{"phase": "code", "message": "ran mini test"}],
                output_dir=output_dir,
            )

            md_path = Path(paths["science_analysis_record"])
            json_path = Path(paths["science_analysis_manifest"])
            self.assertTrue(md_path.exists())
            self.assertTrue(json_path.exists())
            self.assertIn("maximize validation F1", md_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "parameter_optimization_run")

    def test_science_profiler_recognizes_optimization_record(self):
        p = Path("science_analysis_inputs/parameter_optimization/proj/opt/parameter_optimization_run.md")
        self.assertEqual(_science_guess_file_role(p, "best_parameters.json"), "parameter_optimization_evidence")


if __name__ == "__main__":
    unittest.main()
