"""Tests for CodeEngine artifact registration and conversation cleanup."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from helpers import serialize_code_result
from routes import chat as chat_routes


class TestChatArtifactCleanup(unittest.TestCase):
    def test_serialize_code_result_records_artifact_paths_and_exec_dir(self):
        with tempfile.TemporaryDirectory(prefix="sage_exec_") as tmp:
            root = Path(tmp)
            script = root / "analysis.py"
            plan = root / "engineering_plan.md"
            fig = root / "figure.png"
            script.write_text("print('ok')\n", encoding="utf-8")
            plan.write_text("## Engineering Plan\n", encoding="utf-8")
            fig.write_bytes(b"\x89PNG\r\n\x1a\n")
            exec_result = SimpleNamespace(
                stdout="[SAGE_TEST] ok",
                output_files=[str(plan)],
                figures=[str(fig)],
                exec_dir=str(root),
            )
            result = SimpleNamespace(
                success=True,
                response="ok",
                code="print('ok')",
                stdout="[SAGE_TEST] ok",
                figures=[str(fig)],
                output_files=[str(plan)],
                debug_trace=[],
                plan=["do it"],
                attempts=1,
                script_path=str(script),
                exec_result=exec_result,
            )

            payload = serialize_code_result(result, skill_used="")

        self.assertIn("artifact_paths", payload)
        self.assertIn(str(script), payload["artifact_paths"])
        self.assertIn(str(plan), payload["artifact_paths"])
        self.assertIn(str(fig), payload["artifact_paths"])
        self.assertEqual(payload["exec_dir"], str(root))

    def test_cleanup_deletes_safe_exec_dir_but_not_project_files(self):
        with tempfile.TemporaryDirectory(prefix="sage_exec_") as tmp:
            root = Path(tmp)
            plan = root / "engineering_plan.md"
            output = root / "result.txt"
            plan.write_text("## Engineering Plan\n", encoding="utf-8")
            output.write_text("result\n", encoding="utf-8")

            unsafe = PROJECT_ROOT / ".sage_runtime" / "unsafe_cleanup_probe.txt"
            unsafe.parent.mkdir(parents=True, exist_ok=True)
            unsafe.write_text("keep me\n", encoding="utf-8")
            self.addCleanup(lambda: unsafe.unlink(missing_ok=True))

            conv = {
                "id": "conv_cleanup",
                "messages": [
                    {
                        "kind": "code_result",
                        "data": {
                            "exec_dir": str(root),
                            "artifact_paths": [str(plan), str(output), str(unsafe)],
                        },
                    }
                ],
            }

            removed = chat_routes._cleanup_conversation_code_artifacts(conv)

            self.assertGreaterEqual(removed, 1)
            self.assertFalse(root.exists())
            self.assertTrue(unsafe.exists())


if __name__ == "__main__":
    unittest.main()
