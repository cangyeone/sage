"""
Regression tests for Web-layer security helpers.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from helpers import path_is_within_root, safe_child_path
from state import UPLOAD_FOLDER_CHAT
from routes import chat as chat_routes
from routes import knowledge as knowledge_routes


class TestPathSandboxHelpers(unittest.TestCase):
    def test_sibling_prefix_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            sibling = Path(tmp) / "data2"
            root.mkdir()
            sibling.mkdir()

            self.assertTrue(path_is_within_root(root / "file.txt", root))
            self.assertFalse(path_is_within_root(sibling / "file.txt", root))

    def test_safe_child_path_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "uploads"
            root.mkdir()

            with self.assertRaises(ValueError):
                safe_child_path(root, "../outside.pdf")


class TestCommandBuilders(unittest.TestCase):
    def test_pick_command_keeps_user_path_as_single_argument(self):
        injected_path = "/tmp/waveforms; touch /tmp/sage_pwned"
        argv = chat_routes._build_pick_command(
            {"input_dir": injected_path, "device": "cpu"},
            "pick_test",
        )

        self.assertIsInstance(argv, list)
        self.assertEqual(argv[argv.index("-i") + 1], injected_path)

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            chat_routes._build_pick_command(
                {"input_dir": "/tmp/waveforms", "device": "cpu; touch x"},
                "pick_test",
            )

    def test_run_task_invokes_subprocess_without_shell(self):
        task_id = "security_test"
        chat_routes.tasks[task_id] = {"id": task_id, "status": "queued"}

        completed = subprocess.CompletedProcess(["python", "--version"], 0, "ok", "")
        with patch.object(chat_routes.subprocess, "run", return_value=completed) as run:
            chat_routes.run_task(task_id, ["python", "--version"], "security")

        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertEqual(chat_routes.tasks[task_id]["status"], "completed")
        chat_routes.tasks.pop(task_id, None)


class TestUploadPaths(unittest.TestCase):
    def test_chat_pdf_upload_path_is_sanitized_and_sandboxed(self):
        path, doc_name = chat_routes._safe_pdf_upload_path("../../paper.pdf", "../bad")

        self.assertEqual(doc_name, "paper.pdf")
        self.assertTrue(path_is_within_root(path, UPLOAD_FOLDER_CHAT))

    def test_chinese_pdf_name_keeps_pdf_extension(self):
        path, doc_name = chat_routes._safe_pdf_upload_path("测试.pdf", "default")

        self.assertEqual(doc_name, "upload.pdf")
        self.assertTrue(path_is_within_root(path, UPLOAD_FOLDER_CHAT))

    def test_knowledge_pdf_upload_path_is_sanitized_and_sandboxed(self):
        path, doc_name = knowledge_routes._safe_pdf_upload_path("../../paper.pdf")

        self.assertEqual(doc_name, "paper.pdf")
        self.assertTrue(path_is_within_root(path, UPLOAD_FOLDER_CHAT))

    def test_non_pdf_upload_is_rejected(self):
        with self.assertRaises(ValueError):
            chat_routes._safe_pdf_upload_path("notes.txt", "default")


if __name__ == "__main__":
    unittest.main()
