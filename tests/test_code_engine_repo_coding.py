"""Tests for repository-coding helpers in CodeEngine."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from seismo_code.code_engine import CodeEngine
from seismo_code.ce_prompts import _CODEGEN_SYSTEM
from routes import code as code_routes


class TestRepoCodingHelpers(unittest.TestCase):
    def _engine(self, root: str) -> CodeEngine:
        return CodeEngine(
            llm_config={"provider": "test", "api_base": "http://test", "model": "test"},
            project_root=root,
        )

    def test_rg_hits_and_symbol_index_find_relevant_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "alpha.py").write_text(
                "class AlphaService:\n"
                "    def compute_value(self, x):\n"
                "        return x + 1\n",
                encoding="utf-8",
            )
            engine = self._engine(tmp)

            hits = engine._repo_rg_hits("where is compute_value implemented?")
            symbols = engine._repo_symbol_index()

            self.assertTrue(any("compute_value" in h for h in hits))
            self.assertTrue(any("AlphaService" in s for s in symbols))

    def test_repo_validation_runs_py_compile_and_targeted_pytest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "tests").mkdir()
            (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
            (root / "pkg" / "maths.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_maths.py").write_text(
                "from pkg.maths import add\n\n"
                "def test_add():\n"
                "    assert add(2, 3) == 5\n",
                encoding="utf-8",
            )
            engine = self._engine(tmp)

            ok, note = engine._run_repo_validation(
                "implement add function",
                ["pkg/maths.py", "tests/test_maths.py"],
            )

            self.assertTrue(ok, note)
            self.assertIn("py_compile passed", note)
            self.assertIn("pytest passed", note)

    def test_repo_context_allows_reasoned_test_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_old.py").write_text(
                "def test_old_behavior():\n"
                "    assert True\n",
                encoding="utf-8",
            )
            engine = self._engine(tmp)

            context = engine._build_repo_context("删除过时的 test_old_behavior 测试")

            self.assertIn("delete focused unit tests", context)
            self.assertIn("print the reason", context)
            self.assertIn("delete focused tests", _CODEGEN_SYSTEM)


class TestCodeRouteRepoGuard(unittest.TestCase):
    def test_codebase_location_request_routes_to_code(self):
        app = Flask(__name__)
        with app.test_request_context(
            "/api/chat/route",
            method="POST",
            json={
                "message": "帮我定位 CodeEngine 的 run 函数在哪里",
                "history": [],
                "kb_has_docs": False,
            },
        ):
            resp = code_routes.chat_route()
            data = resp.get_json()

        self.assertEqual(data["intent"], "code")
        self.assertEqual(data["rule"], "codebase_guard")


if __name__ == "__main__":
    unittest.main()
