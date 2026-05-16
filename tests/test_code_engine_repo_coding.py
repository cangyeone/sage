"""Tests for repository-coding helpers in CodeEngine."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

PROJECT_ROOT = Path(__file__).parent.parent
WEB_APP_DIR = PROJECT_ROOT / "web_app"
for path in (str(WEB_APP_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from seismo_code.code_engine import CodeEngine
from seismo_code.ce_prompts import _CODEGEN_SYSTEM
from seismo_code.safe_executor import ExecutionResult
from seismo_code.repo_intelligence import build_repo_intelligence
from routes import code as code_routes
import helpers as web_helpers


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

    def test_builtin_repo_intelligence_maps_symbols(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "alpha.py").write_text(
                "class AlphaService:\n"
                "    def compute_value(self, x):\n"
                "        return x + 1\n",
                encoding="utf-8",
            )
            (root / "pkg" / "beta.py").write_text(
                "from pkg.alpha import AlphaService\n\n"
                "def call_alpha():\n"
                "    return AlphaService().compute_value(2)\n",
                encoding="utf-8",
            )

            context = build_repo_intelligence(
                root,
                "修改 AlphaService.compute_value 并同步测试",
                ["pkg/alpha.py", "pkg/beta.py"],
            )

            self.assertTrue(context.available, context.error)
            self.assertIn("AlphaService", context.repo_map)
            self.assertIn("compute_value", context.repo_map)
            self.assertIn("pkg/alpha.py", context.ranked_files)

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

    def test_repo_validation_runs_related_existing_tests(self):
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
                "fix add behavior in maths",
                ["pkg/maths.py"],
            )

            self.assertTrue(ok, note)
            self.assertIn("py_compile passed", note)
            self.assertIn("pytest passed", note)
            self.assertIn("changed/related", note)

    def test_repo_validation_requires_tests_for_python_behavior_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pkg").mkdir()
            (root / "pkg" / "maths.py").write_text(
                "def add(a, b):\n"
                "    return a + b\n",
                encoding="utf-8",
            )
            engine = self._engine(tmp)

            ok, note = engine._run_repo_validation(
                "fix add behavior in maths",
                ["pkg/maths.py"],
            )

            self.assertFalse(ok)
            self.assertIn("no focused tests", note)

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
            self.assertIn("SAGE Repo Map", context)
            self.assertIn("SAGE Built-in Editing Discipline", context)
            self.assertIn("delete focused tests", _CODEGEN_SYSTEM)
            self.assertIn("[SAGE_AGENT] located", context)

    def test_local_api_context_documents_bvalue_and_plot_signatures(self):
        engine = self._engine(str(PROJECT_ROOT))

        context = engine._build_local_api_context(
            "帮我随机生成10000个0-7之间的随机数，并且把这个假设为震级计算一下b值"
        )

        self.assertIn("calc_bvalue_mle", context)
        self.assertIn("BvalueResult", context)
        self.assertIn("b_uncertainty", context)
        self.assertIn("plot_gr(result, output_path)", context)
        self.assertIn("plot_all(result, catalog, output_prefix)", context)

    def test_standalone_plot_request_with_profile_context_is_not_repo_task(self):
        message = (
            "帮我用python 绘制一下中国地形图\n\n"
            "===== Long-term user profile (soft context; do not mention unless useful) =====\n"
            "用户正在优化 coding agent，希望它能添加测试、更新代码和定位函数。"
        )

        self.assertFalse(CodeEngine._looks_like_repo_task(message))
        self.assertFalse(CodeEngine._repo_behavior_change_request(message))

    def test_repo_edit_request_still_detected_with_primary_text(self):
        message = "帮我优化 coding agent，让它能更新多个文件并写单元测试"

        self.assertTrue(CodeEngine._looks_like_repo_task(message))
        self.assertTrue(CodeEngine._repo_behavior_change_request(message))

    def test_active_project_marker_promotes_feature_requests_to_repo_task(self):
        message = (
            "实现登录功能\n\n"
            "===== Active coding project =====\n"
            "Project root: /tmp/myapp\n"
            "Use this as the primary repository/workspace for code search, edits, builds, and tests.\n"
        )

        self.assertTrue(CodeEngine._looks_like_repo_task(message))

    def test_full_project_validation_runs_pytest_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tests").mkdir()
            (root / "tests" / "test_ok.py").write_text(
                "def test_ok():\n"
                "    assert True\n",
                encoding="utf-8",
            )
            engine = self._engine(tmp)

            ok, note = engine._run_repo_validation("运行全量测试", [])

            self.assertTrue(ok, note)
            self.assertIn("full validation passed", note)

    def test_web_code_engine_uses_project_root_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = web_helpers.get_code_engine(
                "unit_project_session",
                {"provider": "test", "api_base": "http://test", "model": "test"},
                tmp,
            )

            self.assertEqual(Path(engine.project_root).resolve(), Path(tmp).resolve())

    def test_engineering_plan_contains_api_and_unit_test_details(self):
        engine = self._engine(str(PROJECT_ROOT))
        plan_text = (
            "## Engineering Plan\n"
            "### Route\n- locate route and helper\n"
            "### Files\n- `web_app/routes/chat.py`: update builder\n"
            "### API Details\n- `_build_rag_messages(data)`: returns messages, sources, config\n"
            "### Unit Tests\n- `tests/test_chat_references.py`: assert sources include skills\n"
            "### Validation\n- `python -m pytest tests/test_chat_references.py`: focused regression\n"
        )

        with patch("seismo_code.code_engine._call_llm", return_value=plan_text):
            plan = engine._build_engineering_plan(
                "优化 QA sources",
                repo_ctx="## Repository Context\nweb_app/routes/chat.py",
                local_api_ctx="## Local API reference",
            )

        self.assertIn("### API Details", plan)
        self.assertIn("_build_rag_messages", plan)
        self.assertIn("### Unit Tests", plan)
        self.assertIn("tests/test_chat_references.py", plan)

    def test_engineering_plan_fallback_requires_api_and_tests(self):
        engine = self._engine(str(PROJECT_ROOT))

        with patch("seismo_code.code_engine._call_llm", side_effect=RuntimeError("llm down")):
            plan = engine._build_engineering_plan(
                "实现一个新 API",
                repo_ctx="## Repository Context\npkg/api.py",
            )

        self.assertIn("## Engineering Plan", plan)
        self.assertIn("### API Details", plan)
        self.assertIn("### Unit Tests", plan)
        self.assertIn("py_compile", plan)

    def test_engineering_plan_persists_and_loads_for_debug(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._engine(tmp)
            plan = (
                "## Engineering Plan\n"
                "### API Details\n- `add(a, b)`: returns sum\n"
                "### Unit Tests\n- `tests/test_maths.py`: assert add\n"
            )

            path = engine._persist_engineering_plan(
                plan,
                exec_dir=tmp,
                reason="unit test",
            )
            loaded = engine._load_engineering_plan_for_debug()

            self.assertTrue(Path(path).is_file())
            self.assertIn("### API Details", loaded)
            self.assertIn("tests/test_maths.py", loaded)

    def test_mini_test_rejects_replacing_uniform_random_magnitudes(self):
        engine = self._engine(str(PROJECT_ROOT))
        code = (
            "import numpy as np\n"
            "magnitudes = np.random.exponential(size=10000)\n"
            "print('[SAGE_TEST] Generated 10000 magnitudes from Gutenberg-Richter')\n"
        )
        exec_res = ExecutionResult(success=True, stdout="[SAGE_TEST] ok")

        ok, reason = engine._mini_test_ok(
            "帮我随机生成10000个0-7之间的随机数，并且把这个假设为震级计算一下b值",
            code,
            exec_res,
        )

        self.assertFalse(ok)
        self.assertIn("do not replace", reason)

    def test_mini_test_rejects_wrong_plot_all_for_bvalue(self):
        engine = self._engine(str(PROJECT_ROOT))
        code = (
            "from seismo_stats.plotting import plot_all\n"
            "plot_all(magnitudes, Mc, b_value=b_val, outfile='fmd_random.png')\n"
            "print('[SAGE_TEST] ok')\n"
        )
        exec_res = ExecutionResult(success=True, stdout="[SAGE_TEST] ok")

        ok, reason = engine._mini_test_ok("计算b值并绘制FMD", code, exec_res)

        self.assertFalse(ok)
        self.assertIn("plot_gr", reason)


class TestCodeRouteRepoGuard(unittest.TestCase):
    def test_code_draft_request_routes_to_code_draft(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="code_draft") as mock_router,
        ):
            for message in ("帮我写一个计算b值的程序", "给我一个计算b值的程序。"):
                with self.subTest(message=message):
                    with app.test_request_context(
                        "/api/chat/route",
                        method="POST",
                        json={
                            "message": message,
                            "history": [],
                            "kb_has_docs": False,
                        },
                    ):
                        resp = code_routes.chat_route()
                        data = resp.get_json()

                    self.assertEqual(data["intent"], "code_draft")
            self.assertGreaterEqual(mock_router.call_count, 2)
            self.assertIn("code_draft", mock_router.call_args_list[-1].args[0][0]["content"])

    def test_execute_with_path_still_routes_to_code(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="code") as mock_router,
        ):
            with app.test_request_context(
                "/api/chat/route",
                method="POST",
                json={
                    "message": "帮我运行这个程序计算b值 /tmp/catalog.csv",
                    "history": [],
                    "kb_has_docs": False,
                },
            ):
                resp = code_routes.chat_route()
                data = resp.get_json()

        self.assertEqual(data["intent"], "code")
        self.assertEqual(mock_router.call_count, 1)

    def test_codebase_location_request_routes_to_code(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="code") as mock_router,
        ):
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
        self.assertEqual(mock_router.call_count, 1)

    def test_python_plot_artifact_request_is_forced_to_code(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="qa") as mock_router,
        ):
            with app.test_request_context(
                "/api/chat/route",
                method="POST",
                json={
                    "message": "帮我用python 绘制一下中国地形图",
                    "history": [],
                    "kb_has_docs": False,
                },
            ):
                resp = code_routes.chat_route()
                data = resp.get_json()

        self.assertEqual(data["intent"], "code")
        self.assertEqual(mock_router.call_count, 1)

    def test_gui_operation_request_is_forced_to_code(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="qa") as mock_router,
        ):
            with app.test_request_context(
                "/api/chat/route",
                method="POST",
                json={
                    "message": "帮我截图并点击 GUI 里的确定按钮",
                    "history": [],
                    "kb_has_docs": False,
                },
            ):
                resp = code_routes.chat_route()
                data = resp.get_json()

        self.assertEqual(data["intent"], "code")
        self.assertEqual(mock_router.call_count, 1)

    def test_previous_code_refinement_request_is_forced_to_code(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", return_value="qa") as mock_router,
        ):
            with app.test_request_context(
                "/api/chat/route",
                method="POST",
                json={
                    "message": "注意标题支持MAC系统的中文显示。",
                    "history": [
                        {"role": "user", "content": "帮我用python 绘制一下中国地形图"},
                        {"role": "assistant", "content": "执行成功\n输出:\n[SAGE_TEST] 地形图生成成功\n生成了 1 张图像"},
                    ],
                    "kb_has_docs": False,
                    "last_intent_was_code": True,
                },
            ):
                resp = code_routes.chat_route()
                data = resp.get_json()

        self.assertEqual(data["intent"], "code")
        self.assertEqual(mock_router.call_count, 1)

    def test_router_fallback_does_not_default_questions_to_qa(self):
        app = Flask(__name__)
        with (
            patch.object(
                code_routes,
                "get_llm_config",
                return_value={"provider": "test", "api_base": "http://test", "model": "test"},
            ),
            patch("helpers.llm_call", side_effect=RuntimeError("router down")),
        ):
            with app.test_request_context(
                "/api/chat/route",
                method="POST",
                json={
                    "message": "什么是 b 值？",
                    "history": [],
                    "kb_has_docs": True,
                },
            ):
                resp = code_routes.chat_route()
                data = resp.get_json()

        self.assertEqual(data["intent"], "chat")
        self.assertTrue(data.get("fallback"))


if __name__ == "__main__":
    unittest.main()
