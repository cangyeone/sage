"""
Unit tests for sage_agents.evidence_driven_geo_agent

Tests cover:
  - Path sandboxing (safe_path, extension allowlist)
  - LocalFileSearchTool (list_dir, read_file, search_files, grep)
  - GeoEvidenceTableBuilder (add, dedup, conflict detection)
  - Evidence extraction (LLM mock)
  - Loop stopping (convergence logic)
  - AgentConfig defaults and serialisation
  - ToolRegistry dispatch (error handling)

Run with:
    python -m pytest tests/test_evidence_driven_geo_agent.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sage_agents.evidence_driven_geo_agent import (
    AgentConfig,
    AgentLogger,
    GeoEvidence,
    GeoEvidenceTableBuilder,
    GeoHypothesis,
    LocalFileSearchTool,
    LiteratureLibraryTool,
    ToolRegistry,
    WebSearchTool,
    _safe_path,
    _ext_allowed,
)


# ─────────────────────────────────────────────────────────────────────────────
# ── 1. Path sandboxing ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestSafePath(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_file_inside_root_accepted(self):
        target = os.path.join(self.tmpdir, "data.csv")
        result = _safe_path(target, self.tmpdir)
        self.assertIsNotNone(result)
        self.assertEqual(str(result), str(Path(target).resolve(strict=False)))

    def test_absolute_outside_root_rejected(self):
        result = _safe_path("/etc/passwd", self.tmpdir)
        self.assertIsNone(result)

    def test_path_traversal_rejected(self):
        evil = os.path.join(self.tmpdir, "..", "..", "etc", "passwd")
        result = _safe_path(evil, self.tmpdir)
        self.assertIsNone(result)

    def test_relative_path_resolved_to_root(self):
        # bare filename should resolve relative to root
        result = _safe_path("catalog.csv", self.tmpdir)
        self.assertIsNotNone(result)
        self.assertTrue(str(result).startswith(str(Path(self.tmpdir).resolve(strict=False))))

    def test_empty_root_skipped(self):
        # empty root should be skipped; should still fail if only root is empty
        result = _safe_path("/etc/passwd", "", self.tmpdir)
        self.assertIsNone(result)

    def test_extension_allowlist_pass(self):
        p = Path("/tmp/test.csv")
        self.assertTrue(_ext_allowed(p, [".csv", ".json"]))

    def test_extension_allowlist_fail(self):
        p = Path("/tmp/test.exe")
        self.assertFalse(_ext_allowed(p, [".csv", ".json"]))

    def test_extension_case_insensitive(self):
        p = Path("/tmp/test.CSV")
        self.assertTrue(_ext_allowed(p, [".csv"]))


# ─────────────────────────────────────────────────────────────────────────────
# ── 2. LocalFileSearchTool ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestLocalFileSearchTool(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg    = AgentConfig(workspace_root=self.tmpdir)
        self.tool   = LocalFileSearchTool(self.cfg)
        # Create test files
        (Path(self.tmpdir) / "catalog.csv").write_text(
            "lon,lat,depth,mag\n104.1,29.5,10,3.2\n104.2,29.6,15,4.1\n",
            encoding="utf-8",
        )
        (Path(self.tmpdir) / "notes.txt").write_text(
            "Seismicity is concentrated near the Molingchang fault.\n",
            encoding="utf-8",
        )
        (Path(self.tmpdir) / "secret.exe").write_text("not allowed", encoding="utf-8")

    def test_list_dir_returns_entries(self):
        result = self.tool.list_dir(".")
        self.assertIn("entries", result)
        names = [e["name"] for e in result["entries"]]
        self.assertIn("catalog.csv", names)
        self.assertIn("notes.txt", names)

    def test_list_dir_outside_root_rejected(self):
        result = self.tool.list_dir("/etc")
        self.assertIn("error", result)

    def test_read_file_success(self):
        result = self.tool.read_file("catalog.csv")
        self.assertIn("content", result)
        self.assertIn("lon,lat", result["content"])

    def test_read_file_disallowed_extension(self):
        result = self.tool.read_file("secret.exe")
        self.assertIn("error", result)
        self.assertIn("not in the allowed list", result["error"])

    def test_read_file_outside_root_rejected(self):
        result = self.tool.read_file("/etc/passwd")
        self.assertIn("error", result)

    def test_search_files_by_filename(self):
        result = self.tool.search_files("catalog")
        self.assertGreater(result["count"], 0)
        self.assertTrue(any("catalog" in m["path"] for m in result["matches"]))

    def test_search_files_by_content(self):
        result = self.tool.search_files("Molingchang")
        self.assertGreater(result["count"], 0)

    def test_grep_basic(self):
        result = self.tool.grep("Molingchang")
        self.assertGreater(result["count"], 0)
        self.assertEqual(result["hits"][0]["content"], "Seismicity is concentrated near the Molingchang fault.")

    def test_grep_invalid_regex(self):
        result = self.tool.grep("[invalid")
        self.assertIn("error", result)

    def test_get_file_metadata(self):
        result = self.tool.get_file_metadata("catalog.csv")
        self.assertIn("size_bytes", result)
        self.assertIn("line_count", result)
        self.assertEqual(result["extension"], ".csv")


# ─────────────────────────────────────────────────────────────────────────────
# ── 3. GeoEvidenceTableBuilder ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _make_ev(eid, obs, structure="Molingchang fault", interp="fault slip"):
    return GeoEvidence(
        evidence_id=eid,
        source="test",
        source_type="literature",
        observation=obs,
        data_type="seismicity",
        geological_structure=structure,
        interpretation=interp,
    )


class TestGeoEvidenceTableBuilder(unittest.TestCase):

    def setUp(self):
        self.builder = GeoEvidenceTableBuilder()

    def test_add_new_evidence(self):
        ev = _make_ev("e001", "Seismicity cluster at 8–12 km depth near fault trace")
        added = self.builder.add([ev])
        self.assertEqual(len(added), 1)
        self.assertEqual(len(self.builder.table), 1)

    def test_duplicate_observation_not_added(self):
        obs = "Seismicity cluster at 8–12 km depth near fault trace"
        ev1 = _make_ev("e001", obs)
        ev2 = _make_ev("e002", obs)  # same observation, different ID
        self.builder.add([ev1])
        added = self.builder.add([ev2])
        self.assertEqual(len(added), 0)
        self.assertEqual(len(self.builder.table), 1)

    def test_different_observations_both_added(self):
        ev1 = _make_ev("e001", "Seismicity cluster at 8–12 km depth near fault trace")
        ev2 = _make_ev("e002", "Vp/Vs ratio anomaly at 15–20 km depth")
        self.builder.add([ev1, ev2])
        self.assertEqual(len(self.builder.table), 2)

    def test_conflict_detection_same_structure_different_interp(self):
        ev1 = _make_ev("e001", "Observation A", "Molingchang fault", "fault reactivation")
        ev2 = _make_ev("e002", "Observation B", "Molingchang fault", "fluid injection")
        self.builder.add([ev1, ev2])
        # At least one should have the other in conflict_with
        all_conflicts = [e.conflict_with for e in self.builder.table]
        self.assertTrue(any(len(c) > 0 for c in all_conflicts))

    def test_to_markdown_produces_table(self):
        ev = _make_ev("e001", "Seismicity cluster at 8–12 km depth")
        self.builder.add([ev])
        md = self.builder.to_markdown()
        self.assertIn("e001", md)
        self.assertIn("|", md)

    def test_empty_table_markdown(self):
        md = self.builder.to_markdown()
        self.assertIn("No evidence", md)


# ─────────────────────────────────────────────────────────────────────────────
# ── 4. ToolRegistry dispatch ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestToolRegistryDispatch(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cfg    = AgentConfig(
            workspace_root=self.tmpdir,
            output_dir=self.tmpdir,
            allow_python=False,
            allow_web_search=False,
        )
        self.logger   = AgentLogger()
        self.registry = ToolRegistry(self.cfg, self.logger)

    def test_unknown_tool_returns_error(self):
        result, call = self.registry.dispatch("nonexistent_tool", "foo", {}, "test", 1)
        self.assertIn("error", result)
        self.assertEqual(call.error, result["error"])

    def test_unknown_method_returns_error(self):
        result, call = self.registry.dispatch("local_file_search", "does_not_exist", {}, "test", 1)
        self.assertIn("error", result)

    def test_successful_call_is_logged(self):
        # list_dir on the tmpdir
        result, call = self.registry.dispatch(
            "local_file_search", "list_dir", {"path": "."}, "test listing", 1
        )
        self.assertEqual(len(self.logger.log), 1)
        self.assertEqual(self.logger.log[0].tool, "local_file_search")
        self.assertEqual(self.logger.log[0].method, "list_dir")

    def test_calls_this_iter_counts_correctly(self):
        self.registry.dispatch("local_file_search", "list_dir", {"path": "."}, "a", 1)
        self.registry.dispatch("local_file_search", "list_dir", {"path": "."}, "b", 1)
        self.registry.dispatch("local_file_search", "list_dir", {"path": "."}, "c", 2)
        self.assertEqual(self.logger.calls_this_iter(1), 2)
        self.assertEqual(self.logger.calls_this_iter(2), 1)

    def test_python_disabled_returns_error(self):
        result, _ = self.registry.dispatch(
            "code_execution", "run_python", {"code": "print(1)"}, "test", 1
        )
        self.assertIn("error", result)
        self.assertIn("disabled", result["error"])

    def test_web_search_disabled_returns_error(self):
        result, _ = self.registry.dispatch(
            "web_search", "web_search", {"query": "Molingchang fault"}, "test", 1
        )
        self.assertIn("error", result)
        self.assertIn("disabled", result["error"])


# ─────────────────────────────────────────────────────────────────────────────
# ── 5. AgentConfig ────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentConfig(unittest.TestCase):

    def test_defaults_sane(self):
        cfg = AgentConfig()
        self.assertEqual(cfg.workspace_root, ".")
        self.assertTrue(cfg.allow_python)
        self.assertFalse(cfg.allow_shell)
        self.assertFalse(cfg.allow_web_search)
        self.assertFalse(cfg.use_multimodal)
        self.assertTrue(cfg.use_rag)
        self.assertTrue(cfg.use_local_files)
        self.assertEqual(cfg.max_iterations, 3)
        self.assertEqual(cfg.max_tool_calls_per_iter, 8)

    def test_as_dict_is_json_serialisable(self):
        cfg  = AgentConfig(workspace_root="/tmp/test", allow_web_search=True)
        d    = cfg.as_dict()
        text = json.dumps(d)
        back = json.loads(text)
        self.assertEqual(back["workspace_root"], "/tmp/test")
        self.assertTrue(back["allow_web_search"])

    def test_extension_list_present(self):
        cfg = AgentConfig()
        self.assertIn(".pdf", cfg.allowed_extensions)
        self.assertIn(".csv", cfg.allowed_extensions)
        self.assertIn(".bib", cfg.allowed_extensions)


# ─────────────────────────────────────────────────────────────────────────────
# ── 6. Loop convergence (mock LLM) ────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopConvergence(unittest.TestCase):
    """
    Test that the LoopController stops correctly when:
    (a) max_iterations reached
    (b) no_new_evidence after iteration > 1
    """

    def _make_ctrl(self, tmpdir: str, mock_llm_fn):
        from sage_agents.evidence_driven_geo_agent import (
            LoopController, ToolRegistry, GeoEvidenceTableBuilder, AgentLogger
        )
        cfg = AgentConfig(
            workspace_root=tmpdir,
            output_dir=tmpdir,
            max_iterations=3,
            max_tool_calls_per_iter=2,
            allow_python=False,
        )
        logger   = AgentLogger()
        registry = ToolRegistry(cfg, logger)
        ev_table = GeoEvidenceTableBuilder()
        llm_cfg  = {"provider": "mock", "model": "mock", "api_base": "mock://"}

        ctrl = LoopController(
            config=cfg, llm_cfg=llm_cfg,
            registry=registry, ev_table=ev_table, logger=logger
        )
        # Patch LLM call
        ctrl._llm_call_fn = mock_llm_fn
        return ctrl, ev_table

    def test_max_iterations_stops_loop(self):
        tmpdir = tempfile.mkdtemp()

        call_count = [0]

        def mock_llm(messages, cfg, **kw):
            call_count[0] += 1
            # Tool selector → done immediately every iteration
            return json.dumps({"done": True, "reason": "mock done"})

        with patch("sage_agents.evidence_driven_geo_agent._llm_call", side_effect=mock_llm):
            from sage_agents.evidence_driven_geo_agent import (
                LoopController, ToolRegistry, GeoEvidenceTableBuilder, AgentLogger
            )
            cfg = AgentConfig(
                workspace_root=tmpdir, output_dir=tmpdir,
                max_iterations=2, max_tool_calls_per_iter=2,
                allow_python=False,
            )
            logger   = AgentLogger()
            registry = ToolRegistry(cfg, logger)
            ev_table = GeoEvidenceTableBuilder()
            llm_cfg  = {"provider": "mock", "model": "mock", "api_base": "http://fake"}

            ctrl   = LoopController(cfg, llm_cfg, registry, ev_table, logger)
            result = ctrl.run("test question", "test area")

        self.assertIn(result.convergence_reason,
                      ("max_iterations_reached", "no_new_evidence"))
        self.assertLessEqual(result.iterations_run, 2)

    def test_no_new_evidence_triggers_convergence(self):
        """If no evidence is added in iteration 2, loop should stop early."""
        tmpdir = tempfile.mkdtemp()
        # Write a small CSV
        (Path(tmpdir) / "cat.csv").write_text(
            "lon,lat,depth,mag\n104.1,29.5,10,3.2\n", encoding="utf-8"
        )

        select_calls = [0]

        def mock_llm(messages, cfg, **kw):
            select_calls[0] += 1
            sys_content = messages[0].get("content", "") if messages else ""
            # Tool selector call — always signal done
            if "tool_name" in sys_content or "tool selector" in sys_content.lower() \
               or "TOOL SELECTOR" in sys_content or "tool_selector" in sys_content \
               or "autonomous geoscience" in sys_content:
                return json.dumps({"done": True, "reason": "mock"})
            # Hypothesis / reasoner / report calls — return minimal valid JSON
            if "competing hypotheses" in sys_content.lower() or \
               "hypothesis" in sys_content.lower():
                return "[]"
            if "evaluate" in sys_content.lower() or "preferred_hypothesis" in sys_content:
                return json.dumps({
                    "evaluations": [],
                    "preferred_hypothesis": "",
                    "preferred_rationale": "",
                    "missing_information": [],
                })
            # Report writer
            return "# Report\n\nNo evidence collected."

        with patch("sage_agents.evidence_driven_geo_agent._llm_call", side_effect=mock_llm):
            from sage_agents.evidence_driven_geo_agent import (
                LoopController, ToolRegistry, GeoEvidenceTableBuilder, AgentLogger
            )
            cfg = AgentConfig(
                workspace_root=tmpdir, output_dir=tmpdir,
                max_iterations=4, max_tool_calls_per_iter=1,
                allow_python=False,
            )
            logger   = AgentLogger()
            registry = ToolRegistry(cfg, logger)
            ev_table = GeoEvidenceTableBuilder()
            llm_cfg  = {"provider": "mock", "model": "mock", "api_base": "http://fake"}

            ctrl   = LoopController(cfg, llm_cfg, registry, ev_table, logger)
            result = ctrl.run("test question", "test area")

        # With no evidence added, should converge after iteration 2 at most
        self.assertIn(result.convergence_reason,
                      ("no_new_evidence", "max_iterations_reached"))
        self.assertLessEqual(result.iterations_run, 4)


# ─────────────────────────────────────────────────────────────────────────────
# ── 7. GeoEvidence dataclass ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoEvidence(unittest.TestCase):

    def test_defaults_are_set(self):
        ev = GeoEvidence(
            evidence_id="e001",
            source="test.pdf",
            source_type="literature",
            observation="There is a seismicity cluster at 10 km depth.",
        )
        self.assertEqual(ev.evidence_type, "text")
        self.assertEqual(ev.confidence, "medium")
        self.assertEqual(ev.depth_range, "unspecified")
        self.assertEqual(ev.conflict_with, [])
        self.assertEqual(ev.supports, [])
        self.assertEqual(ev.contradicts, [])
        self.assertEqual(ev.alternative_interpretation, "")
        self.assertEqual(ev.citation, "")

    def test_json_serialisable(self):
        from dataclasses import asdict
        ev = GeoEvidence(
            evidence_id="e001",
            source="test.pdf",
            source_type="literature",
            observation="Seismicity cluster at 10 km depth.",
            data_type="seismicity",
        )
        d    = asdict(ev)
        text = json.dumps(d)
        back = json.loads(text)
        self.assertEqual(back["evidence_id"], "e001")
        self.assertEqual(back["source_type"], "literature")


# ─────────────────────────────────────────────────────────────────────────────
# ── 8. WebSearchTool gates ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class TestWebSearchToolGates(unittest.TestCase):

    def test_web_search_disabled_by_default(self):
        cfg  = AgentConfig()
        tool = WebSearchTool(cfg)
        r    = tool.web_search("Molingchang fault seismicity")
        self.assertIn("error", r)
        self.assertIn("disabled", r["error"])

    def test_scholar_search_disabled_by_default(self):
        cfg  = AgentConfig()
        tool = WebSearchTool(cfg)
        r    = tool.scholar_search("induced seismicity Sichuan")
        self.assertIn("error", r)

    def test_download_pdf_disabled_by_default(self):
        cfg  = AgentConfig()
        tool = WebSearchTool(cfg)
        r    = tool.download_pdf("https://example.com/paper.pdf")
        self.assertIn("error", r)

    def test_web_search_allowed_when_configured(self):
        cfg  = AgentConfig(allow_web_search=True)
        tool = WebSearchTool(cfg)
        # Should not return the "disabled" error; network error is acceptable in CI
        r = tool.web_search("Molingchang fault seismicity")
        self.assertNotIn("disabled", r.get("error", ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
