"""Tests for phase-picking code output guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from seismo_code.code_engine import CodeEngine
from seismo_code.safe_executor import ExecutionResult


class TestPhasePickingGuard(unittest.TestCase):
    def test_rejects_explicit_sage_test_fail_despite_zero_exit(self):
        engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
        result = ExecutionResult(
            success=True,
            stdout="[SAGE_TEST] FAIL: china_topography.png was not generated.",
            stderr="",
            output_files=[],
            figures=[],
        )

        self.assertFalse(engine._execution_success(result))
        ok, reason = engine._mini_test_ok("使用 GMT 绘制中国地形图并输出 PNG", "", result)
        self.assertFalse(ok)
        self.assertIn("failure", reason.lower())

    def test_rejects_first_sta_lta_trigger_as_pick_for_explicit_classical_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_picks.csv"
            out.write_text("station,phase,time\nSTA,P,0.5\n", encoding="utf-8")
            engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
            code = """
from obspy.signal.trigger import trigger_onset
triggers = trigger_onset(cft, 3.5, 2.0)
pick_idx = triggers[0][0]
print("[SAGE_TEST] Phase picking completed")
"""
            ok, reason = engine._mini_test_ok(
                "使用 STA/LTA 拾取震相",
                code,
                ExecutionResult(
                    success=True,
                    stdout="[SAGE_TEST] Phase picking completed",
                    output_files=[str(out)],
                    exec_dir=tmp,
                ),
            )

        self.assertFalse(ok)
        self.assertIn("first STA/LTA trigger", reason)

    def test_rejects_sta_lta_as_default_phase_picker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_picks.csv"
            out.write_text("station,phase,time\nSTA,P,0.5\n", encoding="utf-8")
            engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
            code = """
from obspy.signal.trigger import classic_sta_lta, trigger_onset
sta_len = 0.5
lta_len = 5.0
cft = classic_sta_lta(tr.data, int(sta_len * fs), int(lta_len * fs))
triggers = trigger_onset(cft, 3.0, 1.5)
print("[SAGE_TEST] Phase picking completed")
"""
            ok, reason = engine._mini_test_ok(
                "帮我拾取震相",
                code,
                ExecutionResult(
                    success=True,
                    stdout="[SAGE_TEST] Phase picking completed",
                    output_files=[str(out)],
                    exec_dir=tmp,
                ),
            )

        self.assertFalse(ok)
        self.assertIn("fell back to STA/LTA", reason)

    def test_requires_pnsn_for_default_phase_picker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_picks.csv"
            out.write_text("station,phase,time\nSTA,P,0.5\n", encoding="utf-8")
            engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
            code = """
from obspy import read
st = read("waveforms/*.SAC")
print("[SAGE_TEST] Phase picking completed")
"""
            ok, reason = engine._mini_test_ok(
                "检测一下这个波形的震相",
                code,
                ExecutionResult(
                    success=True,
                    stdout="[SAGE_TEST] Phase picking completed",
                    output_files=[str(out)],
                    exec_dir=tmp,
                ),
            )

        self.assertFalse(ok)
        self.assertIn("did not use the PNSN picker", reason)

    def test_allows_visualizing_existing_phase_pick_result_without_re_picking(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig = Path(tmp) / "waveform_with_picks.png"
            Image.new("RGB", (4, 4), color="white").save(fig)
            engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
            code = """
def parse_pnsn_pick_text(path):
    return [{"phase": "Pg", "relative_time_s": 12.3}]
plot_stream(st, picks=parse_pnsn_pick_text("sage_picks.txt"))
print("[SAGE_TEST] plotted 1 picks from sage_picks.txt")
"""
            ok, reason = engine._mini_test_ok(
                "把这个拾取结果绘制到波形上",
                code,
                ExecutionResult(
                    success=True,
                    stdout="[SAGE_TEST] plotted 1 picks from sage_picks.txt",
                    figures=[str(fig)],
                    exec_dir=tmp,
                ),
            )

        self.assertTrue(ok, reason)

    def test_rejects_picks_found_but_zero_plotted(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig = Path(tmp) / "waveform_with_picks.png"
            Image.new("RGB", (4, 4), color="white").save(fig)
            engine = CodeEngine(llm_config={"provider": "test", "api_base": "http://test", "model": "test"})
            code = """
from seismo_skill.skills.pnsn_phase_detection.pnsn import PNSNPicker
picker = PNSNPicker()
picks = picker.pick_stream(st)
plot_picks = []
plot_stream(st, picks=plot_picks)
"""
            ok, reason = engine._mini_test_ok(
                "拾取震相，并把拾取结果绘制到波形上",
                code,
                ExecutionResult(
                    success=True,
                    stdout="[SAGE_TEST] PNSN 拾取到 3 个震相\n用于绘图的 picks 数量：0",
                    figures=[str(fig)],
                    exec_dir=tmp,
                ),
            )

        self.assertFalse(ok)
        self.assertIn("zero picks were passed to plotting", reason)


if __name__ == "__main__":
    unittest.main()
