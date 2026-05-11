"""Tests for phase-picking code output guards."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from seismo_code.code_engine import CodeEngine
from seismo_code.safe_executor import ExecutionResult


class TestPhasePickingGuard(unittest.TestCase):
    def test_rejects_first_sta_lta_trigger_as_pick(self):
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
                "拾取震相",
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


if __name__ == "__main__":
    unittest.main()
