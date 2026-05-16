"""Tests for built-in GUI automation helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from seismo_code import gui_automation as gui
from seismo_code.code_engine import CodeEngine


class FakeImage:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"fake png")


class FakePyAutoGUI:
    def __init__(self):
        self.calls = []

    def screenshot(self):
        self.calls.append(("screenshot",))
        return FakeImage()

    def click(self, **kwargs):
        self.calls.append(("click", kwargs))

    def moveTo(self, *args, **kwargs):
        self.calls.append(("moveTo", args, kwargs))

    def write(self, *args, **kwargs):
        self.calls.append(("write", args, kwargs))

    def hotkey(self, *args):
        self.calls.append(("hotkey", args))

    def dragTo(self, *args, **kwargs):
        self.calls.append(("dragTo", args, kwargs))

    def scroll(self, amount):
        self.calls.append(("scroll", amount))


class TestGuiAutomation(unittest.TestCase):
    def test_backend_status_is_structured(self):
        status = gui.backend_status()

        self.assertIn("platform", status)
        self.assertIn("pyautogui", status)
        self.assertIn("notes", status)

    def test_pyautogui_backend_actions_are_wrapped(self):
        fake = FakePyAutoGUI()
        with patch.object(gui, "_import_pyautogui", return_value=fake):
            click_result = gui.click(10.2, 20.7, clicks=2)
            move_result = gui.move_to(11, 22)
            type_result = gui.type_text("abc")
            hotkey_result = gui.hotkey("ctrl", "s")
            scroll_result = gui.scroll(-3)
            drag_result = gui.drag(1, 2, 30, 40)

        self.assertEqual(click_result["backend"], "pyautogui")
        self.assertEqual(click_result["x"], 10)
        self.assertEqual(click_result["y"], 21)
        self.assertEqual(move_result["action"], "move_to")
        self.assertEqual(type_result["chars"], 3)
        self.assertEqual(hotkey_result["keys"], ["ctrl", "s"])
        self.assertEqual(scroll_result["clicks"], -3)
        self.assertEqual(drag_result["to"], [30, 40])
        self.assertTrue(any(call[0] == "click" for call in fake.calls))
        self.assertTrue(any(call[0] == "hotkey" for call in fake.calls))
        self.assertTrue(any(call[0] == "dragTo" for call in fake.calls))

    def test_screenshot_saves_file_with_pyautogui_backend(self):
        fake = FakePyAutoGUI()
        with tempfile.TemporaryDirectory() as tmp, patch.object(gui, "_import_pyautogui", return_value=fake):
            out = gui.screenshot(str(Path(tmp) / "screen.png"))

            self.assertTrue(Path(out).is_file())
            self.assertIn(("screenshot",), fake.calls)

    def test_invalid_button_and_text_click_fail_clearly(self):
        with self.assertRaises(gui.GuiAutomationError):
            gui.click(1, 2, button="side")

        with self.assertRaises(gui.GuiAutomationError) as ctx:
            gui.click_text("OK")
        self.assertIn("click(x, y)", str(ctx.exception))

    def test_code_engine_local_api_documents_gui_helpers(self):
        engine = CodeEngine(
            llm_config={"provider": "test", "api_base": "http://test", "model": "test"},
            project_root=str(PROJECT_ROOT),
        )

        context = engine._build_local_api_context("帮我截图并点击 GUI 里的确定按钮")

        self.assertIn("seismo_code.gui_automation", context)
        self.assertIn("backend_status", context)
        self.assertIn("click(x, y)", context)
        self.assertIn("GUI automation notes", context)


if __name__ == "__main__":
    unittest.main()
