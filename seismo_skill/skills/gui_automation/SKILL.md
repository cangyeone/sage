---
name: gui_automation
category: automation
keywords:
  - GUI
  - desktop
  - mouse
  - click
  - button
  - window
  - screenshot
  - keyboard
  - hotkey
  - pyautogui
  - xdotool
  - 鼠标
  - 点击
  - 按钮
  - 窗口
  - 界面
  - 截图
  - 键盘
  - 快捷键
summary: Control desktop GUI tasks with screenshot, mouse, keyboard, and scrolling helpers.
---

# GUI Automation Skill

Use this skill when the user explicitly asks SAGE to control a desktop GUI:
screenshot the screen, click coordinates, type text, press hotkeys, scroll, drag,
or inspect whether GUI automation is available.

Prefer browser automation or Playwright for normal web pages. Use this desktop
GUI skill when the task is outside the browser automation surface, when the user
specifically asks for mouse/keyboard control, or when only pixel-level UI control
is practical.

## Local API

Import the built-in helpers from `seismo_code.gui_automation`:

```python
from seismo_code.gui_automation import (
    backend_status,
    screenshot,
    click,
    drag,
    move_to,
    type_text,
    hotkey,
    scroll,
    GuiAutomationError,
)
```

Important functions:

- `backend_status() -> dict`: report platform and available backends.
- `screenshot(output_path="screenshot.png") -> str`: save a screenshot and return the absolute path.
- `click(x, y, button="left", clicks=1, interval=0.05) -> dict`: click screen coordinates.
- `move_to(x, y, duration=0.0) -> dict`: move the pointer.
- `drag(from_x, from_y, to_x, to_y, duration=0.2, button="left") -> dict`: drag between two coordinate points.
- `type_text(text, interval=0.0) -> dict`: type literal text into the focused element.
- `hotkey(*keys) -> dict`: press a shortcut, for example `hotkey("ctrl", "s")`.
- `scroll(clicks, x=None, y=None) -> dict`: scroll up for positive clicks, down for negative clicks.

Do not invent OCR or accessibility APIs. `click_text(...)` is intentionally not
implemented in the built-in backend yet; if text targeting is needed, first take
a screenshot, inspect or ask for coordinates, then call `click(x, y)`.

## Workflow

1. Print `backend_status()` before the first GUI action so the user can see which backend is active.
2. Take a screenshot before coordinate clicking unless the user already supplied reliable coordinates.
3. Keep operations small and reversible: click, inspect, then continue.
4. For destructive actions such as delete, overwrite, purchase, send, or submit, stop and ask the user to confirm the exact action and target.
5. Print `[SAGE_TEST]` lines for backend status, screenshot path, and each action result.

## Cross-platform Notes

- Windows/macOS/Linux: `pyautogui` is the preferred optional backend.
- macOS: Screen Recording and Accessibility permissions may be required. Without `pyautogui`, mouse-only fallback can use `cliclick` if installed.
- Linux X11: `xdotool` supports mouse and keyboard fallback; `gnome-screenshot` or ImageMagick `import` can take screenshots.
- Linux Wayland: global mouse/keyboard automation may be blocked by the compositor; tell the user when no backend is available.

## Example

```python
# lang:python
from seismo_code.gui_automation import backend_status, screenshot, click, type_text, hotkey

print("[SAGE_TEST] GUI backend:", backend_status())
screen_path = screenshot("screen_before.png")
print("[SAGE_TEST] Screenshot:", screen_path)

# Example coordinate click. Replace coordinates only when the target is known.
result = click(200, 150)
print("[SAGE_TEST] Click:", result)

type_text("hello from SAGE")
hotkey("ctrl", "s")
print("[SAGE_TEST] GUI sequence completed")
```
