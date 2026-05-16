"""
Cross-platform GUI automation helpers for SAGE generated code.

The module intentionally has no required third-party dependencies. It prefers
`pyautogui` when installed, then falls back to small platform command-line tools
when available. Generated programs can import these functions directly instead
of guessing OS-specific mouse and keyboard APIs.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class GuiAutomationError(RuntimeError):
    """Raised when the requested GUI action cannot be performed."""


def _import_pyautogui():
    try:
        import pyautogui  # type: ignore
    except Exception:
        return None
    return pyautogui


def _tool(name: str) -> Optional[str]:
    return shutil.which(name)


def _run(cmd: Iterable[str]) -> None:
    subprocess.run(list(cmd), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _point(x: int | float, y: int | float) -> tuple[int, int]:
    ix, iy = int(round(float(x))), int(round(float(y)))
    if ix < 0 or iy < 0:
        raise GuiAutomationError("GUI coordinates must be non-negative screen pixels.")
    return ix, iy


def _button(button: str) -> str:
    value = (button or "left").lower()
    if value not in {"left", "right", "middle"}:
        raise GuiAutomationError("button must be one of: left, right, middle")
    return value


def _xdotool_button(button: str) -> str:
    return {"left": "1", "middle": "2", "right": "3"}[_button(button)]


def _backend_for_action(action: str) -> str:
    pyautogui = _import_pyautogui()
    if pyautogui is not None:
        return "pyautogui"
    system = platform.system().lower()
    if action == "screenshot":
        if system == "darwin" and _tool("screencapture"):
            return "screencapture"
        if _tool("gnome-screenshot"):
            return "gnome-screenshot"
        if _tool("import"):
            return "imagemagick-import"
    if _tool("xdotool") and os.environ.get("DISPLAY"):
        return "xdotool"
    if system == "darwin" and _tool("cliclick") and action in {"click", "move_to"}:
        return "cliclick"
    raise GuiAutomationError(
        "No GUI automation backend is available. Install `pyautogui`, or on Linux "
        "install `xdotool`/`gnome-screenshot`; on macOS grant screen recording and "
        "accessibility permissions, or install `cliclick` for mouse-only fallback."
    )


def backend_status() -> Dict[str, Any]:
    """Return available GUI automation backends and useful platform diagnostics."""
    pyautogui = _import_pyautogui()
    return {
        "platform": platform.system(),
        "release": platform.release(),
        "pyautogui": pyautogui is not None,
        "xdotool": bool(_tool("xdotool")),
        "cliclick": bool(_tool("cliclick")),
        "screencapture": bool(_tool("screencapture")),
        "gnome_screenshot": bool(_tool("gnome-screenshot")),
        "imagemagick_import": bool(_tool("import")),
        "display": os.environ.get("DISPLAY", ""),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY", ""),
        "notes": [
            "Browser UI should usually use Playwright/browser automation instead of pixel clicks.",
            "macOS may require Accessibility and Screen Recording permissions.",
            "Linux Wayland sessions may block global mouse/keyboard automation; X11 works better.",
        ],
    }


def screenshot(output_path: str = "screenshot.png") -> str:
    """Save a screenshot and return its absolute path."""
    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    backend = _backend_for_action("screenshot")
    if backend == "pyautogui":
        image = _import_pyautogui().screenshot()
        image.save(str(out))
    elif backend == "screencapture":
        _run(["screencapture", "-x", str(out)])
    elif backend == "gnome-screenshot":
        _run(["gnome-screenshot", "-f", str(out)])
    elif backend == "imagemagick-import":
        _run(["import", "-window", "root", str(out)])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support screenshots.")
    return str(out)


def move_to(x: int | float, y: int | float, duration: float = 0.0) -> Dict[str, Any]:
    """Move the pointer to screen coordinates `(x, y)`."""
    ix, iy = _point(x, y)
    backend = _backend_for_action("move_to")
    if backend == "pyautogui":
        _import_pyautogui().moveTo(ix, iy, duration=max(0.0, float(duration)))
    elif backend == "xdotool":
        _run(["xdotool", "mousemove", str(ix), str(iy)])
    elif backend == "cliclick":
        _run(["cliclick", f"m:{ix},{iy}"])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support pointer movement.")
    return {"ok": True, "action": "move_to", "backend": backend, "x": ix, "y": iy}


def click(
    x: int | float,
    y: int | float,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.05,
) -> Dict[str, Any]:
    """Click at screen coordinates `(x, y)`."""
    ix, iy = _point(x, y)
    btn = _button(button)
    count = max(1, int(clicks))
    pause = max(0.0, float(interval))
    backend = _backend_for_action("click")
    if backend == "pyautogui":
        _import_pyautogui().click(x=ix, y=iy, clicks=count, interval=pause, button=btn)
    elif backend == "xdotool":
        _run(["xdotool", "mousemove", str(ix), str(iy)])
        for _ in range(count):
            _run(["xdotool", "click", _xdotool_button(btn)])
            if pause:
                time.sleep(pause)
    elif backend == "cliclick":
        for _ in range(count):
            _run(["cliclick", f"c:{ix},{iy}"])
            if pause:
                time.sleep(pause)
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support clicking.")
    return {"ok": True, "action": "click", "backend": backend, "x": ix, "y": iy, "button": btn, "clicks": count}


def drag(
    from_x: int | float,
    from_y: int | float,
    to_x: int | float,
    to_y: int | float,
    duration: float = 0.2,
    button: str = "left",
) -> Dict[str, Any]:
    """Drag from one screen coordinate to another."""
    x1, y1 = _point(from_x, from_y)
    x2, y2 = _point(to_x, to_y)
    btn = _button(button)
    backend = _backend_for_action("drag")
    if backend == "pyautogui":
        pg = _import_pyautogui()
        pg.moveTo(x1, y1)
        pg.dragTo(x2, y2, duration=max(0.0, float(duration)), button=btn)
    elif backend == "xdotool":
        _run(["xdotool", "mousemove", str(x1), str(y1)])
        _run(["xdotool", "mousedown", _xdotool_button(btn)])
        _run(["xdotool", "mousemove", str(x2), str(y2)])
        _run(["xdotool", "mouseup", _xdotool_button(btn)])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support dragging.")
    return {
        "ok": True,
        "action": "drag",
        "backend": backend,
        "from": [x1, y1],
        "to": [x2, y2],
        "button": btn,
    }


def type_text(text: str, interval: float = 0.0) -> Dict[str, Any]:
    """Type literal text into the currently focused UI element."""
    backend = _backend_for_action("type_text")
    if backend == "pyautogui":
        _import_pyautogui().write(str(text), interval=max(0.0, float(interval)))
    elif backend == "xdotool":
        _run(["xdotool", "type", "--delay", str(int(max(0.0, float(interval)) * 1000)), str(text)])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support typing text.")
    return {"ok": True, "action": "type_text", "backend": backend, "chars": len(str(text))}


def hotkey(*keys: str) -> Dict[str, Any]:
    """Press a keyboard shortcut, for example `hotkey('ctrl', 's')`."""
    clean = [str(k).strip() for k in keys if str(k).strip()]
    if not clean:
        raise GuiAutomationError("hotkey requires at least one key.")
    backend = _backend_for_action("hotkey")
    if backend == "pyautogui":
        _import_pyautogui().hotkey(*clean)
    elif backend == "xdotool":
        _run(["xdotool", "key", "+".join(clean)])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support hotkeys.")
    return {"ok": True, "action": "hotkey", "backend": backend, "keys": clean}


def scroll(clicks: int, x: Optional[int | float] = None, y: Optional[int | float] = None) -> Dict[str, Any]:
    """Scroll at the current pointer position, or move to `(x, y)` first."""
    amount = int(clicks)
    backend = _backend_for_action("scroll")
    if backend == "pyautogui":
        if x is not None and y is not None:
            ix, iy = _point(x, y)
            _import_pyautogui().moveTo(ix, iy)
        _import_pyautogui().scroll(amount)
    elif backend == "xdotool":
        if x is not None and y is not None:
            ix, iy = _point(x, y)
            _run(["xdotool", "mousemove", str(ix), str(iy)])
        button = "4" if amount > 0 else "5"
        for _ in range(abs(amount)):
            _run(["xdotool", "click", button])
    else:
        raise GuiAutomationError(f"Backend {backend!r} does not support scrolling.")
    return {"ok": True, "action": "scroll", "backend": backend, "clicks": amount}


def click_text(text: str, screenshot_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Placeholder for OCR/accessibility based clicking.

    This function is explicit so generated code does not pretend text targeting
    exists when no OCR or accessibility backend is configured.
    """
    raise GuiAutomationError(
        "click_text is not available in the built-in backend yet. Take a screenshot, "
        "inspect the target coordinates, then call click(x, y)."
    )


__all__ = [
    "GuiAutomationError",
    "backend_status",
    "screenshot",
    "move_to",
    "click",
    "drag",
    "type_text",
    "hotkey",
    "scroll",
    "click_text",
]
