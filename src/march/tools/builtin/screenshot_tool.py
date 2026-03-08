"""Screen capture using mss (cross-platform)."""

from __future__ import annotations

import base64
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.screenshot_tool")


@tool(name="screenshot", description="Capture a screenshot of the screen.")
async def screenshot_tool(
    monitor: int = 0,
    output_path: str = "",
) -> str:
    """Capture a screenshot of the screen.

    Args:
        monitor: Monitor index (0 for all monitors, 1+ for specific).
        output_path: Save to this path. If empty, returns base64 data.
    """
    try:
        import mss
        import mss.tools
    except ImportError:
        return "Error: mss not installed. Run: pip install mss"

    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor >= len(monitors):
                return f"Error: Monitor {monitor} not found. Available: 0-{len(monitors)-1}"

            shot = sct.grab(monitors[monitor])
            png_data = mss.tools.to_png(shot.rgb, shot.size)

            if output_path:
                p = Path(output_path).expanduser().resolve()
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(png_data)
                return f"Screenshot saved to: {p} ({len(png_data)} bytes, {shot.width}x{shot.height})"
            else:
                b64 = base64.b64encode(png_data).decode()
                return (
                    f"[Screenshot: {shot.width}x{shot.height}, {len(png_data)} bytes]\n"
                    f"data:image/png;base64,{b64}"
                )
    except Exception as e:
        return f"Error capturing screenshot: {e}"
