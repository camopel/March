"""Browser automation via Playwright: snapshot, screenshot, click, type, navigate."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.browser_tool")

# Global browser state
_browser = None
_context = None
_page = None


async def _ensure_browser():
    """Ensure a Playwright browser instance is running."""
    global _browser, _context, _page
    if _page is not None:
        return _page

    try:
        from playwright.async_api import async_playwright
    except ImportError:
        raise RuntimeError("playwright not installed. Run: pip install playwright && playwright install")

    pw = await async_playwright().__aenter__()
    _browser = await pw.chromium.launch(headless=True)
    _context = await _browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    )
    _page = await _context.new_page()
    return _page


@tool(name="browser", description="Browser automation: navigate, snapshot (accessibility tree), screenshot, click, type, evaluate JS.")
async def browser_tool(
    action: str,
    url: str = "",
    selector: str = "",
    text: str = "",
    key: str = "",
    js: str = "",
    full_page: bool = False,
) -> str:
    """Control a headless browser.

    Args:
        action: Action: navigate, snapshot, screenshot, click, type, press, evaluate, close.
        url: URL for navigate action.
        selector: CSS selector for click/type/press actions.
        text: Text to type (for type action).
        key: Key to press (for press action, e.g. 'Enter', 'Tab').
        js: JavaScript to evaluate (for evaluate action).
        full_page: Take full page screenshot (for screenshot action).
    """
    if action == "close":
        global _browser, _context, _page
        if _browser:
            await _browser.close()
            _browser = None
            _context = None
            _page = None
        return "Browser closed."

    try:
        page = await _ensure_browser()
    except Exception as e:
        return f"Error: {e}"

    try:
        if action == "navigate":
            if not url:
                return "Error: url required for navigate"
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            status = resp.status if resp else "unknown"
            title = await page.title()
            return f"Navigated to: {url}\nTitle: {title}\nStatus: {status}"

        elif action == "snapshot":
            # Return accessibility tree
            tree = await page.accessibility.snapshot()
            if tree is None:
                return "No accessibility tree available."
            return json.dumps(tree, indent=2, default=str)[:50000]

        elif action == "screenshot":
            data = await page.screenshot(full_page=full_page, type="png")
            b64 = base64.b64encode(data).decode()
            return f"[Screenshot: {len(data)} bytes]\ndata:image/png;base64,{b64}"

        elif action == "click":
            if not selector:
                return "Error: selector required for click"
            await page.click(selector, timeout=10000)
            return f"Clicked: {selector}"

        elif action == "type":
            if not selector:
                return "Error: selector required for type"
            await page.fill(selector, text, timeout=10000)
            return f"Typed into: {selector}"

        elif action == "press":
            k = key or "Enter"
            if selector:
                await page.press(selector, k, timeout=10000)
            else:
                await page.keyboard.press(k)
            return f"Pressed: {k}"

        elif action == "evaluate":
            if not js:
                return "Error: js required for evaluate"
            result = await page.evaluate(js)
            return json.dumps(result, indent=2, default=str) if result is not None else "(undefined)"

        else:
            return f"Error: Unknown action '{action}'. Use: navigate, snapshot, screenshot, click, type, press, evaluate, close"

    except Exception as e:
        return f"Browser error ({action}): {e}"
