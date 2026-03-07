"""Web search using ddgs (DuckDuckGo Search), no API key needed."""

from __future__ import annotations

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.web_search")


@tool(
    name="web_search",
    description="Search the web using DuckDuckGo. No API key needed. Returns titles, URLs, and snippets.",
)
async def web_search(
    query: str,
    count: int = 5,
    region: str = "wt-wt",
) -> str:
    """Search the web.

    Args:
        query: Search query string.
        count: Number of results to return (1-20).
        region: Region code (e.g. 'us-en', 'wt-wt' for worldwide).
    """
    if not query.strip():
        return "Error: Empty search query"

    count = max(1, min(20, count))

    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs not installed. Run: pip install ddgs"

    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=count, region=region)
    except Exception as e:
        return f"Error searching: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        snippet = r.get("body", "")
        lines.append(f"{i}. {title}")
        lines.append(f"   {url}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)
