"""Fetch URL content and convert to markdown/text with SSRF protection."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.web_fetch")

_MAX_CHARS = 100_000
_TIMEOUT = 30

# Private/reserved IP ranges for SSRF protection
_BLOCKED_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_ssrf_safe(url: str) -> tuple[bool, str]:
    """Check if a URL is safe from SSRF attacks."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"Unsupported scheme: {parsed.scheme}"

    hostname = parsed.hostname
    if not hostname:
        return False, "No hostname in URL"

    try:
        addrs = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addrs:
            ip = ipaddress.ip_address(sockaddr[0])
            for network in _BLOCKED_RANGES:
                if ip in network:
                    return False, f"Blocked: {hostname} resolves to private IP {ip}"
    except socket.gaierror:
        return False, f"Cannot resolve hostname: {hostname}"

    return True, ""


@tool(name="web_fetch", description="Fetch and extract readable content from a URL as markdown or text.")
async def web_fetch(
    url: str,
    extract_mode: str = "markdown",
    max_chars: int = 0,
) -> str:
    """Fetch URL content and extract readable text.

    Args:
        url: HTTP or HTTPS URL to fetch.
        extract_mode: Extraction mode: 'markdown' or 'text'.
        max_chars: Maximum characters to return (0 for default 100K).
    """
    if not url.strip():
        return "Error: Empty URL"

    # SSRF protection
    safe, reason = _is_ssrf_safe(url)
    if not safe:
        return f"Error: {reason}"

    limit = max_chars if max_chars > 0 else _MAX_CHARS

    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed"

    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MarchAgent/1.0)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} fetching {url}"
    except Exception as e:
        return f"Error fetching URL: {e}"

    content_type = resp.headers.get("content-type", "")
    html = resp.text

    if "text/html" not in content_type and "application/xhtml" not in content_type:
        # Not HTML — return raw text
        text = html[:limit]
        if len(html) > limit:
            text += "\n[truncated]"
        return text

    # Extract readable content
    try:
        from readability import Document
        doc = Document(html)
        title = doc.title()
        clean_html = doc.summary()
    except ImportError:
        # Fallback: strip tags manually
        import re
        title = ""
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()
        clean_html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r"<style[^>]*>.*?</style>", "", clean_html, flags=re.DOTALL | re.IGNORECASE)
        clean_html = re.sub(r"<[^>]+>", " ", clean_html)
        clean_html = re.sub(r"\s+", " ", clean_html).strip()
    except Exception as e:
        return f"Error extracting content: {e}"

    if extract_mode == "markdown":
        try:
            from markdownify import markdownify as md
            text = md(clean_html, heading_style="ATX", strip=["img", "script", "style"])
        except ImportError:
            # Fallback
            import re
            text = re.sub(r"<[^>]+>", "", clean_html)
            text = re.sub(r"\s+", " ", text).strip()
    else:
        import re
        text = re.sub(r"<[^>]+>", " ", clean_html)
        text = re.sub(r"\s+", " ", text).strip()

    # Add title
    result = f"# {title}\n\n{text}" if title else text

    if len(result) > limit:
        result = result[:limit] + "\n[truncated]"

    return result
