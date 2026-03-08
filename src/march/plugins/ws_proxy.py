"""Backward-compatibility shim — WSProxyPlugin has moved to march.channels.ws_channel.

All imports are re-exported from the new location. Update your imports:
    Old: from march.plugins.ws_proxy import WSProxyPlugin
    New: from march.channels.ws_channel import WSChannel
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "march.plugins.ws_proxy is deprecated; "
    "import from march.channels.ws_channel instead",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the new location
from march.channels.ws_channel import (  # noqa: F401
    WSChannel as WSProxyPlugin,
    WSChannel,
    ChatDB,
    _StreamBuffer,
    _WSConn,
    _try_send,
    _resize_image,
    _extract_pdf_text,
    _summarize_with_llm,
    _process_attachment,
    _chunked_summarize,
    DEFAULTS,
)

__all__ = [
    "WSProxyPlugin",
    "WSChannel",
    "ChatDB",
    "_StreamBuffer",
    "_WSConn",
    "_try_send",
    "_resize_image",
    "_extract_pdf_text",
    "_summarize_with_llm",
    "_process_attachment",
    "_chunked_summarize",
    "DEFAULTS",
]
