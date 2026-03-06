"""Attachment management for the March agent framework.

Handles saving, referencing, and retrieving attachments (images, PDFs, audio, etc.)
so that raw binary data is never stored in chat history.

Flow:
1. Attachment arrives (e.g., via WebSocket) → save to disk, generate reference
2. First LLM call → send full content (image bytes, extracted text, etc.)
3. Save to history → replace raw data with a compact reference + description
4. Future LLM calls → history contains only the reference text
5. If user asks to re-examine → reload from disk using the stored path

Attachment references in message content look like:
  [attachment: image "photo.jpg" (45KB) saved to /path/to/file — description of content]
"""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("march.attachments")

# Default directory for saved attachments
DEFAULT_ATTACHMENTS_DIR = Path.home() / ".march" / "attachments"

# Attachment types by category
IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/svg+xml"}
PDF_MIMES = {"application/pdf"}
AUDIO_MIMES = {"audio/webm", "audio/ogg", "audio/mp4", "audio/wav", "audio/mpeg"}


@dataclass
class AttachmentRef:
    """A reference to a saved attachment.

    This is what gets stored in chat history instead of the raw data.

    Attributes:
        path: Filesystem path where the attachment is saved.
        filename: Original filename from the user.
        media_type: MIME type of the attachment.
        size_bytes: Size of the saved file in bytes.
        description: Short LLM-friendly description of the content.
        category: One of: image, pdf, audio, text, binary.
        created_at: Unix timestamp when the attachment was saved.
        content_hash: SHA-256 hash of the content (for deduplication).
    """

    path: str
    filename: str
    media_type: str
    size_bytes: int
    description: str
    category: str = "binary"
    created_at: float = 0.0
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = time.time()

    def to_history_text(self) -> str:
        """Generate the compact text representation for chat history.

        This replaces the raw attachment data in stored messages.
        """
        size_str = _format_size(self.size_bytes)
        return (
            f"[attachment: {self.category} \"{self.filename}\" ({size_str}) "
            f"saved to {self.path} — {self.description}]"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "path": self.path,
            "filename": self.filename,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "description": self.description,
            "category": self.category,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttachmentRef":
        """Deserialize from dict."""
        return cls(
            path=data["path"],
            filename=data["filename"],
            media_type=data["media_type"],
            size_bytes=data["size_bytes"],
            description=data.get("description", ""),
            category=data.get("category", "binary"),
            created_at=data.get("created_at", 0.0),
            content_hash=data.get("content_hash", ""),
        )

    def exists(self) -> bool:
        """Check if the attachment file still exists on disk."""
        return Path(self.path).exists()


class AttachmentStore:
    """Manages saving and retrieving attachments on disk.

    Attachments are organized by date:
      ~/.march/attachments/2025-01-20/abc123_photo.jpg
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_ATTACHMENTS_DIR

    def save(
        self,
        data: bytes,
        filename: str,
        media_type: str,
        description: str = "",
    ) -> AttachmentRef:
        """Save attachment bytes to disk and return a reference.

        Args:
            data: Raw file bytes.
            filename: Original filename.
            media_type: MIME type.
            description: Short description of the content.

        Returns:
            AttachmentRef pointing to the saved file.
        """
        # Compute hash for deduplication and unique naming
        content_hash = hashlib.sha256(data).hexdigest()[:16]

        # Organize by date
        date_str = time.strftime("%Y-%m-%d")
        day_dir = self.base_dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        # Build filename: hash_originalname
        safe_filename = _safe_filename(filename)
        save_name = f"{content_hash}_{safe_filename}"
        save_path = day_dir / save_name

        # Don't re-save if identical content already exists
        if save_path.exists():
            logger.debug("Attachment already exists: %s", save_path)
        else:
            save_path.write_bytes(data)
            logger.info(
                "Saved attachment: %s (%s, %d bytes)",
                save_path, media_type, len(data),
            )

        # Determine category
        category = _categorize(media_type, filename)

        return AttachmentRef(
            path=str(save_path),
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
            description=description or f"{category} file: {filename}",
            category=category,
            content_hash=content_hash,
        )

    def save_from_base64(
        self,
        b64_data: str,
        filename: str,
        media_type: str,
        description: str = "",
    ) -> AttachmentRef:
        """Save a base64-encoded attachment."""
        raw_bytes = base64.b64decode(b64_data)
        return self.save(raw_bytes, filename, media_type, description)

    def load_bytes(self, ref: AttachmentRef) -> bytes | None:
        """Load the raw bytes of an attachment from its reference.

        Returns None if the file no longer exists.
        """
        path = Path(ref.path)
        if not path.exists():
            logger.warning("Attachment file missing: %s", ref.path)
            return None
        return path.read_bytes()

    def load_as_base64(self, ref: AttachmentRef) -> str | None:
        """Load an attachment as a base64 string.

        Returns None if the file no longer exists.
        """
        data = self.load_bytes(ref)
        if data is None:
            return None
        return base64.b64encode(data).decode("ascii")


def strip_attachments_from_content(content: str | list) -> str | list:
    """Replace inline attachment data in message content with reference text.

    For multimodal content (list of blocks), replaces image blocks with
    their text description. For string content, returns as-is.

    This is used when saving messages to history — the raw data is already
    saved to disk, so we only need the reference in history.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return content

    stripped: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            stripped.append(block)
            continue

        if block.get("type") == "image":
            # Replace image block with text reference
            # The attachment_ref metadata should have been set by the channel
            ref_text = block.get("_attachment_text")
            if ref_text:
                stripped.append({"type": "text", "text": ref_text})
            else:
                stripped.append({
                    "type": "text",
                    "text": "[image attachment — data not stored in history]",
                })
        else:
            stripped.append(block)

    return stripped


def strip_attachments_from_messages(
    messages: list[dict[str, Any]],
    skip_last: bool = True,
) -> list[dict[str, Any]]:
    """Strip attachment data from a list of LLM messages.

    Replaces inline image/binary data with text references in all messages
    except optionally the last one (which is the current user request and
    needs the full data for the LLM to process).

    Args:
        messages: List of LLM-format messages.
        skip_last: If True, don't strip the last message (current request).

    Returns:
        New list with attachment data stripped from history messages.
    """
    result: list[dict[str, Any]] = []
    last_idx = len(messages) - 1

    for i, msg in enumerate(messages):
        if skip_last and i == last_idx:
            result.append(msg)
            continue

        content = msg.get("content")
        if isinstance(content, list):
            stripped_content = strip_attachments_from_content(content)
            result.append({**msg, "content": stripped_content})
        else:
            result.append(msg)

    return result


def content_to_history_text(content: str | list) -> str:
    """Convert multimodal content to a plain text string for session history.

    Used when saving user messages to session — extracts text parts and
    replaces attachments with their reference descriptions.
    """
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
            elif block.get("type") == "image":
                ref_text = block.get("_attachment_text")
                if ref_text:
                    parts.append(ref_text)
                else:
                    parts.append("[image attachment]")
            else:
                text = block.get("text", str(block))
                if text:
                    parts.append(text)
        elif isinstance(block, str):
            parts.append(block)

    return "\n".join(parts)


# ── Helpers ──────────────────────────────────────────────────────────────


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f}MB"


def _safe_filename(filename: str) -> str:
    """Sanitize a filename for safe filesystem storage."""
    # Keep only alphanumeric, dots, hyphens, underscores
    safe = "".join(
        c if c.isalnum() or c in ".-_" else "_"
        for c in filename
    )
    # Limit length
    if len(safe) > 100:
        ext = Path(filename).suffix
        safe = safe[:96] + ext
    return safe or "attachment"


def _categorize(media_type: str, filename: str) -> str:
    """Determine attachment category from MIME type and filename."""
    if media_type in IMAGE_MIMES or media_type.startswith("image/"):
        return "image"
    if media_type in PDF_MIMES or filename.lower().endswith(".pdf"):
        return "pdf"
    if media_type in AUDIO_MIMES or media_type.startswith("audio/"):
        return "audio"
    if media_type.startswith("text/") or media_type in (
        "application/json", "application/xml", "application/yaml",
    ):
        return "text"
    return "binary"
