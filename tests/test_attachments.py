"""Tests for the attachment management system."""

import base64
import os
import tempfile
from pathlib import Path

import pytest

from march.core.attachments import (
    AttachmentRef,
    AttachmentStore,
    content_to_history_text,
    strip_attachments_from_content,
    strip_attachments_from_messages,
    _categorize,
    _format_size,
    _safe_filename,
)


class TestAttachmentRef:
    """Tests for AttachmentRef dataclass."""

    def test_to_history_text(self):
        ref = AttachmentRef(
            path="/tmp/abc123_photo.jpg",
            filename="photo.jpg",
            media_type="image/jpeg",
            size_bytes=45000,
            description="Photo of a cat",
            category="image",
        )
        text = ref.to_history_text()
        assert "image" in text
        assert "photo.jpg" in text
        assert "43KB" in text  # 45000 // 1024 = 43
        assert "/tmp/abc123_photo.jpg" in text
        assert "Photo of a cat" in text

    def test_to_dict_and_from_dict(self):
        ref = AttachmentRef(
            path="/tmp/test.pdf",
            filename="test.pdf",
            media_type="application/pdf",
            size_bytes=100000,
            description="A test PDF",
            category="pdf",
            content_hash="abc123",
        )
        d = ref.to_dict()
        restored = AttachmentRef.from_dict(d)
        assert restored.path == ref.path
        assert restored.filename == ref.filename
        assert restored.media_type == ref.media_type
        assert restored.size_bytes == ref.size_bytes
        assert restored.description == ref.description
        assert restored.category == ref.category
        assert restored.content_hash == ref.content_hash

    def test_exists(self):
        # Non-existent path
        ref = AttachmentRef(
            path="/nonexistent/file.jpg",
            filename="file.jpg",
            media_type="image/jpeg",
            size_bytes=100,
            description="test",
        )
        assert not ref.exists()

        # Existing path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test")
            tmp_path = f.name

        try:
            ref2 = AttachmentRef(
                path=tmp_path,
                filename="test.txt",
                media_type="text/plain",
                size_bytes=4,
                description="test",
            )
            assert ref2.exists()
        finally:
            os.unlink(tmp_path)


class TestAttachmentStore:
    """Tests for AttachmentStore."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="march_test_attach_")
        self.store = AttachmentStore(base_dir=self.tmp_dir)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load(self):
        data = b"Hello, this is test data"
        ref = self.store.save(
            data=data,
            filename="test.txt",
            media_type="text/plain",
            description="A test file",
        )

        assert ref.filename == "test.txt"
        assert ref.media_type == "text/plain"
        assert ref.size_bytes == len(data)
        assert ref.category == "text"
        assert ref.description == "A test file"
        assert ref.exists()

        # Load back
        loaded = self.store.load_bytes(ref)
        assert loaded == data

    def test_save_image(self):
        # Fake image data
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG-like header
        ref = self.store.save(
            data=data,
            filename="photo.jpg",
            media_type="image/jpeg",
            description="A photo",
        )

        assert ref.category == "image"
        assert ref.exists()
        assert "photo.jpg" in ref.path

    def test_save_from_base64(self):
        original = b"base64 test data"
        b64 = base64.b64encode(original).decode()
        ref = self.store.save_from_base64(
            b64_data=b64,
            filename="encoded.bin",
            media_type="application/octet-stream",
        )

        loaded = self.store.load_bytes(ref)
        assert loaded == original

    def test_deduplication(self):
        data = b"duplicate content"
        ref1 = self.store.save(data, "file1.txt", "text/plain")
        ref2 = self.store.save(data, "file1.txt", "text/plain")

        # Same content hash → same path
        assert ref1.path == ref2.path
        assert ref1.content_hash == ref2.content_hash

    def test_load_missing_file(self):
        ref = AttachmentRef(
            path="/nonexistent/file.txt",
            filename="file.txt",
            media_type="text/plain",
            size_bytes=100,
            description="missing",
        )
        assert self.store.load_bytes(ref) is None
        assert self.store.load_as_base64(ref) is None

    def test_load_as_base64(self):
        data = b"test for base64 loading"
        ref = self.store.save(data, "test.bin", "application/octet-stream")
        b64 = self.store.load_as_base64(ref)
        assert b64 is not None
        assert base64.b64decode(b64) == data

    def test_date_organized_storage(self):
        import time
        data = b"organized"
        ref = self.store.save(data, "org.txt", "text/plain")
        date_str = time.strftime("%Y-%m-%d")
        assert date_str in ref.path


class TestStripAttachments:
    """Tests for attachment stripping functions."""

    def test_strip_string_content_unchanged(self):
        result = strip_attachments_from_content("Hello world")
        assert result == "Hello world"

    def test_strip_image_blocks(self):
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "data": "AAAA" * 1000},
                "_attachment_text": '[attachment: image "photo.jpg" (10KB) saved to /tmp/photo.jpg — A photo]',
            },
            {
                "type": "text",
                "text": "Describe this image",
            },
        ]
        result = strip_attachments_from_content(content)
        assert isinstance(result, list)
        assert len(result) == 2
        # Image block should be replaced with text
        assert result[0]["type"] == "text"
        assert "photo.jpg" in result[0]["text"]
        assert "base64" not in str(result[0])
        # Text block unchanged
        assert result[1]["text"] == "Describe this image"

    def test_strip_image_without_ref(self):
        content = [
            {
                "type": "image",
                "source": {"type": "base64", "data": "AAAA"},
            },
        ]
        result = strip_attachments_from_content(content)
        assert result[0]["type"] == "text"
        assert "not stored" in result[0]["text"]

    def test_strip_messages_skip_last(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"data": "OLD"}, "_attachment_text": "[old image ref]"},
                {"type": "text", "text": "old message"},
            ]},
            {"role": "assistant", "content": "I see the image"},
            {"role": "user", "content": [
                {"type": "image", "source": {"data": "NEW_BASE64_DATA"}},
                {"type": "text", "text": "new message"},
            ]},
        ]
        result = strip_attachments_from_messages(messages, skip_last=True)

        # First message: image stripped
        assert result[0]["content"][0]["type"] == "text"
        assert "old image ref" in result[0]["content"][0]["text"]

        # Last message: image preserved (current request)
        assert result[2]["content"][0]["type"] == "image"
        assert result[2]["content"][0]["source"]["data"] == "NEW_BASE64_DATA"

    def test_strip_messages_all(self):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"data": "DATA"}, "_attachment_text": "[ref]"},
            ]},
        ]
        result = strip_attachments_from_messages(messages, skip_last=False)
        assert result[0]["content"][0]["type"] == "text"

    def test_strip_preserves_text_messages(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = strip_attachments_from_messages(messages)
        assert result == messages


class TestContentToHistoryText:
    """Tests for content_to_history_text."""

    def test_string_passthrough(self):
        assert content_to_history_text("hello") == "hello"

    def test_multimodal_with_ref(self):
        content = [
            {"type": "image", "_attachment_text": "[image: cat.jpg saved to /tmp/cat.jpg]"},
            {"type": "text", "text": "What is this?"},
        ]
        result = content_to_history_text(content)
        assert "[image: cat.jpg saved to /tmp/cat.jpg]" in result
        assert "What is this?" in result

    def test_multimodal_without_ref(self):
        content = [
            {"type": "image", "source": {"data": "base64stuff"}},
            {"type": "text", "text": "Describe this"},
        ]
        result = content_to_history_text(content)
        assert "[image attachment]" in result
        assert "Describe this" in result
        # No base64 data in the result
        assert "base64stuff" not in result

    def test_text_only_blocks(self):
        content = [
            {"type": "text", "text": "First part"},
            {"type": "text", "text": "Second part"},
        ]
        result = content_to_history_text(content)
        assert "First part" in result
        assert "Second part" in result


class TestHelpers:
    """Tests for helper functions."""

    def test_format_size(self):
        assert _format_size(500) == "500B"
        assert _format_size(1024) == "1KB"
        assert _format_size(45000) == "43KB"
        assert _format_size(1500000) == "1.4MB"

    def test_safe_filename(self):
        assert _safe_filename("photo.jpg") == "photo.jpg"
        assert _safe_filename("my file (1).png") == "my_file__1_.png"
        assert _safe_filename("") == "attachment"
        # Long filename gets truncated
        long_name = "a" * 200 + ".txt"
        result = _safe_filename(long_name)
        assert len(result) <= 100

    def test_categorize(self):
        assert _categorize("image/jpeg", "photo.jpg") == "image"
        assert _categorize("image/png", "screenshot.png") == "image"
        assert _categorize("application/pdf", "doc.pdf") == "pdf"
        assert _categorize("audio/webm", "voice.webm") == "audio"
        assert _categorize("text/plain", "notes.txt") == "text"
        assert _categorize("application/json", "data.json") == "text"
        assert _categorize("application/octet-stream", "file.bin") == "binary"
