"""Comprehensive tests for all built-in tools."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from march.core.message import ToolCall, ToolResult
from march.tools.registry import ToolRegistry
from march.tools.builtin import register_all_builtin_tools


# ─── Registration Tests ───────────────────────────────────────────────────────


class TestRegistration:
    """Test that all tools register correctly."""

    def test_register_all_tools(self):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        assert registry.tool_count == 29

    def test_all_tools_have_metadata(self):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        for name in registry.names():
            tool = registry.get(name)
            assert tool is not None
            assert tool.name == name
            assert tool.description
            assert tool.parameters

    def test_tool_definitions_format(self):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        defs = registry.definitions()
        assert len(defs) == 29
        for d in defs:
            assert d["type"] == "function"
            assert "function" in d
            assert "name" in d["function"]
            assert "description" in d["function"]
            assert "parameters" in d["function"]

    def test_anthropic_definitions_format(self):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)
        defs = registry.definitions_anthropic()
        assert len(defs) == 29
        for d in defs:
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ─── File Read Tests ──────────────────────────────────────────────────────────


class TestFileRead:
    @pytest.fixture
    def tmp_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        return str(f)

    @pytest.fixture
    def tmp_image_file(self, tmp_path):
        f = tmp_path / "test.png"
        # Minimal PNG
        f.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
        return str(f)

    async def test_read_text_file(self, tmp_text_file):
        from march.tools.builtin.file_read import file_read
        result = await file_read(path=tmp_text_file)
        assert "line1" in result
        assert "line5" in result

    async def test_read_with_offset_and_limit(self, tmp_text_file):
        from march.tools.builtin.file_read import file_read
        result = await file_read(path=tmp_text_file, offset=2, limit=2)
        assert "line2" in result
        assert "line3" in result
        assert "line1" not in result

    async def test_read_nonexistent_file(self):
        from march.tools.builtin.file_read import file_read
        result = await file_read(path="/nonexistent/file.txt")
        assert "Error" in result

    async def test_read_image_file(self, tmp_image_file):
        from march.tools.builtin.file_read import file_read
        result = await file_read(path=tmp_image_file)
        assert "Image" in result
        assert "base64" in result


# ─── File Write Tests ─────────────────────────────────────────────────────────


class TestFileWrite:
    async def test_write_new_file(self, tmp_path):
        from march.tools.builtin.file_write import file_write
        target = str(tmp_path / "new.txt")
        result = await file_write(path=target, content="hello world")
        assert "Wrote" in result
        assert Path(target).read_text() == "hello world"

    async def test_write_creates_parents(self, tmp_path):
        from march.tools.builtin.file_write import file_write
        target = str(tmp_path / "a" / "b" / "c.txt")
        result = await file_write(path=target, content="nested")
        assert "Wrote" in result
        assert Path(target).read_text() == "nested"

    async def test_write_overwrite(self, tmp_path):
        from march.tools.builtin.file_write import file_write
        target = tmp_path / "existing.txt"
        target.write_text("old")
        result = await file_write(path=str(target), content="new")
        assert "Wrote" in result
        assert target.read_text() == "new"


# ─── File Edit Tests ──────────────────────────────────────────────────────────


class TestFileEdit:
    async def test_edit_simple_replace(self, tmp_path):
        from march.tools.builtin.file_edit import file_edit
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    return 'hello'\n")
        result = await file_edit(path=str(f), old_string="'hello'", new_string="'world'")
        assert "Edited" in result
        assert "'world'" in f.read_text()

    async def test_edit_not_found(self, tmp_path):
        from march.tools.builtin.file_edit import file_edit
        f = tmp_path / "test.txt"
        f.write_text("original text")
        result = await file_edit(path=str(f), old_string="nonexistent", new_string="new")
        assert "Error" in result
        assert "not found" in result

    async def test_edit_multiple_matches(self, tmp_path):
        from march.tools.builtin.file_edit import file_edit
        f = tmp_path / "test.txt"
        f.write_text("foo bar foo")
        result = await file_edit(path=str(f), old_string="foo", new_string="baz")
        assert "Error" in result
        assert "2 times" in result

    async def test_edit_nonexistent_file(self):
        from march.tools.builtin.file_edit import file_edit
        result = await file_edit(path="/nonexistent", old_string="a", new_string="b")
        assert "Error" in result


# ─── Apply Patch Tests ────────────────────────────────────────────────────────


class TestApplyPatch:
    async def test_apply_simple_patch(self, tmp_path):
        from march.tools.builtin.apply_patch import apply_patch
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\nline3\n")
        patch = (
            "--- a/test.txt\n"
            "+++ b/test.txt\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+modified\n"
            " line3\n"
        )
        result = await apply_patch(path=str(f), patch=patch)
        assert "Applied 1 hunk" in result
        assert "modified" in f.read_text()

    async def test_apply_patch_nonexistent(self):
        from march.tools.builtin.apply_patch import apply_patch
        result = await apply_patch(path="/nonexistent", patch="@@ -1 +1 @@\n-a\n+b\n")
        assert "Error" in result

    async def test_apply_empty_patch(self, tmp_path):
        from march.tools.builtin.apply_patch import apply_patch
        f = tmp_path / "test.txt"
        f.write_text("content\n")
        result = await apply_patch(path=str(f), patch="no hunks here")
        assert "Error" in result
        assert "No hunks" in result


# ─── Glob Tests ───────────────────────────────────────────────────────────────


class TestGlob:
    async def test_glob_py_files(self, tmp_path):
        from march.tools.builtin.glob_tool import glob_tool
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = await glob_tool(pattern="*.py", path=str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    async def test_glob_no_matches(self, tmp_path):
        from march.tools.builtin.glob_tool import glob_tool
        result = await glob_tool(pattern="*.xyz", path=str(tmp_path))
        assert "No files" in result

    async def test_glob_recursive(self, tmp_path):
        from march.tools.builtin.glob_tool import glob_tool
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("")
        result = await glob_tool(pattern="**/*.py", path=str(tmp_path))
        assert "deep.py" in result


# ─── Diff Tests ───────────────────────────────────────────────────────────────


class TestDiff:
    async def test_diff_files(self, tmp_path):
        from march.tools.builtin.diff_tool import diff_tool
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\nworld\n")
        f2.write_text("hello\nearth\n")
        result = await diff_tool(file_a=str(f1), file_b=str(f2))
        assert "-world" in result
        assert "+earth" in result

    async def test_diff_identical(self, tmp_path):
        from march.tools.builtin.diff_tool import diff_tool
        f = tmp_path / "same.txt"
        f.write_text("same content\n")
        result = await diff_tool(file_a=str(f), file_b=str(f))
        assert "No differences" in result

    async def test_diff_text_strings(self):
        from march.tools.builtin.diff_tool import diff_tool
        result = await diff_tool(text_a="hello\n", text_b="world\n")
        assert "-hello" in result
        assert "+world" in result


# ─── Exec Tests ───────────────────────────────────────────────────────────────


class TestExec:
    async def test_exec_simple(self):
        from march.tools.builtin.exec_tool import exec_tool
        result = await exec_tool(command="echo hello")
        assert "hello" in result

    async def test_exec_exit_code(self):
        from march.tools.builtin.exec_tool import exec_tool
        result = await exec_tool(command="exit 42")
        assert "Exit code: 42" in result

    async def test_exec_timeout(self):
        from march.tools.builtin.exec_tool import exec_tool
        result = await exec_tool(command="sleep 10", timeout=1)
        assert "Timed out" in result

    async def test_exec_background(self):
        from march.tools.builtin.exec_tool import exec_tool
        result = await exec_tool(command="sleep 1", background=True)
        assert "Background session started" in result
        assert "PID" in result

    async def test_exec_workdir(self, tmp_path):
        from march.tools.builtin.exec_tool import exec_tool
        result = await exec_tool(command="pwd", workdir=str(tmp_path))
        assert str(tmp_path) in result


# ─── Process Tests ────────────────────────────────────────────────────────────


class TestProcess:
    async def test_process_list_empty(self):
        from march.tools.builtin.exec_tool import get_sessions
        from march.tools.builtin.process_tool import process_tool
        # Clear sessions
        sessions = get_sessions()
        sessions.clear()
        result = await process_tool(action="list")
        assert "No background sessions" in result

    async def test_process_list_with_session(self):
        from march.tools.builtin.exec_tool import exec_tool, get_sessions
        from march.tools.builtin.process_tool import process_tool

        sessions = get_sessions()
        sessions.clear()

        # Start a background process
        await exec_tool(command="sleep 5", background=True)
        result = await process_tool(action="list")
        assert "running" in result
        assert "sleep 5" in result

        # Clean up
        for sid in list(sessions.keys()):
            await process_tool(action="kill", session_id=sid)
        sessions.clear()

    async def test_process_unknown_action(self):
        from march.tools.builtin.process_tool import process_tool
        result = await process_tool(action="invalid")
        assert "Error" in result


# ─── Web Search Tests ─────────────────────────────────────────────────────────


class TestWebSearch:
    async def test_search_empty_query(self):
        from march.tools.builtin.web_search import web_search
        result = await web_search(query="")
        assert "Error" in result

    async def test_search_basic(self):
        from march.tools.builtin.web_search import web_search
        # Use a mock to avoid hitting real API in tests
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Test Result", "href": "https://example.com", "body": "A test snippet"}
        ]
        MockDDGS = MagicMock(return_value=mock_ddgs)
        with patch.dict("sys.modules", {}):
            with patch("duckduckgo_search.DDGS", MockDDGS, create=True):
                # Re-import to pick up the mock
                import importlib
                import march.tools.builtin.web_search as ws_mod
                # The import happens inside the function, so we patch at the source
                with patch("duckduckgo_search.DDGS", MockDDGS):
                    result = await web_search(query="test query")
                    assert "Test Result" in result
                    assert "example.com" in result


# ─── Web Fetch Tests ──────────────────────────────────────────────────────────


class TestWebFetch:
    async def test_fetch_empty_url(self):
        from march.tools.builtin.web_fetch import web_fetch
        result = await web_fetch(url="")
        assert "Error" in result

    async def test_fetch_ssrf_blocked(self):
        from march.tools.builtin.web_fetch import web_fetch
        result = await web_fetch(url="http://127.0.0.1/secret")
        assert "Error" in result
        assert "Blocked" in result or "private" in result.lower()

    async def test_fetch_invalid_scheme(self):
        from march.tools.builtin.web_fetch import web_fetch
        result = await web_fetch(url="ftp://example.com")
        assert "Error" in result


# ─── Browser Tests ────────────────────────────────────────────────────────────


class TestBrowser:
    async def test_browser_close_when_not_open(self):
        from march.tools.builtin.browser_tool import browser_tool
        result = await browser_tool(action="close")
        assert "closed" in result.lower()

    async def test_browser_unknown_action(self):
        from march.tools.builtin.browser_tool import browser_tool
        # This might try to launch a browser in headless mode
        # So we mock _ensure_browser
        import march.tools.builtin.browser_tool as bt
        old_page = bt._page
        bt._page = MagicMock()
        try:
            result = await browser_tool(action="invalid_action")
            assert "Error" in result or "Unknown" in result
        finally:
            bt._page = old_page

    async def test_browser_navigate_no_url(self):
        from march.tools.builtin.browser_tool import browser_tool
        import march.tools.builtin.browser_tool as bt
        mock_page = AsyncMock()
        old_page = bt._page
        bt._page = mock_page
        try:
            result = await browser_tool(action="navigate")
            assert "Error" in result
        finally:
            bt._page = old_page


# ─── Image Tests ──────────────────────────────────────────────────────────────


class TestImage:
    async def test_image_no_input(self):
        from march.tools.builtin.image_tool import image_tool
        result = await image_tool()
        assert "Error" in result

    async def test_image_url(self):
        from march.tools.builtin.image_tool import image_tool
        result = await image_tool(image="https://example.com/image.png")
        assert "Loaded 1 image" in result
        assert "url" in result

    async def test_image_file_not_found(self):
        from march.tools.builtin.image_tool import image_tool
        result = await image_tool(image="/nonexistent/img.png")
        assert "Error" in result

    async def test_image_base64(self):
        from march.tools.builtin.image_tool import image_tool
        import base64
        b64 = base64.b64encode(b"fake").decode()
        result = await image_tool(image=f"data:image/png;base64,{b64}")
        assert "Loaded 1 image" in result


# ─── PDF Tests ────────────────────────────────────────────────────────────────


class TestPdf:
    async def test_pdf_no_input(self):
        from march.tools.builtin.pdf_tool import pdf_tool
        result = await pdf_tool()
        assert "Error" in result

    async def test_pdf_nonexistent(self):
        from march.tools.builtin.pdf_tool import pdf_tool
        result = await pdf_tool(pdf="/nonexistent/doc.pdf")
        assert "Error" in result or "not found" in result.lower()


# ─── Voice to Text Tests ─────────────────────────────────────────────────────


class TestVoiceToText:
    async def test_voice_file_not_found(self):
        from march.tools.builtin.voice_to_text import voice_to_text
        result = await voice_to_text(path="/nonexistent/audio.wav")
        assert "Error" in result

    async def test_voice_basic_stub(self, tmp_path):
        from march.tools.builtin.voice_to_text import voice_to_text
        f = tmp_path / "audio.wav"
        f.write_bytes(b"RIFF" + b"\x00" * 100)
        # This will try to load the model, which may fail without a real audio file
        # But it should handle the error gracefully
        result = await voice_to_text(path=str(f))
        # Either succeeds or returns an error string (not crash)
        assert isinstance(result, str)


# ─── TTS Tests ────────────────────────────────────────────────────────────────


class TestTTS:
    async def test_tts_empty_text(self):
        from march.tools.builtin.tts_tool import tts_tool
        result = await tts_tool(text="")
        assert "Error" in result

    async def test_tts_unknown_backend(self):
        from march.tools.builtin.tts_tool import tts_tool
        result = await tts_tool(text="hello", backend="nonexistent")
        assert "Error" in result

    async def test_tts_elevenlabs_no_key(self):
        from march.tools.builtin.tts_tool import tts_tool
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ELEVENLABS_API_KEY", None)
            result = await tts_tool(text="hello", backend="elevenlabs")
            assert "Error" in result
            assert "key" in result.lower()


# ─── Screenshot Tests ─────────────────────────────────────────────────────────


class TestScreenshot:
    async def test_screenshot_basic(self):
        from march.tools.builtin.screenshot_tool import screenshot_tool
        # This may fail in headless CI environments
        result = await screenshot_tool()
        # Either succeeds (data:image/png;base64,...) or returns error
        assert isinstance(result, str)

    async def test_screenshot_invalid_monitor(self):
        from march.tools.builtin.screenshot_tool import screenshot_tool
        result = await screenshot_tool(monitor=999)
        # Should handle gracefully
        assert isinstance(result, str)


# ─── Clipboard Tests ──────────────────────────────────────────────────────────


class TestClipboard:
    async def test_clipboard_read(self):
        from march.tools.builtin.clipboard_tool import clipboard_tool
        # May not have clipboard access in CI
        result = await clipboard_tool()
        assert isinstance(result, str)

    async def test_clipboard_returns_string(self):
        from march.tools.builtin.clipboard_tool import clipboard_tool
        result = await clipboard_tool()
        assert isinstance(result, str)
        assert "Error" in result or "Clipboard" in result or "empty" in result.lower()


# ─── Translate Tests ──────────────────────────────────────────────────────────


class TestTranslate:
    async def test_translate_basic(self):
        from march.tools.builtin.translate_tool import translate_tool
        result = await translate_tool(text="Hello", target_language="German")
        assert "Translation request" in result
        assert "German" in result

    async def test_translate_empty_text(self):
        from march.tools.builtin.translate_tool import translate_tool
        result = await translate_tool(text="", target_language="French")
        assert "Error" in result

    async def test_translate_no_target(self):
        from march.tools.builtin.translate_tool import translate_tool
        result = await translate_tool(text="Hello", target_language="")
        assert "Error" in result


# ─── GitHub Search Tests ──────────────────────────────────────────────────────


class TestGitHubSearch:
    async def test_search_empty_query(self):
        from march.tools.builtin.github_search import github_search
        result = await github_search(query="")
        assert "Error" in result

    async def test_search_invalid_type(self):
        from march.tools.builtin.github_search import github_search
        result = await github_search(query="test", search_type="invalid")
        assert "Error" in result

    async def test_search_mock(self):
        from march.tools.builtin.github_search import github_search
        with patch("march.tools.builtin.github_search._github_request") as mock_req:
            mock_req.return_value = {
                "total_count": 1,
                "items": [{
                    "full_name": "user/repo",
                    "html_url": "https://github.com/user/repo",
                    "description": "A test repo",
                    "stargazers_count": 42,
                }],
            }
            result = await github_search(query="test")
            assert "user/repo" in result
            assert "42" in result


# ─── GitHub Ops Tests ─────────────────────────────────────────────────────────


class TestGitHubOps:
    async def test_ops_no_repo(self):
        from march.tools.builtin.github_ops import github_ops
        result = await github_ops(action="repo_info")
        assert "Error" in result

    async def test_ops_unknown_action(self):
        from march.tools.builtin.github_ops import github_ops
        result = await github_ops(action="invalid", repo="user/repo")
        assert "Error" in result

    async def test_ops_repo_info_mock(self):
        from march.tools.builtin.github_ops import github_ops
        with patch("march.tools.builtin.github_ops._github_api") as mock_api:
            mock_api.return_value = {
                "full_name": "user/repo",
                "description": "A test repo",
                "stargazers_count": 100,
                "forks_count": 20,
                "language": "Python",
                "default_branch": "main",
                "html_url": "https://github.com/user/repo",
            }
            result = await github_ops(action="repo_info", repo="user/repo")
            assert "user/repo" in result
            assert "100" in result


# ─── HuggingFace Tests ────────────────────────────────────────────────────────


class TestHuggingFace:
    async def test_hf_empty_query(self):
        from march.tools.builtin.huggingface_tool import huggingface_tool
        result = await huggingface_tool(query="")
        assert "Error" in result

    async def test_hf_invalid_type(self):
        from march.tools.builtin.huggingface_tool import huggingface_tool
        result = await huggingface_tool(query="test", search_type="invalid")
        assert "Error" in result

    async def test_hf_mock_search(self):
        from march.tools.builtin.huggingface_tool import huggingface_tool
        with patch("httpx.AsyncClient") as MockClient:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [
                {"modelId": "bert-base", "downloads": 1000, "likes": 50, "pipeline_tag": "fill-mask"}
            ]
            mock_resp.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            result = await huggingface_tool(query="bert")
            assert "bert-base" in result


# ─── Message Tests ────────────────────────────────────────────────────────────


class TestMessage:
    async def test_message_send(self):
        from march.tools.builtin.message_tool import message_tool
        result = await message_tool(action="send", target="user123", message="Hello!")
        assert "queued" in result.lower()

    async def test_message_no_target(self):
        from march.tools.builtin.message_tool import message_tool
        result = await message_tool(action="send", message="Hello!")
        assert "Error" in result

    async def test_message_no_text(self):
        from march.tools.builtin.message_tool import message_tool
        result = await message_tool(action="send", target="user123")
        assert "Error" in result


# ─── Cron Tests ───────────────────────────────────────────────────────────────


class TestCron:
    @pytest.fixture(autouse=True)
    def setup_cron_db(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "test_cron.db")
        monkeypatch.setattr("march.tools.builtin.cron_tool._DB_PATH", db_path)

    async def test_cron_create_and_list(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="create", name="test-job", schedule="*/5 * * * *", command="echo hi")
        assert "Created" in result

        result = await cron_tool(action="list")
        assert "test-job" in result

    async def test_cron_delete(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="create", name="to-delete", schedule="* * * * *", command="echo")
        job_id = result.split("\n")[0].split(": ")[1]
        result = await cron_tool(action="delete", job_id=job_id)
        assert "Deleted" in result

    async def test_cron_enable_disable(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="create", name="toggle", schedule="* * * * *", command="echo")
        job_id = result.split("\n")[0].split(": ")[1]

        result = await cron_tool(action="disable", job_id=job_id)
        assert "disabled" in result

        result = await cron_tool(action="enable", job_id=job_id)
        assert "enabled" in result

    async def test_cron_status(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="create", name="status-job", schedule="0 * * * *", command="echo")
        job_id = result.split("\n")[0].split(": ")[1]

        result = await cron_tool(action="status", job_id=job_id)
        assert "status-job" in result
        assert "0 * * * *" in result

    async def test_cron_unknown_action(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="invalid")
        assert "Error" in result

    async def test_cron_create_missing_fields(self):
        from march.tools.builtin.cron_tool import cron_tool
        result = await cron_tool(action="create", name="test")
        assert "Error" in result


# ─── Sessions Tests ───────────────────────────────────────────────────────────


class TestSessions:
    async def test_sessions_list(self):
        from march.tools.builtin.sessions_tools import sessions_list
        result = await sessions_list()
        assert "session" in result.lower()

    async def test_sessions_history_no_id(self):
        from march.tools.builtin.sessions_tools import sessions_history
        result = await sessions_history(session_id="")
        assert "Error" in result

    async def test_sessions_history(self):
        from march.tools.builtin.sessions_tools import sessions_history
        result = await sessions_history(session_id="test-123")
        assert "test-123" in result

    async def test_sessions_send(self):
        from march.tools.builtin.sessions_tools import sessions_send
        result = await sessions_send(session_id="test", message="hello")
        assert "queued" in result.lower() or "session" in result.lower()

    async def test_sessions_spawn(self):
        from march.tools.builtin.sessions_tools import sessions_spawn
        result = await sessions_spawn(task="Build a feature")
        assert "spawned" in result.lower()

    async def test_sessions_spawn_no_task(self):
        from march.tools.builtin.sessions_tools import sessions_spawn
        result = await sessions_spawn(task="")
        assert "Error" in result

    async def test_subagents_list(self):
        from march.tools.builtin.sessions_tools import subagents_tool
        result = await subagents_tool(action="list")
        assert isinstance(result, str)

    async def test_subagents_kill_no_target(self):
        from march.tools.builtin.sessions_tools import subagents_tool
        result = await subagents_tool(action="kill")
        assert "Error" in result

    async def test_session_status(self):
        from march.tools.builtin.sessions_tools import session_status
        result = await session_status()
        assert "Session Status" in result


# ─── Registry Execute Tests ───────────────────────────────────────────────────


class TestRegistryExecute:
    async def test_execute_tool_call(self, tmp_path):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)

        f = tmp_path / "test.txt"
        f.write_text("hello world\n")

        tc = ToolCall.create("read", {"path": str(f)})
        result = await registry.execute(tc)
        assert not result.is_error
        assert "hello world" in result.content

    async def test_execute_unknown_tool(self):
        from march.tools.registry import ToolNotFound
        registry = ToolRegistry()
        register_all_builtin_tools(registry)

        tc = ToolCall.create("nonexistent_tool", {})
        with pytest.raises(ToolNotFound):
            await registry.execute(tc)

    async def test_execute_batch(self, tmp_path):
        registry = ToolRegistry()
        register_all_builtin_tools(registry)

        f = tmp_path / "test.txt"
        f.write_text("content\n")

        calls = [
            ToolCall.create("read", {"path": str(f)}),
            ToolCall.create("glob", {"pattern": "*.txt", "path": str(tmp_path)}),
        ]
        results = await registry.execute_batch(calls)
        assert len(results) == 2
        assert "content" in results[0].content
        assert "test.txt" in results[1].content


# ─── MCP Client Tests ────────────────────────────────────────────────────────


class TestMCPClient:
    def test_mcp_tool_def(self):
        from march.tools.mcp.client import MCPToolDef
        td = MCPToolDef(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        d = td.to_dict()
        assert d["name"] == "test_tool"
        assert d["description"] == "A test tool"

    def test_mcp_client_not_connected(self):
        from march.tools.mcp.client import MCPClient
        client = MCPClient()
        assert not client.connected


# ─── MCP Discovery Tests ─────────────────────────────────────────────────────


class TestMCPDiscovery:
    def test_load_empty_config(self):
        from march.tools.mcp.discovery import _load_mcp_config
        config = _load_mcp_config("/nonexistent/path.json")
        assert config == {}

    async def test_discover_no_servers(self):
        from march.tools.mcp.discovery import discover_and_register
        registry = ToolRegistry()
        result = await discover_and_register(registry, config_path="/nonexistent/path.json")
        assert result == {}
