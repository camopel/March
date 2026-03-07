"""Text-to-speech with configurable backend (system/elevenlabs/local)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.tts_tool")


@tool(name="tts", description="Convert text to speech audio. Supports system TTS, ElevenLabs, or local models.")
async def tts_tool(
    text: str,
    backend: str = "system",
    voice: str = "",
    output_path: str = "",
    api_key: str = "",
) -> str:
    """Convert text to speech.

    Args:
        text: Text to convert to speech.
        backend: TTS backend: system, elevenlabs, local.
        voice: Voice name or ID (backend-specific).
        output_path: Output file path. Auto-generated if empty.
        api_key: API key for ElevenLabs (or ELEVENLABS_API_KEY env var).
    """
    if not text.strip():
        return "Error: Empty text"

    out = output_path or tempfile.mktemp(suffix=".mp3")

    if backend == "system":
        return await _system_tts(text, voice, out)
    elif backend == "elevenlabs":
        key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not key:
            return "Error: ElevenLabs API key required (pass api_key or set ELEVENLABS_API_KEY)"
        return await _elevenlabs_tts(text, voice, out, key)
    elif backend == "local":
        return await _local_tts(text, voice, out)
    else:
        return f"Error: Unknown backend '{backend}'. Use: system, elevenlabs, local"


async def _system_tts(text: str, voice: str, output: str) -> str:
    """Use system TTS (espeak/say)."""
    import shutil

    if shutil.which("say"):
        # macOS
        cmd = ["say", "-o", output, "--data-format=LEF32@22050"]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
    elif shutil.which("espeak-ng"):
        cmd = ["espeak-ng", "-w", output]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
    elif shutil.which("espeak"):
        cmd = ["espeak", "-w", output]
        if voice:
            cmd.extend(["-v", voice])
        cmd.append(text)
    else:
        return "Error: No system TTS found (need say, espeak-ng, or espeak)"

    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        return f"Error: TTS failed: {stderr.decode()}"

    return f"Audio saved to: {output}"


async def _elevenlabs_tts(text: str, voice: str, output: str, api_key: str) -> str:
    """Use ElevenLabs API."""
    try:
        import httpx
    except ImportError:
        return "Error: httpx not installed"

    voice_id = voice or "21m00Tcm4TlvDq8ikWAM"  # Default: Rachel
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                },
            )
            resp.raise_for_status()
            Path(output).write_bytes(resp.content)
            return f"Audio saved to: {output} ({len(resp.content)} bytes)"
    except Exception as e:
        return f"Error: ElevenLabs API failed: {e}"


async def _local_tts(text: str, voice: str, output: str) -> str:
    """Placeholder for local TTS model."""
    return "Error: Local TTS not yet implemented. Use 'system' or 'elevenlabs' backend."
