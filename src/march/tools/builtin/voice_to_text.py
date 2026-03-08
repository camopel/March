"""Speech-to-text using faster-whisper with auto language detection."""

from __future__ import annotations

from pathlib import Path

from march.logging import get_logger
from march.tools.base import tool

logger = get_logger("march.tools.voice_to_text")

# Chinese script bias prompts
_ZH_SIMPLIFIED_PROMPT = "以下是普通话的句子，使用简体中文转录。"
_ZH_TRADITIONAL_PROMPT = "以下是中文語音，使用繁體中文轉錄。"

# Map config language codes to (whisper_lang, initial_prompt)
_ZH_SIMPLIFIED_CODES = {"zh-cn", "zh_cn", "zh-hans"}
_ZH_TRADITIONAL_CODES = {"zh-tw", "zh_tw", "zh-hant"}


@tool(name="voice_to_text", description="Transcribe audio to text using Whisper (local, no API key).")
async def voice_to_text(
    path: str,
    model_size: str = "base",
    language: str = "",
) -> str:
    """Transcribe audio file to text.

    Args:
        path: Path to the audio file.
        model_size: Whisper model size: tiny, base, small, medium, large-v3.
        language: Language code (e.g. 'en', 'zh-cn', 'de'). Empty for auto-detection.
                  'zh-cn' = auto-detect + Simplified Chinese bias when Chinese is detected.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return f"Error: Audio file not found: {path}"

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "Error: faster-whisper not installed. Run: pip install faster-whisper"

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        lang_lower = language.lower().strip() if language else ""
        is_zh_simplified = lang_lower in _ZH_SIMPLIFIED_CODES
        is_zh_traditional = lang_lower in _ZH_TRADITIONAL_CODES

        if is_zh_simplified or is_zh_traditional:
            # Smart mode: auto-detect first, apply Chinese bias only when zh detected
            segments, info = model.transcribe(str(p))
            detected_lang = info.language if hasattr(info, "language") else "unknown"

            if detected_lang == "zh":
                # Re-transcribe with the appropriate script bias
                prompt = _ZH_SIMPLIFIED_PROMPT if is_zh_simplified else _ZH_TRADITIONAL_PROMPT
                segments, info = model.transcribe(
                    str(p), language="zh", initial_prompt=prompt
                )

            texts = [seg.text.strip() for seg in segments]
        elif lang_lower:
            # Explicit non-Chinese language forced
            segments, info = model.transcribe(str(p), language=lang_lower)
            texts = [seg.text.strip() for seg in segments]
        else:
            # Pure auto-detect, no bias
            segments, info = model.transcribe(str(p))
            texts = [seg.text.strip() for seg in segments]

        transcript = " ".join(texts)
        detected_lang = info.language if hasattr(info, "language") else "unknown"

        return (
            f"Language: {detected_lang} | Duration: {info.duration:.1f}s\n"
            f"\n"
            f"{transcript.strip()}"
        )
    except Exception as e:
        return f"Error transcribing: {e}"
