"""
TTS (Text-to-Speech) Service using gTTS.
Converts text response to an MP3 audio file.
Swahili → gTTS lang='sw'
Kikuyu  → gTTS lang='sw' (fallback; no Kikuyu TTS engine available yet)
"""
import os
import logging
import hashlib
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TMP_DIR = os.getenv("AUDIO_TMP_PATH", "./tmp_audio")


def text_to_speech(text: str, lang: str = "sw") -> str | None:
    """
    Convert text to speech and save as MP3.

    Args:
        text: The text to convert.
        lang: 'sw' (Swahili) or 'ki' (Kikuyu — fallback to Swahili TTS).

    Returns:
        Absolute path to the generated MP3 file, or None on failure.
    """
    os.makedirs(TMP_DIR, exist_ok=True)

    # Use Swahili TTS for both (Kikuyu TTS not yet available)
    gtts_lang = "sw" if lang in ("sw", "ki") else "en"

    # Cache by content hash to avoid regenerating identical audio
    content_hash = hashlib.md5(f"{text}{gtts_lang}".encode()).hexdigest()[:12]
    mp3_path = os.path.join(TMP_DIR, f"response_{content_hash}.mp3")

    if os.path.exists(mp3_path):
        logger.info(f"TTS cache hit: {mp3_path}")
        return mp3_path

    try:
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(mp3_path)
        logger.info(f"TTS generated: {mp3_path} (lang={gtts_lang}, {len(text)} chars)")
        return mp3_path
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None
