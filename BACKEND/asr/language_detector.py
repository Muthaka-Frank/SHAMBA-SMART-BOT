"""
Cascading ASR Language Detector
Runs both Kikuyu and Swahili ASR models and picks the one with higher confidence.
Returns (best_transcript, detected_language) where language is 'ki' or 'sw'.
"""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from asr.kikuyu_asr import transcribe_kikuyu
from asr.swahili_asr import transcribe_swahili

logger = logging.getLogger(__name__)


def detect_language(wav_path: str) -> tuple[str, str]:
    """
    Run both ASR models in parallel, pick the one with higher confidence.

    Args:
        wav_path: Path to a 16kHz mono WAV file.

    Returns:
        (transcript, language_code) — language_code is 'ki' (Kikuyu) or 'sw' (Swahili).
    """
    results = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(transcribe_kikuyu, wav_path): "ki",
            executor.submit(transcribe_swahili, wav_path): "sw",
        }
        for future in as_completed(futures):
            lang = futures[future]
            try:
                transcript, confidence = future.result()
                results[lang] = (transcript, confidence)
                logger.info(f"[{lang.upper()}] confidence={confidence:.3f} | '{transcript[:60]}'")
            except Exception as e:
                logger.error(f"ASR error for {lang}: {e}")
                results[lang] = ("", 0.0)

    ki_transcript, ki_conf = results.get("ki", ("", 0.0))
    sw_transcript, sw_conf = results.get("sw", ("", 0.0))

    if ki_conf >= sw_conf and ki_transcript:
        logger.info(f"Selected KIKUYU (conf={ki_conf:.3f} vs sw={sw_conf:.3f})")
        return ki_transcript, "ki"
    else:
        logger.info(f"Selected SWAHILI (conf={sw_conf:.3f} vs ki={ki_conf:.3f})")
        return sw_transcript or ki_transcript, "sw"
