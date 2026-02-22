"""
Swahili ASR using thinkKenya/wav2vec2-large-xls-r-300m-sw (HuggingFace)
Returns (transcript, avg_log_prob_confidence)
"""
import logging
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

_swahili_pipeline = None


def _get_pipeline():
    global _swahili_pipeline
    if _swahili_pipeline is None:
        logger.info("Loading Swahili ASR model (first run may take a while)...")
        _swahili_pipeline = pipeline(
            "automatic-speech-recognition",
            model="thinkKenya/wav2vec2-large-xls-r-300m-sw",
            return_timestamps=False,
        )
        logger.info("Swahili ASR model loaded.")
    return _swahili_pipeline


def transcribe_swahili(wav_path: str) -> tuple[str, float]:
    """
    Transcribe a WAV file using the Swahili ASR model.

    Args:
        wav_path: Absolute path to a 16kHz mono WAV file.

    Returns:
        (transcript, confidence_score) where confidence is in [0, 1].
        Higher is more confident the audio is Swahili.
    """
    try:
        asr = _get_pipeline()
        result = asr(wav_path)
        transcript = result.get("text", "").strip()
        confidence = _estimate_confidence(result)
        logger.info(f"[Swahili ASR] transcript='{transcript}' confidence={confidence:.3f}")
        return transcript, confidence

    except Exception as e:
        logger.error(f"Swahili ASR error: {e}")
        return "", 0.0


def _estimate_confidence(result: dict) -> float:
    chunks = result.get("chunks", [])
    if chunks:
        scores = [c.get("score", 0.0) for c in chunks if "score" in c]
        if scores:
            return float(np.mean(scores))
    transcript = result.get("text", "")
    return 0.5 if transcript.strip() else 0.0
