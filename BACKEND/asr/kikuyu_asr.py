"""
Kikuyu ASR using badrex/w2v-bert-2.0-kikuyu-asr (HuggingFace)
Returns (transcript, avg_log_prob_confidence)
"""
import logging
import numpy as np
from transformers import pipeline

logger = logging.getLogger(__name__)

_kikuyu_pipeline = None


def _get_pipeline():
    global _kikuyu_pipeline
    if _kikuyu_pipeline is None:
        logger.info("Loading Kikuyu ASR model (first run may take a while)...")
        _kikuyu_pipeline = pipeline(
            "automatic-speech-recognition",
            model="badrex/w2v-bert-2.0-kikuyu-asr",
            return_timestamps=False,
        )
        logger.info("Kikuyu ASR model loaded.")
    return _kikuyu_pipeline


def transcribe_kikuyu(wav_path: str) -> tuple[str, float]:
    """
    Transcribe a WAV file using the Kikuyu ASR model.

    Args:
        wav_path: Absolute path to a 16kHz mono WAV file.

    Returns:
        (transcript, confidence_score) where confidence is in [0, 1].
        Higher is more confident the audio is Kikuyu.
    """
    try:
        asr = _get_pipeline()
        result = asr(wav_path)
        transcript = result.get("text", "").strip()

        # Extract token log-probabilities if available for confidence scoring
        confidence = _estimate_confidence(result)
        logger.info(f"[Kikuyu ASR] transcript='{transcript}' confidence={confidence:.3f}")
        return transcript, confidence

    except Exception as e:
        logger.error(f"Kikuyu ASR error: {e}")
        return "", 0.0


def _estimate_confidence(result: dict) -> float:
    """
    Estimate confidence from the ASR result.
    Uses chunk-level scores if available, else falls back to transcript length heuristic.
    """
    chunks = result.get("chunks", [])
    if chunks:
        scores = [c.get("score", 0.0) for c in chunks if "score" in c]
        if scores:
            return float(np.mean(scores))

    # Fallback: non-empty transcript gets a moderate baseline
    transcript = result.get("text", "")
    return 0.5 if transcript.strip() else 0.0
