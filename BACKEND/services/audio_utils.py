"""
Audio Utilities — download voice notes from Twilio and convert OGG/OGA to WAV.
WhatsApp voice notes are sent as OGG Opus; ASR models need 16kHz mono WAV.

Uses torchaudio (already installed for ASR) instead of pydub to avoid
the audioop removal issue on Python 3.13+.
"""
import os
import logging
import requests
import subprocess
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def download_audio(url: str, output_path: str) -> str:
    """
    Download audio from Twilio media URL (requires HTTP auth).

    Args:
        url: Twilio MediaUrl (requires Basic auth with SID/token).
        output_path: Where to save the downloaded file.

    Returns:
        Path to the downloaded file.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")

    response = requests.get(url, auth=(account_sid, auth_token), timeout=30)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info(f"Downloaded audio: {output_path} ({len(response.content)} bytes)")
    return output_path


def ogg_to_wav(ogg_path: str, wav_path: str, target_sr: int = 16000) -> str:
    """
    Convert OGG/OGA file to 16kHz mono WAV (required by ASR models).

    Uses torchaudio if available, otherwise falls back to ffmpeg subprocess.

    Args:
        ogg_path: Input OGG file path.
        wav_path: Output WAV file path.
        target_sr: Sample rate (default 16000 Hz for ASR).

    Returns:
        Path to the converted WAV file.
    """
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(ogg_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        torchaudio.save(wav_path, waveform, target_sr)
        duration = waveform.shape[1] / target_sr
        logger.info(f"Converted OGG→WAV (torchaudio): {wav_path} ({duration:.1f}s)")
        return wav_path

    except Exception as e:
        logger.warning(f"torchaudio failed ({e}), trying ffmpeg...")

    # Fallback: use ffmpeg via subprocess
    cmd = [
        "ffmpeg", "-y", "-i", ogg_path,
        "-ar", str(target_sr),
        "-ac", "1",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")

    logger.info(f"Converted OGG→WAV (ffmpeg): {wav_path}")
    return wav_path
