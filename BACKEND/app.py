"""
Shamba-Smart — Flask Entry Point
Twilio WhatsApp Webhook Handler
"""
import os
import logging
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv

from asr.language_detector import detect_language
from services.intent_router import route_intent
from services.crop_advisor import get_crop_advice
from services.market_prices import get_market_prices
from services.weather import get_weather_forecast
from services.tts import text_to_speech
from services.audio_utils import download_audio, ogg_to_wav
from database.db import init_db, log_query, get_or_create_user

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
TMP_DIR = os.getenv("AUDIO_TMP_PATH", "./tmp_audio")
os.makedirs(TMP_DIR, exist_ok=True)


@app.route("/webhook", methods=["POST"])
def webhook():
    """Main Twilio WhatsApp webhook — handles incoming voice notes and text."""
    sender = request.form.get("From", "")
    num_media = int(request.form.get("NumMedia", 0))
    body = request.form.get("Body", "").strip()

    response_text = ""
    detected_lang = "sw"   # default to Swahili
    intent = "GENERAL"
    wav_path = None

    try:
        # ── Voice Note Flow ──────────────────────────────────────────────────
        if num_media > 0:
            media_url = request.form.get("MediaUrl0", "")
            media_type = request.form.get("MediaContentType0", "")
            logger.info(f"Received media from {sender}: {media_type}")

            ogg_path = os.path.join(TMP_DIR, "input.ogg")
            wav_path = os.path.join(TMP_DIR, "input.wav")
            download_audio(media_url, ogg_path)
            ogg_to_wav(ogg_path, wav_path)

            transcript, detected_lang = detect_language(wav_path)
            logger.info(f"Transcript: '{transcript}' | Language: {detected_lang}")
            body = transcript

        # ── Intent Routing ───────────────────────────────────────────────────
        if body:
            intent = route_intent(body, detected_lang)
            logger.info(f"Intent: {intent}")

            if intent == "CROP_DIAGNOSIS":
                response_text = get_crop_advice(body, detected_lang)
            elif intent == "MARKET_PRICE":
                response_text = get_market_prices(body, detected_lang)
            elif intent == "WEATHER":
                response_text = get_weather_forecast(detected_lang)
            else:
                response_text = _general_response(detected_lang)
        else:
            response_text = _general_response(detected_lang)

        # ── Persist to DB ────────────────────────────────────────────────────
        user = get_or_create_user(sender, detected_lang)
        log_query(user_id=user.id, transcript=body, intent=intent, response=response_text)

        # ── Build TwiML Response ─────────────────────────────────────────────
        twiml = MessagingResponse()
        msg = twiml.message()
        msg.body(response_text)

        # Attach voice note reply
        audio_path = text_to_speech(response_text, detected_lang)
        if audio_path and os.path.exists(audio_path):
            # Twilio requires a public URL; in production serve via /audio endpoint
            public_audio_url = request.host_url + "audio/" + os.path.basename(audio_path)
            msg.media(public_audio_url)

        return str(twiml), 200, {"Content-Type": "application/xml"}

    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        twiml = MessagingResponse()
        twiml.message("Samahani, kuna hitilafu. Jaribu tena. / Sorry, an error occurred. Please try again.")
        return str(twiml), 200, {"Content-Type": "application/xml"}


@app.route("/audio/<filename>")
def serve_audio(filename):
    """Serve generated audio files."""
    from flask import send_from_directory
    return send_from_directory(TMP_DIR, filename)


@app.route("/health")
def health():
    return {"status": "ok", "service": "Shamba-Smart"}, 200


def _general_response(lang: str) -> str:
    if lang == "ki":
        return (
            "Nĩngũkũrehereria ũtaaro. Ũĩ na ciũria cia:\n"
            "1. Magonjo ma mimea 🌿\n"
            "2. Bei ya soko 📊\n"
            "3. Hali ya hewa 🌦️"
        )
    return (
        "Karibu Shamba-Smart! Ninaweza kukusaidia na:\n"
        "1. Magonjwa ya mazao 🌿\n"
        "2. Bei za soko 📊\n"
        "3. Hali ya hewa 🌦️"
    )


if __name__ == "__main__":
    init_db()
    port = int(os.getenv("FLASK_PORT", 5000))
    logger.info(f"🌱 Shamba-Smart starting on port {port}")
    app.run(debug=os.getenv("FLASK_ENV") == "development", port=port)
