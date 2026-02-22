"""
Intent Router — uses Groq LLM to classify farmer's message into one of:
  CROP_DIAGNOSIS | MARKET_PRICE | WEATHER | GENERAL
"""
import logging
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


INTENT_SYSTEM_PROMPT = """You are an intent classifier for a Kenyan farming assistant bot.
Classify the farmer's message into EXACTLY one of these intents:
- CROP_DIAGNOSIS  : questions about plant diseases, pests, soil, fertilizers, or crop health
- MARKET_PRICE    : questions about commodity prices or market rates
- WEATHER         : questions about weather, rain, forecasts, or planting conditions
- GENERAL         : greetings or anything else

Respond with ONLY the intent label. No explanation. No punctuation. Just the label.
Examples:
  "Mahindi yangu yana madoadoa" → CROP_DIAGNOSIS
  "Bei ya viazi leo" → MARKET_PRICE
  "Je, itanyesha kesho?" → WEATHER
  "Habari" → GENERAL
"""


def route_intent(transcript: str, lang: str = "sw") -> str:
    """
    Classify transcript intent using Groq LLM.

    Args:
        transcript: Farmer's message (Swahili or Kikuyu text).
        lang: Detected language code ('sw' or 'ki').

    Returns:
        One of: 'CROP_DIAGNOSIS', 'MARKET_PRICE', 'WEATHER', 'GENERAL'
    """
    valid_intents = {"CROP_DIAGNOSIS", "MARKET_PRICE", "WEATHER", "GENERAL"}
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        intent = response.choices[0].message.content.strip().upper()
        if intent not in valid_intents:
            logger.warning(f"Unexpected intent '{intent}', defaulting to GENERAL")
            return "GENERAL"
        return intent

    except Exception as e:
        logger.error(f"Intent routing error: {e}")
        return "GENERAL"
