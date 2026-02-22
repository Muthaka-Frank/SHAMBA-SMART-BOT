"""
Weather Service — Open-Meteo free API (no API key required).
Returns a 3-day localized forecast in the farmer's language.
"""
import logging
import requests_cache
from retry_requests import retry
import openmeteo_requests

logger = logging.getLogger(__name__)

# Default coordinates: Nairobi (good fallback for Kenya)
DEFAULT_LAT = -1.286
DEFAULT_LON = 36.817

# WMO weather condition codes → human-readable Swahili/Kikuyu descriptions
WMO_CODES = {
    0: ("anga wazi", "ĩbũ nĩ rĩega"),
    1: ("anga nzuri", "ĩbũ nĩ rĩega"),
    2: ("mawingu kidogo", "ĩbũ nĩ na mawiu manini"),
    3: ("mawingu mengi", "ĩbũ nĩ na mawiu maingi"),
    45: ("ukungu", "ĩbũ nĩ na ukungu"),
    48: ("ukungu mzito", "ĩbũ nĩ na ukungu mũnene"),
    51: ("manyunyu kidogo", "mbura nini"),
    53: ("manyunyu ya wastani", "mbura ya kawaida"),
    55: ("manyunyu mazito", "mbura nene"),
    61: ("mvua kidogo", "mbura nini"),
    63: ("mvua ya wastani", "mbura ya kawaida"),
    65: ("mvua nzito", "mbura nene"),
    80: ("mvua ya asubuhi", "mbura ya gũthĩĩ"),
    81: ("mvua wastani", "mbura ya kawaida"),
    82: ("mvua kali", "mbura nene mũno"),
    95: ("dhoruba", "rũhũnĩ"),
    99: ("dhoruba na mvua", "rũhũnĩ na mbura"),
}

DAYS = {
    "sw": ["Leo", "Kesho", "Kesho kutwa"],
    "ki": ["Ũmũthĩ", "Rũciũ", "Rũciũ rwa gatatũ"],
}


def get_weather_forecast(lang: str = "sw", lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> str:
    """
    Fetch 3-day weather forecast from Open-Meteo.

    Args:
        lang: 'sw' (Swahili) or 'ki' (Kikuyu).
        lat:  Latitude (default Nairobi).
        lon:  Longitude (default Nairobi).

    Returns:
        Formatted 3-day forecast string.
    """
    try:
        cache_session = requests_cache.CachedSession(".weather_cache", expire_after=3600)
        retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=retry_session)

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ["weathercode", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "Africa/Nairobi",
            "forecast_days": 3,
        }
        responses = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        r = responses[0]
        daily = r.Daily()

        codes = [int(daily.Variables(0).ValuesAsNumpy()[i]) for i in range(3)]
        temps_max = [round(float(daily.Variables(1).ValuesAsNumpy()[i]), 1) for i in range(3)]
        temps_min = [round(float(daily.Variables(2).ValuesAsNumpy()[i]), 1) for i in range(3)]
        precip = [round(float(daily.Variables(3).ValuesAsNumpy()[i]), 1) for i in range(3)]

        day_names = DAYS.get(lang, DAYS["sw"])
        lines = []
        for i in range(3):
            code_desc = WMO_CODES.get(codes[i], ("hali ya hewa isiyojulikana", "ĩbũ tigĩthĩĩ"))[
                0 if lang == "sw" else 1
            ]
            rain_mm = precip[i]
            if lang == "ki":
                line = (
                    f"🌤️ {day_names[i]}: {code_desc}, "
                    f"baridi {temps_min[i]}°C–{temps_max[i]}°C, "
                    f"mbura {rain_mm}mm"
                )
            else:
                line = (
                    f"🌤️ {day_names[i]}: {code_desc}, "
                    f"joto {temps_min[i]}°C–{temps_max[i]}°C, "
                    f"mvua {rain_mm}mm"
                )
            lines.append(line)

        if lang == "ki":
            header = "Hali ya hewa (siku 3 zijazo, Nairobi):"
        else:
            header = "Hali ya hewa (siku 3 zijazo, Nairobi):"

        return header + "\n" + "\n".join(lines)

    except Exception as e:
        logger.error(f"Weather service error: {e}")
        if lang == "ki":
            return "Tũndũ tũthĩ na hali ya hewa rĩu. Jaribu baadaye."
        return "Samahani, hali ya hewa haipatikani sasa. Jaribu tena baadaye."
