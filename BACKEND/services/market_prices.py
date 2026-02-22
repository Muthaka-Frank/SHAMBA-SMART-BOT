"""
Market Prices Service
Reads commodity prices from DATABASE/knowledge_base/market_prices.json
and returns a formatted response in the farmer's language.
"""
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _load_prices() -> dict:
    kb_path = os.getenv("KNOWLEDGE_BASE_PATH", "../DATABASE/knowledge_base")
    prices_file = Path(kb_path) / "market_prices.json"
    if not prices_file.exists():
        logger.warning(f"market_prices.json not found at {prices_file}")
        return {}
    with open(prices_file, encoding="utf-8") as f:
        return json.load(f)


def get_market_prices(query: str, lang: str) -> str:
    """
    Parse the farmer's query to identify commodity, then return prices
    from Marigiti and Wakulima markets.

    Args:
        query: Farmer's message (e.g. "Bei ya mahindi Marigiti").
        lang:  'sw' or 'ki'.

    Returns:
        Formatted price string in the farmer's language.
    """
    prices = _load_prices()
    if not prices:
        return _no_data_msg(lang)

    query_lower = query.lower()
    matched_commodity = None

    # Keyword matching against commodity names and aliases
    for commodity, data in prices.items():
        aliases = [commodity.lower()] + [a.lower() for a in data.get("aliases", [])]
        if any(alias in query_lower for alias in aliases):
            matched_commodity = commodity
            break

    if not matched_commodity:
        # Return top 5 prices as a summary
        return _summary_response(prices, lang)

    return _commodity_response(matched_commodity, prices[matched_commodity], lang)


def _commodity_response(name: str, data: dict, lang: str) -> str:
    markets = data.get("markets", {})
    unit = data.get("unit", "kg")
    lines = []
    for market, info in markets.items():
        price = info.get("price_ksh")
        lines.append(f"  • {market}: KSh {price}/{unit}")

    price_block = "\n".join(lines)
    if lang == "ki":
        return f"Bei ya {name} (tarehe ya leo):\n{price_block}\n\nĨndi ũhũthire ũhoro ũyũ kũrĩa gũkũria ĩkomu yaku."
    return f"Bei ya {name} leo:\n{price_block}\n\nTumia habari hii kufanya uamuzi wa soko lako."


def _summary_response(prices: dict, lang: str) -> str:
    lines = []
    for commodity, data in list(prices.items())[:6]:
        unit = data.get("unit", "kg")
        for market, info in data.get("markets", {}).items():
            price = info.get("price_ksh")
            lines.append(f"  • {commodity} ({market}): KSh {price}/{unit}")
            break  # one market per commodity in summary
    block = "\n".join(lines)
    if lang == "ki":
        return f"Bei za mazao leo (soko):\n{block}"
    return f"Bei za mazao leo:\n{block}"


def _no_data_msg(lang: str) -> str:
    if lang == "ki":
        return "Tũndũ tũthĩ na bei ĩyo rĩu. Jaribu baadaye."
    return "Samahani, bei za soko hazipatikani sasa. Jaribu tena baadaye."
