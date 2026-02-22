"""
Unit Tests for Shamba-Smart Bot
Run with: pytest tests/ -v
"""
import pytest
from unittest.mock import patch, MagicMock


# ─── Test Intent Router ───────────────────────────────────────────────────────

class TestIntentRouter:

    @patch("services.intent_router._get_client")
    def test_crop_diagnosis_intent(self, mock_get_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "CROP_DIAGNOSIS"
        mock_get_client.return_value.chat.completions.create.return_value = mock_resp

        from services.intent_router import route_intent
        result = route_intent("Mahindi yangu yana madoadoa", "sw")
        assert result == "CROP_DIAGNOSIS"

    @patch("services.intent_router._get_client")
    def test_market_price_intent(self, mock_get_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "MARKET_PRICE"
        mock_get_client.return_value.chat.completions.create.return_value = mock_resp

        from services.intent_router import route_intent
        result = route_intent("Bei ya viazi leo Marigiti", "sw")
        assert result == "MARKET_PRICE"

    @patch("services.intent_router._get_client")
    def test_weather_intent(self, mock_get_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "WEATHER"
        mock_get_client.return_value.chat.completions.create.return_value = mock_resp

        from services.intent_router import route_intent
        result = route_intent("Je itanyesha kesho", "sw")
        assert result == "WEATHER"

    @patch("services.intent_router._get_client")
    def test_invalid_intent_defaults_to_general(self, mock_get_client):
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "RANDOM_GARBAGE"
        mock_get_client.return_value.chat.completions.create.return_value = mock_resp

        from services.intent_router import route_intent
        result = route_intent("blah blah", "sw")
        assert result == "GENERAL"

    @patch("services.intent_router._get_client")
    def test_error_returns_general(self, mock_get_client):
        mock_get_client.side_effect = Exception("API down")
        from services.intent_router import route_intent
        result = route_intent("test", "sw")
        assert result == "GENERAL"


# ─── Test Market Prices ───────────────────────────────────────────────────────

class TestMarketPrices:

    def test_maize_price_in_swahili(self, tmp_path, monkeypatch):
        import json
        prices = {
            "Mahindi": {
                "aliases": ["maize", "mahindi"],
                "unit": "kg",
                "markets": {
                    "Marigiti": {"price_ksh": 42},
                    "Wakulima": {"price_ksh": 40}
                }
            }
        }
        price_file = tmp_path / "market_prices.json"
        with open(price_file, "w") as f:
            json.dump({"commodities": prices}, f)

        monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(tmp_path))

        # Reload module to pick up new env var
        import importlib, services.market_prices as mp
        importlib.reload(mp)

        result = mp.get_market_prices("Bei ya mahindi", "sw")
        assert "Mahindi" in result or "mahindi" in result.lower()
        assert "42" in result or "40" in result

    def test_no_data_returns_error_message(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_BASE_PATH", str(tmp_path))

        import importlib, services.market_prices as mp
        importlib.reload(mp)

        result = mp.get_market_prices("bei ya sukari", "sw")
        assert "Samahani" in result or "hazipatikani" in result


# ─── Test Language Detector (unit level — mocked ASR) ────────────────────────

class TestLanguageDetector:

    @patch("asr.language_detector.transcribe_kikuyu", return_value=("mwariki", 0.85))
    @patch("asr.language_detector.transcribe_swahili", return_value=("mti", 0.60))
    def test_selects_kikuyu_when_higher_confidence(self, mock_sw, mock_ki):
        from asr.language_detector import detect_language
        transcript, lang = detect_language("fake.wav")
        assert lang == "ki"
        assert transcript == "mwariki"

    @patch("asr.language_detector.transcribe_kikuyu", return_value=("mwariki", 0.40))
    @patch("asr.language_detector.transcribe_swahili", return_value=("mahindi", 0.75))
    def test_selects_swahili_when_higher_confidence(self, mock_sw, mock_ki):
        from asr.language_detector import detect_language
        transcript, lang = detect_language("fake.wav")
        assert lang == "sw"
        assert transcript == "mahindi"

    @patch("asr.language_detector.transcribe_kikuyu", return_value=("", 0.0))
    @patch("asr.language_detector.transcribe_swahili", return_value=("", 0.0))
    def test_empty_audio_returns_swahili_default(self, mock_sw, mock_ki):
        from asr.language_detector import detect_language
        transcript, lang = detect_language("empty.wav")
        assert lang == "sw"


# ─── Test TTS ────────────────────────────────────────────────────────────────

class TestTTS:

    def test_tts_generates_mp3(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIO_TMP_PATH", str(tmp_path))
        
        # Mock gTTS to avoid network call in test
        with patch("services.tts.gTTS") as mock_gtts:
            mock_instance = MagicMock()
            mock_gtts.return_value = mock_instance
            
            from services.tts import text_to_speech
            result = text_to_speech("Karibu Shamba-Smart", "sw")
            
            assert result is not None
            assert result.endswith(".mp3")
            mock_instance.save.assert_called_once()

    def test_kikuyu_uses_swahili_tts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AUDIO_TMP_PATH", str(tmp_path))
        
        with patch("services.tts.gTTS") as mock_gtts:
            mock_gtts.return_value = MagicMock()
            from services.tts import text_to_speech
            text_to_speech("Nĩngũkũrehereria ũtaaro", "ki")
            
            # Should have been called with lang="sw" (Swahili fallback)
            call_kwargs = mock_gtts.call_args[1]
            assert call_kwargs["lang"] == "sw"
