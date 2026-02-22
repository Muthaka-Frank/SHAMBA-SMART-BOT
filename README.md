# 🌱 Shamba-Smart — Voice Assistant for Kenyan Farmers

A WhatsApp-linked voice bot that detects Swahili or Kikuyu automatically, retrieves agricultural advice from a RAG knowledge base, and replies with expert voice responses.

---

## 📁 Project Structure

```
MSHAURI-WA-WAKULIMA-BOT/
├── BACKEND/
│   ├── app.py                  ← Flask webhook server
│   ├── requirements.txt        ← Python dependencies
│   ├── .env.example            ← Config template
│   ├── asr/
│   │   ├── kikuyu_asr.py       ← Kikuyu ASR (HuggingFace)
│   │   ├── swahili_asr.py      ← Swahili ASR (HuggingFace)
│   │   └── language_detector.py← Cascading ASR — picks best model
│   ├── services/
│   │   ├── intent_router.py    ← Groq LLM: classify query intent
│   │   ├── crop_advisor.py     ← RAG-powered crop diagnosis
│   │   ├── rag_indexer.py      ← One-time indexer for knowledge base
│   │   ├── market_prices.py    ← Marigiti & Wakulima prices
│   │   ├── weather.py          ← Open-Meteo 3-day forecast
│   │   ├── tts.py              ← gTTS text-to-speech
│   │   └── audio_utils.py      ← OGG→WAV conversion
│   ├── database/
│   │   └── db.py               ← SQLAlchemy ORM + helpers
│   └── tests/
│       └── test_shamba_smart.py← Unit tests
│
├── DATABASE/
│   ├── init_db.py              ← Create SQLite schema
│   ├── knowledge_base/
│   │   ├── kalro_crop_diseases.txt  ← Crop disease guide
│   │   ├── soil_management.txt      ← Fertilizers & soil
│   │   ├── pest_control.txt         ← IPM pest guide
│   │   ├── seasonal_calendar.txt    ← Planting calendar by region
│   │   └── market_prices.json       ← Commodity price data
│   └── chroma_store/           ← Created by rag_indexer.py
│
└── FRONTEND/                   ← Future admin dashboard
```

---

## ⚡ Quick Start

### 1. Install Dependencies

```powershell
cd MSHAURI-WA-WAKULIMA-BOT\BACKEND
..\ukulima-env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment

```powershell
Copy-Item .env.example .env
# Edit .env and fill in your Twilio + Groq API keys
notepad .env
```

### 3. Initialize the Database

```powershell
python ..\DATABASE\init_db.py
```

### 4. Index the Knowledge Base (RAG)

```powershell
# This embeds all seed documents into ChromaDB
python services\rag_indexer.py
```

> ⏱️ First run downloads the embedding model (~400MB). Subsequent runs are instant.

### 5. Start the Bot Server

```powershell
python app.py
```

### 6. Expose with ngrok

```powershell
ngrok http 5000
```

### 7. Configure Twilio

Set your Twilio WhatsApp Sandbox webhook URL to:
```
https://<your-ngrok-url>/webhook
```

---

## 🧪 Run Tests

```powershell
..\ukulima-env\Scripts\Activate.ps1
python -m pytest tests/ -v
```

---

## 🗣️ Example Usage

| Farmer says | Language | Bot action |
|---|---|---|
| *"Mahindi yangu yana madoadoa"* | Swahili | Crop diagnosis (Grey Leaf Spot) |
| *"Mwariki wakwa nĩũmĩte mathangũ"* | Kikuyu | Castor wilt diagnosis |
| *"Bei ya mahindi Marigiti"* | Swahili | Market price query |
| *"Je, itanyesha kesho?"* | Swahili | 3-day weather forecast |

---

## 🔑 Required API Keys

| Service | Get it at | Cost |
|---|---|---|
| Twilio | [twilio.com](https://twilio.com) | Free trial |
| Groq | [console.groq.com](https://console.groq.com) | Free |
| Open-Meteo (weather) | Automatic | Always free |
| HuggingFace (ASR) | Automatic download | Always free |
