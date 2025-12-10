# 🎭 Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AI-powered emotion detection from text supporting **English** and **Indonesian** languages.

## ✨ Features

- 🌐 **Bilingual**: English (LSTM) & Indonesian (IndoBERT 95.8% accuracy)
- 🚀 **REST API**: FastAPI with auto-generated Swagger documentation
- 💻 **Web Frontend**: Modern dark theme with real-time analysis
- ⚡ **Optimized**: ONNX conversion & prediction caching
- 📊 **6 Emotions**: joy, sadness, anger, fear, love, neutral

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/emotion-detection-system.git
cd emotion-detection-system

# Setup
cd emotion-detection-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt

# Run
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Access:**
- 🌐 Frontend: Open `emotion-detection-frontend/index.html`
- 📖 API Docs: http://localhost:8000/docs

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | English prediction |
| POST | `/api/v1/predict/id` | Indonesian prediction |
| POST | `/api/v1/predict/batch` | Batch prediction |
| GET | `/api/v1/health` | Health check |

## 💡 Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/predict/id",
    json={"text": "Saya sangat senang!"}
)
print(response.json())
# {"emotion": "joy", "confidence": 0.98, ...}
```

## 📁 Structure

```
├── emotion-detection-backend/   # FastAPI + ML
├── emotion-detection-frontend/  # HTML/CSS/JS UI
├── MANUAL_BOOK.md              # User guide
└── README.md
```

## 📊 Model Performance

| Model | Language | Accuracy |
|-------|----------|----------|
| LSTM | English | 73% |
| IndoBERT | Indonesian | **95.8%** |

## 📖 Documentation

- [User Manual](MANUAL_BOOK.md)
- [API Documentation](http://localhost:8000/docs)

## 📝 License

MIT License - see [LICENSE](LICENSE)
