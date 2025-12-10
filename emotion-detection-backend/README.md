# 🎭 Emotion Detection System

AI-powered emotion detection from text supporting **English** and **Indonesian** languages.

## 🌟 Features

- **Bilingual Support**: English (LSTM) & Indonesian (IndoBERT 95.8% accuracy)
- **REST API**: FastAPI with auto-generated Swagger docs
- **Web Frontend**: Modern dark theme with probability charts
- **Optimized**: ONNX conversion & prediction caching
- **6 Emotions**: joy, sadness, anger, fear, love, neutral

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd emotion-detection-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt
```

### 2. Start API Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 3. Access Application
- **Frontend**: Open `emotion-detection-frontend/index.html`
- **API Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/api/v1/health

## 📡 API Endpoints

### English Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict` | Single text prediction |
| POST | `/api/v1/predict/batch` | Batch prediction (up to 100) |
| GET | `/api/v1/emotions` | List emotions |
| GET | `/api/v1/health` | Health check |

### Indonesian Predictions 🇮🇩
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/predict/id` | Single text (IndoBERT) |
| POST | `/api/v1/predict/id/batch` | Batch prediction |
| GET | `/api/v1/emotions/id` | List emotions |
| GET | `/api/v1/health/id` | Health check |

## 💡 Usage Examples

### Python
```python
import requests

# English
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={"text": "I am so happy today!"}
)
print(response.json())
# {"emotion": "joy", "confidence": 0.95, ...}

# Indonesian
response = requests.post(
    "http://localhost:8000/api/v1/predict/id",
    json={"text": "Saya sangat senang!"}
)
print(response.json())
# {"emotion": "joy", "confidence": 0.99, ...}
```

### cURL
```bash
# English
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'

# Indonesian
curl -X POST "http://localhost:8000/api/v1/predict/id" \
  -H "Content-Type: application/json" \
  -d '{"text": "Aku cinta kamu!"}'
```

## 📁 Project Structure

```
project-emotion-detected-system/
├── emotion-detection-backend/
│   ├── app/
│   │   ├── api/           # FastAPI routes & schemas
│   │   ├── ml/
│   │   │   ├── model.py           # English LSTM predictor
│   │   │   ├── model_indobert.py  # Indonesian predictor
│   │   │   ├── optimization.py    # ONNX & caching
│   │   │   └── saved_models/
│   │   └── config.py
│   ├── tests/             # Unit & integration tests
│   ├── docs/              # Postman collection
│   ├── scripts/           # Training & utility scripts
│   └── main.py
├── emotion-detection-frontend/
│   ├── index.html
│   ├── styles.css
│   └── app.js
└── saved_models_indobert/
    ├── indobert_emotion/  # PyTorch model
    └── indobert_onnx/     # ONNX model
```

## 🧪 Testing

```bash
cd emotion-detection-backend

# Install test dependencies
pip install pytest httpx

# Run unit tests
pytest tests/test_unit.py -v

# Run integration tests (API must be running)
pytest tests/test_integration.py -v
```

## 📊 Model Performance

| Model | Language | Test Accuracy | Inference |
|-------|----------|---------------|-----------|
| LSTM | English | 73% | Fast |
| IndoBERT | Indonesian | **95.8%** | 158ms |
| IndoBERT ONNX | Indonesian | 95.8% | **1.14x faster** |

## 🔧 Configuration

Environment variables (`.env`):
```env
MODEL_PATH=app/ml/saved_models/emotion_lstm.h5
TOKENIZER_PATH=app/ml/saved_models/tokenizer.pkl
MAX_WORDS=10000
MAX_SEQUENCE_LENGTH=100
API_TITLE=Emotion Detection API
```

## 📦 Dependencies

- Python 3.10+
- TensorFlow/Keras
- PyTorch + Transformers (IndoBERT)
- FastAPI + Uvicorn
- ONNX Runtime (optional optimization)

## 📝 License

MIT License

## 👤 Author

Emotion Detection System - AI-powered text emotion analysis
