# 📖 Emotion Detection System - User Manual

## Daftar Isi

1. [Pendahuluan](#1-pendahuluan)
2. [Instalasi](#2-instalasi)
3. [Menjalankan Aplikasi](#3-menjalankan-aplikasi)
4. [Menggunakan Frontend](#4-menggunakan-frontend)
5. [Menggunakan API](#5-menggunakan-api)
6. [Training Model](#6-training-model)
7. [Troubleshooting](#7-troubleshooting)
8. [FAQ](#8-faq)

---

## 1. Pendahuluan

### 1.1 Apa itu Emotion Detection System?
Sistem deteksi emosi berbasis AI yang dapat menganalisis teks dan menentukan emosi yang terkandung di dalamnya. Mendukung dua bahasa:
- **Bahasa Inggris** - menggunakan model LSTM
- **Bahasa Indonesia** - menggunakan model IndoBERT (95.8% akurasi)

### 1.2 Emosi yang Didukung
| Emosi | Emoji | Deskripsi |
|-------|-------|-----------|
| Joy | 😊 | Kebahagiaan, kesenangan |
| Sadness | 😢 | Kesehihan, duka |
| Anger | 😠 | Kemarahan, frustasi |
| Fear | 😨 | Ketakutan, kecemasan |
| Love | 😍 | Cinta, kasih sayang |
| Neutral | 😐 | Netral (hanya Indonesia) |

### 1.3 Arsitektur Sistem
```
┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   FastAPI       │
│   (HTML/JS)     │     │   Backend       │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐           ┌───────▼───────┐
              │   LSTM    │           │   IndoBERT    │
              │ (English) │           │ (Indonesian)  │
              └───────────┘           └───────────────┘
```

---

## 2. Instalasi

### 2.1 Persyaratan Sistem
- Python 3.10 atau lebih baru
- Minimal 4GB RAM (8GB recommended untuk IndoBERT)
- 2GB ruang disk

### 2.2 Clone Repository
```bash
git clone https://github.com/USERNAME/emotion-detection-system.git
cd emotion-detection-system
```

### 2.3 Setup Virtual Environment
```bash
# Windows
cd emotion-detection-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
cd emotion-detection-backend
python -m venv .venv
source .venv/bin/activate
```

### 2.4 Install Dependencies
```bash
pip install -r requirements.txt
```

### 2.5 Download Model IndoBERT (Opsional)
Jika model IndoBERT belum tersedia, Anda perlu melatih menggunakan Google Colab:
1. Buka `notebooks/IndoBERT_Emotion_Training.ipynb`
2. Upload ke Google Colab
3. Jalankan semua cell
4. Download model dan extract ke `saved_models_indobert/`

---

## 3. Menjalankan Aplikasi

### 3.1 Menjalankan Backend API
```bash
cd emotion-detection-backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Anda akan melihat output:
```
🚀 Starting Emotion Detection API...
✅ English LSTM model loaded!
✅ Indonesian IndoBERT model loaded!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3.2 Menjalankan Frontend
**Opsi 1: Langsung buka file**
- Buka file `emotion-detection-frontend/index.html` di browser

**Opsi 2: Menggunakan HTTP Server**
```bash
cd emotion-detection-frontend
python -m http.server 3000
# Buka http://localhost:3000
```

### 3.3 Mengakses Dokumentasi API
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 4. Menggunakan Frontend

### 4.1 Tampilan Utama
![Frontend](../emotion-detection-frontend/screenshot.png)

### 4.2 Langkah-langkah
1. **Pilih Bahasa**: Klik tombol 🇮🇩 Indonesia atau 🇺🇸 English
2. **Masukkan Teks**: Ketik atau paste teks di text area
3. **Klik Analisis**: Tekan tombol "Analisis Emosi"
4. **Lihat Hasil**: Emosi yang terdeteksi akan muncul dengan confidence score

### 4.3 Fitur Tambahan
- **History**: Riwayat analisis tersimpan otomatis
- **Probability Chart**: Grafik distribusi probabilitas semua emosi
- **Keyboard Shortcut**: Ctrl+Enter untuk submit

---

## 5. Menggunakan API

### 5.1 Health Check
```bash
# English Model
curl http://localhost:8000/api/v1/health

# Indonesian Model
curl http://localhost:8000/api/v1/health/id
```

### 5.2 Prediksi Tunggal

**English:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I am so happy today!"}'
```

**Indonesian:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/id" \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya sangat senang hari ini!"}'
```

**Response:**
```json
{
  "text": "Saya sangat senang hari ini!",
  "emotion": "joy",
  "confidence": 0.9876,
  "probabilities": {
    "anger": 0.001,
    "fear": 0.002,
    "joy": 0.9876,
    "love": 0.005,
    "neutral": 0.002,
    "sadness": 0.003
  }
}
```

### 5.3 Batch Prediksi
```bash
curl -X POST "http://localhost:8000/api/v1/predict/id/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Saya senang",
      "Saya sedih",
      "Saya marah"
    ]
  }'
```

### 5.4 Python Client
```python
import requests

def predict_emotion(text, lang="id"):
    endpoint = "/api/v1/predict/id" if lang == "id" else "/api/v1/predict"
    response = requests.post(
        f"http://localhost:8000{endpoint}",
        json={"text": text}
    )
    return response.json()

# Contoh penggunaan
result = predict_emotion("Aku cinta kamu!", "id")
print(f"Emosi: {result['emotion']} ({result['confidence']*100:.1f}%)")
```

---

## 6. Training Model

### 6.1 Dataset
Dataset disimpan di:
- `archive/` - Dataset English (train.txt, val.txt, test.txt)
- `dataset-bahasa-processed/` - Dataset Indonesian

Format dataset: `text;label` (semicolon separated)

### 6.2 Training English Model
```bash
cd emotion-detection-backend
python -m app.ml.train
```

### 6.3 Training IndoBERT (Google Colab)
1. Upload `notebooks/IndoBERT_Emotion_Training.ipynb` ke Colab
2. Aktifkan GPU: Runtime → Change runtime type → GPU
3. Upload dataset (train.txt, val.txt, test.txt)
4. Run all cells
5. Download model dan letakkan di `saved_models_indobert/indobert_emotion/`

---

## 7. Troubleshooting

### 7.1 API Tidak Bisa Diakses
**Problem:** Browser tidak bisa connect ke localhost:8000

**Solusi:**
1. Pastikan API sudah running
2. Check port tidak sedang digunakan: `netstat -an | findstr 8000`
3. Coba ganti port: `uvicorn main:app --port 8001`

### 7.2 Model Tidak Ter-load
**Problem:** Error "Model not loaded"

**Solusi:**
1. Pastikan file model ada di `app/ml/saved_models/`
2. Check permission file
3. Pastikan TensorFlow terinstall dengan benar

### 7.3 IndoBERT Lambat
**Problem:** Prediksi Indonesian memakan waktu lama

**Solusi:**
1. Gunakan ONNX model untuk 1.14x speedup
2. Enable caching untuk repeat predictions
3. Upgrade RAM minimal 8GB

### 7.4 CORS Error di Frontend
**Problem:** Frontend tidak bisa akses API

**Solusi:**
1. Pastikan CORS sudah enabled di `main.py`
2. Jalankan frontend via HTTP server, bukan file://

---

## 8. FAQ

### Q: Berapa akurasi model?
**A:** English LSTM: 73%, Indonesian IndoBERT: 95.8%

### Q: Bahasa apa saja yang didukung?
**A:** English dan Indonesian

### Q: Berapa maksimal panjang teks?
**A:** 1000 karakter per request

### Q: Apakah gratis?
**A:** Ya, open source dengan lisensi MIT

### Q: Bagaimana menambah bahasa baru?
**A:** Anda perlu:
1. Kumpulkan dataset dengan format text;label
2. Train model baru
3. Buat predictor module
4. Tambahkan API endpoint

### Q: Apakah bisa deploy ke cloud?
**A:** Ya, project ini siap deploy ke:
- AWS (EC2, Lambda)
- Google Cloud (Cloud Run, GCE)
- Azure (App Service)
- Heroku, Railway, Vercel

---

## 📞 Kontak & Support

Jika ada pertanyaan atau masalah, silakan:
1. Buka Issue di GitHub
2. Check dokumentasi API di `/docs`
3. Review kode di repository

---

*Emotion Detection System - AI-powered text emotion analysis*
