# LAPORAN TEKNIS PROJECT
# Emotion Detection System
## Sistem Deteksi Emosi Berbasis Artificial Intelligence

---

# BAB 1: PENDAHULUAN

## 1.1 Latar Belakang
Deteksi emosi dari teks merupakan salah satu cabang dari Natural Language Processing (NLP) yang bertujuan untuk mengidentifikasi emosi yang terkandung dalam sebuah teks. Sistem ini memiliki banyak aplikasi seperti analisis sentimen media sosial, customer feedback analysis, dan mental health monitoring.

## 1.2 Tujuan Project
Membangun sistem deteksi emosi yang dapat:
1. Menganalisis teks berbahasa Inggris dan Indonesia
2. Mengklasifikasikan teks ke dalam 6 kategori emosi
3. Menyediakan REST API untuk integrasi dengan aplikasi lain
4. Menyediakan antarmuka web yang user-friendly

## 1.3 Ruang Lingkup
- Deteksi 6 emosi: joy, sadness, anger, fear, love, neutral
- Support 2 bahasa: English dan Indonesian
- Platform: Web-based application

---

# BAB 2: TEKNOLOGI YANG DIGUNAKAN

## 2.1 Technology Stack

### Frontend
| Teknologi | Versi | Keterangan |
|-----------|-------|------------|
| HTML5 | - | Struktur halaman web |
| CSS3 | - | Styling dengan dark theme |
| JavaScript | ES6+ | Logic dan API integration |

### Backend
| Teknologi | Versi | Keterangan |
|-----------|-------|------------|
| Python | 3.12 | Bahasa pemrograman utama |
| FastAPI | 0.100+ | REST API framework |
| Uvicorn | 0.23+ | ASGI web server |
| Pydantic | 2.0+ | Data validation |

### Machine Learning
| Teknologi | Versi | Keterangan |
|-----------|-------|------------|
| TensorFlow | 2.15+ | Deep learning framework |
| Keras | 3.0+ | High-level neural network API |
| PyTorch | 2.0+ | Deep learning (untuk IndoBERT) |
| Transformers | 4.35+ | HuggingFace library |
| NLTK | 3.8+ | Text preprocessing |

## 2.2 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACE                          │
│                  (HTML + CSS + JavaScript)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP Request
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     REST API (FastAPI)                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   /predict  │  │ /predict/id │  │  /predict/batch     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
└─────────┼────────────────┼────────────────────┼─────────────┘
          │                │                    │
          ▼                ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│   LSTM Model    │  │  IndoBERT Model │  │  Batch Processor │
│   (English)     │  │  (Indonesian)   │  │                  │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

---

# BAB 3: DATASET

## 3.1 Dataset English
- **Sumber**: Kaggle - "Emotions Dataset for NLP"
- **Format**: Text file dengan separator semicolon (;)
- **Jumlah Data**:
  - Training: 16,000 samples
  - Validation: 2,000 samples
  - Test: 2,000 samples
- **Label**: joy, sadness, anger, fear, surprise, love

## 3.2 Dataset Indonesian
- **Sumber**: Indonesian Public Opinion Dataset (Twitter)
- **Format**: CSV file per emosi
- **Jumlah Data**:
  - Training: 5,664 samples
  - Validation: 708 samples
  - Test: 708 samples
- **Label**: joy, sadness, anger, fear, love, neutral

## 3.3 Preprocessing Data
1. **Lowercase**: Konversi ke huruf kecil
2. **Remove URLs**: Hapus tautan
3. **Remove Mentions**: Hapus @username
4. **Remove Hashtags**: Hapus simbol #
5. **Tokenization**: Pecah teks menjadi token
6. **Padding**: Samakan panjang sequence

---

# BAB 4: MODEL DAN ALGORITMA

## 4.1 Model English - LSTM

### Arsitektur Neural Network:
```
Input Text
    ↓
Embedding Layer (10,000 vocab × 100 dim)
    ↓
SpatialDropout1D (0.2)
    ↓
Bidirectional LSTM (128 units)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dropout (0.3)
    ↓
Dense (64 units, ReLU)
    ↓
Dense (32 units, ReLU)
    ↓
Dense (6 units, Softmax)
    ↓
Output: Emotion Probability
```

### Hyperparameters:
| Parameter | Nilai |
|-----------|-------|
| Max Words | 10,000 |
| Embedding Dim | 100 |
| Max Sequence Length | 100 |
| LSTM Units | 128, 64 |
| Dropout Rate | 0.2, 0.3 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 10 (early stopping) |

### Akurasi: **73%**

## 4.2 Model Indonesian - IndoBERT

### Tentang IndoBERT:
- **Base Model**: indobenchmark/indobert-base-p1
- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Pre-training**: Corpus bahasa Indonesia (Wikipedia, news, dll)
- **Parameters**: ~110 million

### Fine-tuning Process:
1. Load pre-trained IndoBERT
2. Add classification head (6 classes)
3. Fine-tune dengan dataset emosi Indonesia
4. Training di Google Colab dengan GPU T4

### Hyperparameters:
| Parameter | Nilai |
|-----------|-------|
| Max Length | 128 tokens |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Epochs | 5 |
| Warmup Ratio | 0.1 |
| Weight Decay | 0.01 |

### Akurasi: **95.8%**

---

# BAB 5: HASIL DAN EVALUASI

## 5.1 Perbandingan Model

| Model | Bahasa | Akurasi | Inference Time |
|-------|--------|---------|----------------|
| LSTM | English | 73% | ~10ms |
| IndoBERT | Indonesian | 95.8% | ~160ms |
| IndoBERT ONNX | Indonesian | 95.8% | ~140ms |

## 5.2 Confusion Matrix (Approx)

### English LSTM:
- Joy: 75%
- Sadness: 80%
- Anger: 65%
- Fear: 78%
- Surprise: 60%
- Love: 70%

### Indonesian IndoBERT:
- Joy: 87.5%
- Sadness: 100%
- Anger: 100%
- Fear: 100%
- Love: 100%
- Neutral: 87.5%

## 5.3 Optimisasi
- **ONNX Conversion**: 1.14x faster inference
- **Caching**: Near-instant untuk repeated predictions

---

# BAB 6: FITUR APLIKASI

## 6.1 REST API Endpoints

| Method | Endpoint | Fungsi |
|--------|----------|--------|
| GET | /api/v1/health | Health check English model |
| GET | /api/v1/health/id | Health check Indonesian model |
| POST | /api/v1/predict | Prediksi emosi English |
| POST | /api/v1/predict/id | Prediksi emosi Indonesian |
| POST | /api/v1/predict/batch | Batch prediction English |
| POST | /api/v1/predict/id/batch | Batch prediction Indonesian |
| GET | /api/v1/emotions | List emosi English |
| GET | /api/v1/emotions/id | List emosi Indonesian |

## 6.2 Fitur Frontend
1. ✅ Bilingual support (EN/ID)
2. ✅ Real-time emotion analysis
3. ✅ Probability distribution chart
4. ✅ Analysis history (localStorage)
5. ✅ Dark theme UI
6. ✅ Responsive design
7. ✅ Keyboard shortcuts

---

# BAB 7: KESIMPULAN

## 7.1 Hasil yang Dicapai
1. Berhasil membangun sistem deteksi emosi bilingual
2. Model IndoBERT mencapai akurasi 95.8% untuk bahasa Indonesia
3. REST API yang scalable dan well-documented
4. Frontend yang modern dan user-friendly

## 7.2 Kelebihan
- Akurasi tinggi untuk bahasa Indonesia
- API yang mudah diintegrasikan
- Dokumentasi lengkap
- Open source

## 7.3 Keterbatasan
- Model English accuracy masih 73%
- IndoBERT membutuhkan resource lebih besar
- Belum support bahasa lain

## 7.4 Pengembangan Selanjutnya
- Auto language detection
- Docker containerization
- Cloud deployment
- Mobile application

---

# REFERENSI

1. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Wilie, B., et al. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation.
4. FastAPI Documentation. https://fastapi.tiangolo.com/
5. HuggingFace Transformers. https://huggingface.co/docs/transformers/

---

**Repository**: https://github.com/alokmakanjambu/emotion-detection-system

---
*Dokumen ini dibuat sebagai laporan teknis project Emotion Detection System*
