"""
Test IndoBERT Emotion Model.
Tests the fine-tuned IndoBERT model trained on Google Colab.
"""
import os
import sys
import re
import pickle
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model path - adjust this path if needed
MODEL_DIR = Path(r"d:\project-emotion-detected-system\saved_models_indobert\indobert_emotion")

def preprocess_text(text):
    """Simple text cleaning."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_model():
    """Load IndoBERT model, tokenizer, and label encoder."""
    print("📂 Loading IndoBERT model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR / "model"))
    
    # Load label encoder
    with open(str(MODEL_DIR / "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    return model, tokenizer, label_encoder, device

def predict_emotion(text, model, tokenizer, label_encoder, device, max_len=128):
    """Predict emotion for a single text."""
    processed = preprocess_text(text)
    
    encoding = tokenizer(
        processed,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    emotion = label_encoder.inverse_transform([pred_idx])[0]
    all_probs = {label: probs[0][i].item() for i, label in enumerate(label_encoder.classes_)}
    
    return emotion, confidence, all_probs

def main():
    print("="*70)
    print("🇮🇩 TESTING INDOBERT EMOTION MODEL")
    print("="*70)
    
    # Load model
    model, tokenizer, label_encoder, device = load_model()
    
    # Test cases
    test_cases = {
        "joy": [
            "Alhamdulillah senang banget hari ini",
            "Yeay akhirnya liburan!",
            "Bahagia banget bisa ketemu keluarga",
            "Seru banget acaranya tadi",
            "Wah dapat nilai bagus, seneng!",
            "Hari ini hari terbaik dalam hidupku",
            "Excited banget buat besok",
            "Asik dapet hadiah!",
        ],
        "sadness": [
            "Sedih banget dia pergi",
            "Hatiku hancur berkeping-keping",
            "Kenapa hidupku begini terus",
            "Menyesal banget udah lakuin itu",
            "Kangen mama, pengen pulang",
            "Nangis terus dari tadi",
            "Kesepian banget akhir-akhir ini",
            "Patah hati lagi, capek",
        ],
        "anger": [
            "Kesel banget sama orang itu!",
            "Geram aku lihat kelakuannya",
            "Benci banget sama pembohong",
            "Emosi jiwa gue liat berita ini",
            "Muak sama attitude dia",
            "Nyebelin banget sih orang itu",
            "Marah besar aku!",
            "Kurang ajar banget dia",
        ],
        "fear": [
            "Takut banget sama hantu",
            "Ngeri denger cerita tadi",
            "Khawatir sama kondisi ekonomi",
            "Cemas banget nunggu hasil tes",
            "Deg-degan mau interview",
            "Parno banget keluar malam",
            "Serem banget tempat ini",
            "Panik aku kehilangan dompet",
        ],
        "love": [
            "Sayang banget sama pacar",
            "Cinta mati sama kamu",
            "Kangen pelukan kamu",
            "Rindu banget sama doi",
            "Love you so much sayangku",
            "Aku sayang keluargaku",
            "Dia cinta pertamaku",
            "Romantis banget momen itu",
        ],
        "neutral": [
            "Hari ini cuaca mendung",
            "Besok ada rapat jam 9",
            "Lagi nunggu bus",
            "Tadi makan nasi goreng",
            "Baru bangun tidur",
            "Lagi kerja dari rumah",
            "Minggu depan ada ujian",
            "Butuh istirahat sebentar",
        ],
    }
    
    results = {e: {"correct": 0, "total": 0} for e in test_cases}
    
    for expected, texts in test_cases.items():
        print(f"\n{'='*70}")
        print(f"🎯 {expected.upper()} ({len(texts)} samples)")
        print("-"*70)
        
        for text in texts:
            emotion, conf, _ = predict_emotion(text, model, tokenizer, label_encoder, device)
            is_correct = emotion == expected
            results[expected]["total"] += 1
            if is_correct:
                results[expected]["correct"] += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} \"{text[:40]}{'...' if len(text)>40 else ''}\" → {emotion} ({conf*100:.1f}%)")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    total_c, total_t = 0, 0
    for e, d in results.items():
        acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        total_c += d["correct"]
        total_t += d["total"]
        bar = "█" * int(acc/10) + "░" * (10-int(acc/10))
        print(f"{e:10} [{bar}] {acc:5.1f}% ({d['correct']}/{d['total']})")
    
    overall = total_c / total_t * 100
    print("-"*70)
    print(f"{'OVERALL':10} [{'█'*int(overall/10)}{'░'*(10-int(overall/10))}] {overall:5.1f}% ({total_c}/{total_t})")

if __name__ == "__main__":
    main()
