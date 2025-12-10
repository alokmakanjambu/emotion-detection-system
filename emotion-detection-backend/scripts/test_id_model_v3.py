"""
Test Indonesian Model v3 with comprehensive test suite.
"""
import os
import sys
import pickle
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_id_v3"

# Same slang dict as training
SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'ngga': 'tidak',
    'nggak': 'tidak', 'enggak': 'tidak', 'tdk': 'tidak',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'tp': 'tapi', 'jg': 'juga', 'sm': 'sama', 'bgt': 'banget',
    'bngt': 'banget', 'skrg': 'sekarang', 'blm': 'belum',
    'sdh': 'sudah', 'udh': 'sudah', 'kmrn': 'kemarin', 'bsk': 'besok',
    'org': 'orang', 'lg': 'lagi', 'dr': 'dari', 'kl': 'kalau',
    'klo': 'kalau', 'klu': 'kalau', 'emg': 'memang', 'gmn': 'bagaimana',
    'gimana': 'bagaimana', 'knp': 'kenapa', 'spy': 'supaya',
    'bkn': 'bukan', 'bnyk': 'banyak', 'sy': 'saya', 'ak': 'aku',
    'gue': 'aku', 'gw': 'aku', 'lo': 'kamu', 'lu': 'kamu',
    'doi': 'dia', 'dy': 'dia', 'mrk': 'mereka',
}

def preprocess(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    return re.sub(r'\s+', ' ', ' '.join(words)).strip()

def load_model():
    model = tf.keras.models.load_model(str(MODEL_DIR / "emotion_lstm_id_v3.h5"))
    with open(str(MODEL_DIR / "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(str(MODEL_DIR / "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict(text, model, tokenizer, label_encoder):
    processed = preprocess(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded, verbose=0)[0]
    emotion_idx = predictions.argmax()
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    confidence = float(predictions[emotion_idx])
    return emotion, confidence

def main():
    print("="*70)
    print("🇮🇩 TESTING INDONESIAN MODEL V3 (Slang Normalization Only)")
    print("="*70)
    
    model, tokenizer, label_encoder = load_model()
    print(f"Classes: {list(label_encoder.classes_)}\n")
    
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
            emotion, conf = predict(text, model, tokenizer, label_encoder)
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
