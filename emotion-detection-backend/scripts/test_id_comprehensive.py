"""
Comprehensive Testing for Indonesian Emotion Model.
Tests with various Indonesian texts across all emotion categories.
"""
import os
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.ml.preprocessing import TextPreprocessor

MODEL_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_id"

def load_model():
    model = tf.keras.models.load_model(str(MODEL_DIR / "emotion_lstm_id.h5"))
    with open(str(MODEL_DIR / "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(str(MODEL_DIR / "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

def predict(text, model, tokenizer, label_encoder, preprocessor):
    processed = preprocessor.preprocess(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded, verbose=0)[0]
    emotion_idx = predictions.argmax()
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    confidence = float(predictions[emotion_idx])
    return emotion, confidence, predictions

def main():
    print("="*70)
    print("🇮🇩 COMPREHENSIVE INDONESIAN EMOTION MODEL TESTING")
    print("="*70)
    
    model, tokenizer, label_encoder = load_model()
    preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)
    classes = list(label_encoder.classes_)
    print(f"Classes: {classes}\n")
    
    # Test cases grouped by emotion
    test_cases = {
        "joy": [
            "Alhamdulillah senang banget hari ini",
            "Yeay akhirnya liburan!",
            "Bahagia banget bisa ketemu keluarga",
            "Seru banget acaranya tadi",
            "Wah dapat nilai bagus, seneng!",
            "Makasih ya, aku terharu banget",
            "Hari ini hari terbaik dalam hidupku",
            "Excited banget buat besok",
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
    
    results = {emotion: {"correct": 0, "total": 0, "details": []} for emotion in test_cases}
    
    for expected_emotion, texts in test_cases.items():
        print(f"\n{'='*70}")
        print(f"🎯 Testing {expected_emotion.upper()} ({len(texts)} samples)")
        print("-"*70)
        
        for text in texts:
            emotion, conf, _ = predict(text, model, tokenizer, label_encoder, preprocessor)
            is_correct = emotion == expected_emotion
            results[expected_emotion]["total"] += 1
            if is_correct:
                results[expected_emotion]["correct"] += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{status} \"{text[:45]}{'...' if len(text)>45 else ''}\"")
            print(f"   → {emotion} ({conf*100:.1f}%)" + (f" [Expected: {expected_emotion}]" if not is_correct else ""))
            
            results[expected_emotion]["details"].append({
                "text": text, "predicted": emotion, "confidence": conf, "correct": is_correct
            })
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY BY EMOTION")
    print("="*70)
    
    total_correct = 0
    total_samples = 0
    
    for emotion, data in results.items():
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        total_correct += data["correct"]
        total_samples += data["total"]
        bar = "█" * int(acc/10) + "░" * (10 - int(acc/10))
        print(f"{emotion:10} [{bar}] {acc:5.1f}% ({data['correct']}/{data['total']})")
    
    overall_acc = total_correct / total_samples * 100
    print("-"*70)
    print(f"{'OVERALL':10} [{('█' * int(overall_acc/10)) + ('░' * (10 - int(overall_acc/10)))}] {overall_acc:5.1f}% ({total_correct}/{total_samples})")
    
    # Misclassified examples
    print("\n" + "="*70)
    print("❌ MISCLASSIFIED EXAMPLES")
    print("="*70)
    
    for emotion, data in results.items():
        for item in data["details"]:
            if not item["correct"]:
                print(f'Expected {emotion}: "{item["text"]}"')
                print(f'   → Predicted: {item["predicted"]} ({item["confidence"]*100:.1f}%)\n')

if __name__ == "__main__":
    main()
