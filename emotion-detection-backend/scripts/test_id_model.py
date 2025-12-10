"""
Test Indonesian Emotion Model with sample texts.
"""
import os
import sys
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.ml.preprocessing import TextPreprocessor

# Paths
MODEL_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_id"

def load_model_artifacts():
    """Load model, tokenizer, and label encoder."""
    model = tf.keras.models.load_model(str(MODEL_DIR / "emotion_lstm_id.h5"))
    
    with open(str(MODEL_DIR / "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    
    with open(str(MODEL_DIR / "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    return model, tokenizer, label_encoder

def predict_emotion(text, model, tokenizer, label_encoder, preprocessor):
    """Predict emotion for a single text."""
    # Preprocess
    processed = preprocessor.preprocess(text)
    
    # Tokenize
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
    
    # Predict
    predictions = model.predict(padded, verbose=0)[0]
    
    # Get results
    emotion_idx = predictions.argmax()
    emotion = label_encoder.inverse_transform([emotion_idx])[0]
    confidence = float(predictions[emotion_idx])
    
    return emotion, confidence, dict(zip(label_encoder.classes_, predictions.tolist()))

def main():
    print("="*60)
    print("🇮🇩 TESTING INDONESIAN EMOTION MODEL")
    print("="*60)
    
    # Load model
    print("\n📂 Loading model artifacts...")
    model, tokenizer, label_encoder = load_model_artifacts()
    preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)
    
    print(f"✅ Model loaded!")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    # Test texts in Indonesian
    test_texts = [
        ("Saya sangat senang hari ini!", "joy"),
        ("Saya sedih sekali karena gagal ujian", "sadness"),
        ("Aku marah banget sama kamu!", "anger"),
        ("Saya takut dengan keadaan ini", "fear"),
        ("Aku cinta kamu selamanya", "love"),
        ("Hari ini biasa saja", "neutral"),
        ("Wah keren banget hadiahnya!", "joy"),
        ("Hatiku hancur melihat dia pergi", "sadness"),
        ("Geram aku lihat berita ini", "anger"),
        ("Ngeri banget gempa tadi malam", "fear"),
        ("Sayang, aku kangen kamu", "love"),
    ]
    
    print("\n" + "-"*60)
    print("📝 TEST PREDICTIONS")
    print("-"*60)
    
    correct = 0
    total = len(test_texts)
    
    for text, expected in test_texts:
        emotion, confidence, all_probs = predict_emotion(
            text, model, tokenizer, label_encoder, preprocessor
        )
        is_correct = emotion == expected
        if is_correct:
            correct += 1
        
        status = "✅" if is_correct else "❌"
        print(f"\n{status} Text: \"{text}\"")
        print(f"   Expected: {expected}")
        print(f"   Predicted: {emotion} ({confidence*100:.1f}%)")
        if not is_correct:
            print(f"   All probs: {', '.join([f'{k}:{v:.2f}' for k,v in sorted(all_probs.items(), key=lambda x:-x[1])[:3]])}")
    
    print("\n" + "="*60)
    print(f"📊 ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
