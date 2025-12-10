"""
Emotion Prediction Module.
Provides inference capabilities for the trained emotion detection model.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from threading import Lock

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Add parent path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.ml.preprocessing import TextPreprocessor
from app.config import settings


class EmotionPredictor:
    """
    Singleton class for emotion prediction.
    
    Loads the trained model, tokenizer, and label encoder once,
    then provides methods for single and batch predictions.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the predictor if not already done."""
        if self._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)
        self.max_sequence_length = settings.MAX_SEQUENCE_LENGTH
        
        self._initialized = True
    
    def load_model(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        label_encoder_path: Optional[str] = None
    ) -> bool:
        """
        Load the trained model and preprocessing objects.
        
        Args:
            model_path: Path to Keras model file
            tokenizer_path: Path to tokenizer pickle file
            label_encoder_path: Path to label encoder pickle file
            
        Returns:
            True if loading was successful, False otherwise
        """
        # Use default paths if not provided
        model_path = model_path or settings.MODEL_PATH
        tokenizer_path = tokenizer_path or settings.TOKENIZER_PATH
        label_encoder_path = label_encoder_path or settings.LABEL_ENCODER_PATH
        
        try:
            # Load Keras model
            print(f"📦 Loading model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path)
            
            # Load tokenizer
            print(f"📦 Loading tokenizer from {tokenizer_path}...")
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load label encoder
            print(f"📦 Loading label encoder from {label_encoder_path}...")
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            print("✅ Model loaded successfully!")
            print(f"   Emotions: {list(self.label_encoder.classes_)}")
            
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found - {e}")
            print("   Please train the model first using: python -m app.ml.train")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return (
            self.model is not None and
            self.tokenizer is not None and
            self.label_encoder is not None
        )
    
    def _prepare_text(self, text: str) -> np.ndarray:
        """
        Preprocess and convert text to model input format.
        
        Args:
            text: Raw input text
            
        Returns:
            Padded sequence array
        """
        # Preprocess
        cleaned_text = self.preprocessor.preprocess(text)
        
        # Convert to sequence
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        
        # Pad sequence
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return padded
    
    def predict(self, text: str) -> Dict:
        """
        Predict emotion for a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with:
                - emotion: Predicted emotion label
                - confidence: Prediction confidence (0-1)
                - probabilities: Dict of all emotions and their probabilities
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare input
        X = self._prepare_text(text)
        
        # Get predictions
        probabilities = self.model.predict(X, verbose=0)[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(probabilities[predicted_idx])
        
        # Build probability dictionary
        prob_dict = {
            emotion: float(prob)
            for emotion, prob in zip(self.label_encoder.classes_, probabilities)
        }
        
        return {
            'emotion': predicted_emotion,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of input text strings
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        # Prepare all inputs
        cleaned_texts = [self.preprocessor.preprocess(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        # Batch prediction
        all_probabilities = self.model.predict(X, verbose=0)
        
        # Process results
        results = []
        for probs in all_probabilities:
            predicted_idx = np.argmax(probs)
            predicted_emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            prob_dict = {
                emotion: float(prob)
                for emotion, prob in zip(self.label_encoder.classes_, probs)
            }
            
            results.append({
                'emotion': predicted_emotion,
                'confidence': float(probs[predicted_idx]),
                'probabilities': prob_dict
            })
        
        return results
    
    def get_emotions(self) -> List[str]:
        """Get list of all supported emotion labels."""
        if not self.is_loaded():
            return []
        return list(self.label_encoder.classes_)


# Singleton access function
def get_predictor() -> EmotionPredictor:
    """Get or create the EmotionPredictor singleton instance."""
    return EmotionPredictor()


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("🎭 EMOTION PREDICTION TEST")
    print("=" * 60)
    
    # Initialize predictor
    predictor = get_predictor()
    
    # Load model
    success = predictor.load_model()
    
    if not success:
        print("\n⚠️  Model not found. Please train the model first.")
        print("   Run: python -m app.ml.train")
        sys.exit(1)
    
    print(f"\n📋 Available emotions: {predictor.get_emotions()}")
    
    # Test predictions
    test_texts = [
        "I'm so happy and excited about this wonderful news!",
        "This is absolutely terrible, I hate everything about it.",
        "I'm feeling very sad and lonely today, nobody understands me.",
        "What just happened? I can't believe this is real!",
        "I'm scared about what might happen tomorrow.",
        "I love spending time with my family, they mean everything to me.",
    ]
    
    print("\n" + "-" * 60)
    print("PREDICTIONS:")
    print("-" * 60)
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\n📝 Text: \"{text[:60]}...\"" if len(text) > 60 else f"\n📝 Text: \"{text}\"")
        print(f"   🎭 Emotion: {result['emotion'].upper()}")
        print(f"   📊 Confidence: {result['confidence']:.2%}")
        print(f"   📈 Top 3 probabilities:")
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        for emotion, prob in sorted_probs:
            print(f"      - {emotion}: {prob:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION TEST COMPLETED!")
    print("=" * 60)
