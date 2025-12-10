"""
IndoBERT Emotion Predictor Module.
Handles inference for Indonesian emotion detection using fine-tuned IndoBERT.
"""
import os
import re
import pickle
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class IndoBERTPredictor:
    """
    Singleton predictor for IndoBERT emotion detection.
    Thread-safe implementation for production use.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = None
        self.max_len = 128
        self._initialized = True
    
    def load_model(
        self,
        model_dir: Optional[str] = None
    ) -> bool:
        """
        Load IndoBERT model, tokenizer, and label encoder.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            True if loaded successfully
        """
        if model_dir is None:
            # Default path
            model_dir = Path(__file__).parent.parent.parent.parent / "saved_models_indobert" / "indobert_emotion"
        else:
            model_dir = Path(model_dir)
        
        try:
            print(f"📂 Loading IndoBERT from {model_dir}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir / "model"))
            
            # Load label encoder
            with open(str(model_dir / "label_encoder.pkl"), "rb") as f:
                self.label_encoder = pickle.load(f)
            
            # Set device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ IndoBERT loaded on {self.device}")
            print(f"   Classes: {list(self.label_encoder.classes_)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading IndoBERT: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
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
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict emotion for a single text.
        
        Args:
            text: Input text in Indonesian
            
        Returns:
            Dictionary with emotion, confidence, and probabilities
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess
        processed = self.preprocess_text(text)
        
        if not processed:
            return {
                "text": text,
                "emotion": "neutral",
                "confidence": 0.0,
                "probabilities": {}
            }
        
        # Tokenize
        encoding = self.tokenizer(
            processed,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        emotion = self.label_encoder.inverse_transform([pred_idx])[0]
        all_probs = {label: float(probs[0][i]) for i, label in enumerate(self.label_encoder.classes_)}
        
        return {
            "text": text,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": all_probs
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict(text) for text in texts]
    
    def get_emotions(self) -> List[str]:
        """Get list of supported emotions."""
        if self.label_encoder is None:
            return []
        return list(self.label_encoder.classes_)


# Singleton accessor
_indobert_predictor: Optional[IndoBERTPredictor] = None

def get_indobert_predictor() -> IndoBERTPredictor:
    """Get the singleton IndoBERT predictor instance."""
    global _indobert_predictor
    if _indobert_predictor is None:
        _indobert_predictor = IndoBERTPredictor()
    return _indobert_predictor
