"""
Model Optimization Module
- ONNX conversion for faster inference
- Quantization for reduced model size
- Caching for repeat predictions
"""
import os
import time
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

from cachetools import TTLCache, cached
import onnxruntime as ort
import numpy as np


class PredictionCache:
    """
    LRU + TTL cache for repeat predictions.
    Caches results for identical text inputs.
    """
    
    def __init__(self, maxsize: int = 1000, ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            maxsize: Maximum number of cached items
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self._cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0
    
    def _hash_text(self, text: str, lang: str = "en") -> str:
        """Generate cache key from text and language."""
        return hashlib.md5(f"{lang}:{text}".encode()).hexdigest()
    
    def get(self, text: str, lang: str = "en") -> Optional[Dict[str, Any]]:
        """Get cached prediction if exists."""
        key = self._hash_text(text, lang)
        result = self._cache.get(key)
        if result is not None:
            self._hits += 1
        else:
            self._misses += 1
        return result
    
    def set(self, text: str, result: Dict[str, Any], lang: str = "en"):
        """Cache a prediction result."""
        key = self._hash_text(text, lang)
        self._cache[key] = result
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "maxsize": self._cache.maxsize,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }


class ONNXOptimizedPredictor:
    """
    ONNX-optimized predictor for IndoBERT.
    Provides faster inference than PyTorch.
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, label_encoder_path: str):
        """
        Initialize ONNX predictor.
        
        Args:
            model_path: Path to ONNX model file
            tokenizer_path: Path to tokenizer
            label_encoder_path: Path to label encoder
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.label_encoder_path = Path(label_encoder_path)
        
        self.session = None
        self.tokenizer = None
        self.label_encoder = None
        self.cache = PredictionCache()
        self.max_len = 128
    
    def load(self):
        """Load ONNX model and tokenizer."""
        from transformers import AutoTokenizer
        
        # Load ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        
        # Load label encoder
        with open(str(self.label_encoder_path), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"✅ ONNX model loaded from {self.model_path}")
    
    def predict(self, text: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Predict emotion with optional caching.
        
        Args:
            text: Input text
            use_cache: Whether to use prediction cache
            
        Returns:
            Prediction result dictionary
        """
        # Check cache first
        if use_cache:
            cached_result = self.cache.get(text, "id")
            if cached_result is not None:
                cached_result["cached"] = True
                return cached_result
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # Run inference
        inputs = {
            'input_ids': encoding['input_ids'].astype(np.int64),
            'attention_mask': encoding['attention_mask'].astype(np.int64)
        }
        
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        
        # Process results
        probs = self._softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        emotion = self.label_encoder.inverse_transform([pred_idx])[0]
        
        result = {
            "text": text,
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": {
                label: float(probs[i]) 
                for i, label in enumerate(self.label_encoder.classes_)
            },
            "cached": False
        }
        
        # Cache result
        if use_cache:
            self.cache.set(text, result, "id")
        
        return result
    
    def _softmax(self, x):
        """Compute softmax values."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats


def convert_to_onnx(
    model_path: str,
    tokenizer_path: str,
    output_path: str,
    quantize: bool = True
):
    """
    Convert HuggingFace model to ONNX format.
    
    Args:
        model_path: Path to HuggingFace model
        tokenizer_path: Path to tokenizer
        output_path: Output path for ONNX model
        quantize: Whether to apply quantization
    """
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    
    print(f"📦 Converting model to ONNX...")
    print(f"   Source: {model_path}")
    print(f"   Output: {output_path}")
    
    # Load and convert
    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        export=True
    )
    
    # Save ONNX model
    model.save_pretrained(output_path)
    
    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)
    
    if quantize:
        print("⚡ Applying quantization...")
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        
        quantizer = ORTQuantizer.from_pretrained(output_path)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
        quantizer.quantize(save_dir=f"{output_path}_quantized", quantization_config=qconfig)
        print(f"✅ Quantized model saved to {output_path}_quantized")
    
    print("✅ ONNX conversion complete!")


def benchmark_models(pytorch_predictor, onnx_predictor, test_texts: list, iterations: int = 10):
    """
    Benchmark PyTorch vs ONNX model performance.
    
    Args:
        pytorch_predictor: PyTorch model predictor
        onnx_predictor: ONNX model predictor
        test_texts: List of test texts
        iterations: Number of iterations per text
        
    Returns:
        Benchmark results dictionary
    """
    results = {
        "pytorch": {"times": [], "avg_ms": 0},
        "onnx": {"times": [], "avg_ms": 0},
        "onnx_cached": {"times": [], "avg_ms": 0}
    }
    
    print(f"\n🏁 Benchmarking with {len(test_texts)} texts x {iterations} iterations...")
    
    # Warm up
    for text in test_texts[:2]:
        pytorch_predictor.predict(text)
        onnx_predictor.predict(text, use_cache=False)
    
    # Clear cache for fair test
    onnx_predictor.cache.clear()
    
    # Benchmark PyTorch
    print("\n📊 Testing PyTorch model...")
    for text in test_texts:
        for _ in range(iterations):
            start = time.perf_counter()
            pytorch_predictor.predict(text)
            elapsed = (time.perf_counter() - start) * 1000
            results["pytorch"]["times"].append(elapsed)
    
    # Benchmark ONNX (no cache)
    print("📊 Testing ONNX model (no cache)...")
    onnx_predictor.cache.clear()
    for text in test_texts:
        for _ in range(iterations):
            start = time.perf_counter()
            onnx_predictor.predict(text, use_cache=False)
            elapsed = (time.perf_counter() - start) * 1000
            results["onnx"]["times"].append(elapsed)
    
    # Benchmark ONNX (with cache)
    print("📊 Testing ONNX model (with cache)...")
    onnx_predictor.cache.clear()
    for text in test_texts:
        # First call populates cache
        onnx_predictor.predict(text, use_cache=True)
        for _ in range(iterations):
            start = time.perf_counter()
            onnx_predictor.predict(text, use_cache=True)
            elapsed = (time.perf_counter() - start) * 1000
            results["onnx_cached"]["times"].append(elapsed)
    
    # Calculate averages
    for key in results:
        times = results[key]["times"]
        results[key]["avg_ms"] = sum(times) / len(times) if times else 0
        results[key]["min_ms"] = min(times) if times else 0
        results[key]["max_ms"] = max(times) if times else 0
    
    # Print results
    print("\n" + "="*60)
    print("📈 BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Model':<20} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-"*60)
    for key, data in results.items():
        print(f"{key:<20} {data['avg_ms']:<12.2f} {data['min_ms']:<12.2f} {data['max_ms']:<12.2f}")
    
    # Speedup
    pytorch_avg = results["pytorch"]["avg_ms"]
    onnx_avg = results["onnx"]["avg_ms"]
    cached_avg = results["onnx_cached"]["avg_ms"]
    
    print("-"*60)
    print(f"ONNX speedup vs PyTorch: {pytorch_avg/onnx_avg:.2f}x")
    print(f"ONNX+Cache speedup: {pytorch_avg/cached_avg:.2f}x")
    print("="*60)
    
    return results


# Global cache instance for API use
_prediction_cache = PredictionCache(maxsize=2000, ttl=7200)

def get_prediction_cache() -> PredictionCache:
    """Get global prediction cache instance."""
    return _prediction_cache
