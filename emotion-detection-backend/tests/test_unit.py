"""
Unit Tests for Emotion Detection System
Tests preprocessing, model loading, and prediction functions.
"""
import pytest
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class TestPreprocessing:
    """Tests for text preprocessing module."""
    
    def test_preprocessing_import(self):
        """Test preprocessing module can be imported."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        assert preprocessor is not None
    
    def test_clean_text_lowercase(self):
        """Test text is converted to lowercase."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("HELLO WORLD")
        assert result == result.lower()
    
    def test_clean_text_removes_urls(self):
        """Test URLs are removed from text."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Check this https://example.com link")
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_mentions(self):
        """Test @mentions are removed."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("Hello @username how are you")
        assert "@username" not in result
    
    def test_clean_text_removes_hashtags(self):
        """Test hashtags are processed."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_text("I love #python programming")
        assert "#" not in result
    
    def test_preprocess_empty_string(self):
        """Test preprocessing handles empty string."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("")
        assert result == ""
    
    def test_preprocess_returns_string(self):
        """Test preprocessing returns string type."""
        from app.ml.preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        result = preprocessor.preprocess("Hello world")
        assert isinstance(result, str)


class TestPredictionCache:
    """Tests for prediction caching module."""
    
    def test_cache_import(self):
        """Test cache module can be imported."""
        from app.ml.optimization import PredictionCache
        cache = PredictionCache()
        assert cache is not None
    
    def test_cache_set_and_get(self):
        """Test cache stores and retrieves values."""
        from app.ml.optimization import PredictionCache
        cache = PredictionCache()
        
        test_result = {"emotion": "joy", "confidence": 0.95}
        cache.set("test text", test_result, "en")
        
        retrieved = cache.get("test text", "en")
        assert retrieved is not None
        assert retrieved["emotion"] == "joy"
    
    def test_cache_miss_returns_none(self):
        """Test cache returns None for missing keys."""
        from app.ml.optimization import PredictionCache
        cache = PredictionCache()
        
        result = cache.get("nonexistent text", "en")
        assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        from app.ml.optimization import PredictionCache
        cache = PredictionCache()
        
        cache.set("text1", {"emotion": "joy"}, "en")
        cache.get("text1", "en")  # hit
        cache.get("text2", "en")  # miss
        
        stats = cache.stats
        assert "hits" in stats
        assert "misses" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1
    
    def test_cache_clear(self):
        """Test cache clearing."""
        from app.ml.optimization import PredictionCache
        cache = PredictionCache()
        
        cache.set("text1", {"emotion": "joy"}, "en")
        cache.clear()
        
        result = cache.get("text1", "en")
        assert result is None


class TestEnglishModel:
    """Tests for English LSTM model."""
    
    def test_predictor_import(self):
        """Test predictor can be imported."""
        from app.ml.model import get_predictor
        predictor = get_predictor()
        assert predictor is not None
    
    def test_predictor_singleton(self):
        """Test predictor is singleton."""
        from app.ml.model import get_predictor
        p1 = get_predictor()
        p2 = get_predictor()
        assert p1 is p2
    
    def test_predictor_load_model(self):
        """Test model can be loaded."""
        from app.ml.model import get_predictor
        predictor = get_predictor()
        
        # Skip if model files don't exist
        if not predictor.is_loaded():
            try:
                predictor.load_model()
            except Exception:
                pytest.skip("Model files not available")
        
        assert predictor.is_loaded()
    
    def test_predictor_predict_returns_dict(self):
        """Test prediction returns dictionary."""
        from app.ml.model import get_predictor
        predictor = get_predictor()
        
        if not predictor.is_loaded():
            try:
                predictor.load_model()
            except Exception:
                pytest.skip("Model files not available")
        
        result = predictor.predict("I am happy")
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
    
    def test_predictor_emotions_list(self):
        """Test get_emotions returns list."""
        from app.ml.model import get_predictor
        predictor = get_predictor()
        
        if not predictor.is_loaded():
            try:
                predictor.load_model()
            except Exception:
                pytest.skip("Model files not available")
        
        emotions = predictor.get_emotions()
        assert isinstance(emotions, list)
        assert len(emotions) > 0


class TestIndonesianModel:
    """Tests for Indonesian IndoBERT model."""
    
    def test_indobert_import(self):
        """Test IndoBERT predictor can be imported."""
        from app.ml.model_indobert import get_indobert_predictor
        predictor = get_indobert_predictor()
        assert predictor is not None
    
    def test_indobert_singleton(self):
        """Test IndoBERT predictor is singleton."""
        from app.ml.model_indobert import get_indobert_predictor
        p1 = get_indobert_predictor()
        p2 = get_indobert_predictor()
        assert p1 is p2
    
    def test_indobert_load_model(self):
        """Test IndoBERT model can be loaded."""
        from app.ml.model_indobert import get_indobert_predictor
        predictor = get_indobert_predictor()
        
        if not predictor.is_loaded():
            try:
                predictor.load_model()
            except Exception:
                pytest.skip("IndoBERT model files not available")
        
        assert predictor.is_loaded()
    
    def test_indobert_predict_returns_dict(self):
        """Test IndoBERT prediction returns dictionary."""
        from app.ml.model_indobert import get_indobert_predictor
        predictor = get_indobert_predictor()
        
        if not predictor.is_loaded():
            try:
                predictor.load_model()
            except Exception:
                pytest.skip("IndoBERT model files not available")
        
        result = predictor.predict("Saya senang sekali")
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert "probabilities" in result
    
    def test_indobert_preprocess_text(self):
        """Test IndoBERT text preprocessing."""
        from app.ml.model_indobert import get_indobert_predictor
        predictor = get_indobert_predictor()
        
        text = "HELLO @user https://example.com #hashtag"
        processed = predictor.preprocess_text(text)
        
        assert processed == processed.lower()
        assert "@" not in processed
        assert "https" not in processed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
