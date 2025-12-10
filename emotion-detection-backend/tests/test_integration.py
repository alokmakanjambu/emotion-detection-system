"""
Integration Tests for Emotion Detection API
Tests all API endpoints with real HTTP requests.
"""
import pytest
import httpx
import asyncio
from typing import Generator

# API Base URL
BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="module")
def client() -> Generator[httpx.Client, None, None]:
    """Create HTTP client for tests."""
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        yield client


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_english_health(self, client):
        """Test English model health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
    
    def test_indonesian_health(self, client):
        """Test Indonesian model health endpoint."""
        response = client.get("/api/v1/health/id")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"


class TestEnglishPrediction:
    """Tests for English prediction endpoints."""
    
    def test_predict_joy(self, client):
        """Test prediction for joy emotion."""
        response = client.post(
            "/api/v1/predict",
            json={"text": "I am so happy and excited today!"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "emotion" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["emotion"] in ["joy", "love", "surprise"]
    
    def test_predict_sadness(self, client):
        """Test prediction for sadness emotion."""
        response = client.post(
            "/api/v1/predict",
            json={"text": "I feel so sad and lonely today"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "sadness"
        assert data["confidence"] > 0.5
    
    def test_predict_anger(self, client):
        """Test prediction for anger emotion."""
        response = client.post(
            "/api/v1/predict",
            json={"text": "I am furious and hate this situation!"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] in ["anger", "sadness"]
    
    def test_predict_fear(self, client):
        """Test prediction for fear emotion."""
        response = client.post(
            "/api/v1/predict",
            json={"text": "I am scared and worried about tomorrow"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "fear"
    
    def test_predict_love(self, client):
        """Test prediction for love emotion."""
        response = client.post(
            "/api/v1/predict",
            json={"text": "I love my family so much"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] in ["love", "joy"]
    
    def test_predict_empty_text_error(self, client):
        """Test empty text returns validation error."""
        response = client.post(
            "/api/v1/predict",
            json={"text": ""}
        )
        assert response.status_code == 422
    
    def test_predict_missing_text_error(self, client):
        """Test missing text field returns error."""
        response = client.post(
            "/api/v1/predict",
            json={}
        )
        assert response.status_code == 422


class TestIndonesianPrediction:
    """Tests for Indonesian prediction endpoints."""
    
    def test_predict_id_joy(self, client):
        """Test Indonesian joy prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Saya sangat senang dan bahagia hari ini!"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "emotion" in data
        assert data["emotion"] == "joy"
        assert data["confidence"] > 0.5
    
    def test_predict_id_sadness(self, client):
        """Test Indonesian sadness prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Aku sedih banget hari ini"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "sadness"
    
    def test_predict_id_anger(self, client):
        """Test Indonesian anger prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Marah banget sama kamu!"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "anger"
    
    def test_predict_id_fear(self, client):
        """Test Indonesian fear prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Takut banget dengan keadaan ini"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "fear"
    
    def test_predict_id_love(self, client):
        """Test Indonesian love prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Aku cinta kamu selamanya"}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["emotion"] == "love"
    
    def test_predict_id_neutral(self, client):
        """Test Indonesian neutral prediction."""
        response = client.post(
            "/api/v1/predict/id",
            json={"text": "Hari ini cuaca biasa saja"}
        )
        assert response.status_code == 200
        
        data = response.json()
        # Neutral is harder to predict, accept various results
        assert data["emotion"] in ["neutral", "joy", "sadness"]


class TestBatchPrediction:
    """Tests for batch prediction endpoints."""
    
    def test_batch_predict_english(self, client):
        """Test English batch prediction."""
        response = client.post(
            "/api/v1/predict/batch",
            json={
                "texts": [
                    "I am happy",
                    "I am sad",
                    "I am angry"
                ]
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "count" in data
        assert data["count"] == 3
        assert len(data["results"]) == 3
    
    def test_batch_predict_indonesian(self, client):
        """Test Indonesian batch prediction."""
        response = client.post(
            "/api/v1/predict/id/batch",
            json={
                "texts": [
                    "Saya senang",
                    "Saya sedih",
                    "Saya marah"
                ]
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["count"] == 3
    
    def test_batch_predict_empty_list(self, client):
        """Test empty batch returns error."""
        response = client.post(
            "/api/v1/predict/batch",
            json={"texts": []}
        )
        assert response.status_code == 422


class TestEmotionsEndpoint:
    """Tests for emotions list endpoints."""
    
    def test_get_emotions_english(self, client):
        """Test get English emotions list."""
        response = client.get("/api/v1/emotions")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 6
    
    def test_get_emotions_indonesian(self, client):
        """Test get Indonesian emotions list."""
        response = client.get("/api/v1/emotions/id")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert "joy" in data
        assert "sadness" in data


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""
    
    def test_swagger_docs(self, client):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self, client):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        data = response.json()
        assert "openapi" in data
        assert "paths" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
