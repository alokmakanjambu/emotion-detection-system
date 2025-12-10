"""
API Schemas - Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class EmotionRequest(BaseModel):
    """Request model for single text emotion prediction."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to analyze for emotion detection",
        json_schema_extra={"example": "I'm so happy and excited today!"}
    )


class EmotionBatchRequest(BaseModel):
    """Request model for batch emotion prediction."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of texts to analyze (max 100)",
        json_schema_extra={"example": ["I'm happy!", "This is sad.", "I'm angry!"]}
    )


class EmotionResponse(BaseModel):
    """Response model for emotion prediction."""
    text: str = Field(..., description="Original input text")
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution for all emotions"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "I'm so happy today!",
                "emotion": "joy",
                "confidence": 0.95,
                "probabilities": {
                    "joy": 0.95,
                    "sadness": 0.01,
                    "anger": 0.01,
                    "fear": 0.01,
                    "surprise": 0.01,
                    "love": 0.01
                }
            }
        }
    }


class EmotionBatchResponse(BaseModel):
    """Response model for batch emotion prediction."""
    results: List[EmotionResponse] = Field(..., description="List of prediction results")
    count: int = Field(..., description="Number of texts processed")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    emotions: List[str] = Field(..., description="List of supported emotions")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    detail: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")
