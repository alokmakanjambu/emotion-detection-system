"""
API Routes - Emotion detection endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from typing import List

from app.api.schemas import (
    EmotionRequest,
    EmotionBatchRequest,
    EmotionResponse,
    EmotionBatchResponse,
    HealthResponse,
    ErrorResponse
)
from app.ml.model import get_predictor
from app.config import settings


# Create router
router = APIRouter(prefix="/api/v1", tags=["Emotion Detection"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check API status and model availability"
)
async def health_check():
    """
    Health check endpoint.
    Returns API status and whether the model is loaded.
    """
    predictor = get_predictor()
    
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded(),
        emotions=predictor.get_emotions() if predictor.is_loaded() else [],
        version=settings.API_VERSION
    )


@router.post(
    "/predict",
    response_model=EmotionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary="Predict Emotion",
    description="Analyze a single text and predict its emotion"
)
async def predict_emotion(request: EmotionRequest):
    """
    Predict emotion for a single text.
    
    Returns the predicted emotion, confidence score, and probability distribution.
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        result = predictor.predict(request.text)
        
        return EmotionResponse(
            text=request.text,
            emotion=result['emotion'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=EmotionBatchResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary="Batch Predict Emotions",
    description="Analyze multiple texts and predict their emotions"
)
async def predict_emotions_batch(request: EmotionBatchRequest):
    """
    Predict emotions for multiple texts.
    
    Accepts up to 100 texts and returns predictions for each.
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        results = predictor.predict_batch(request.texts)
        
        responses = [
            EmotionResponse(
                text=text,
                emotion=result['emotion'],
                confidence=result['confidence'],
                probabilities=result['probabilities']
            )
            for text, result in zip(request.texts, results)
        ]
        
        return EmotionBatchResponse(
            results=responses,
            count=len(responses)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/emotions",
    response_model=List[str],
    summary="Get Supported Emotions",
    description="Get list of all supported emotion labels"
)
async def get_emotions():
    """
    Get list of all emotions the model can detect.
    """
    predictor = get_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    return predictor.get_emotions()


# ========== INDONESIAN INDOBERT ENDPOINTS ==========

from app.ml.model_indobert import get_indobert_predictor


@router.get(
    "/health/id",
    response_model=HealthResponse,
    summary="Health Check (Indonesian)",
    description="Check Indonesian model status and availability",
    tags=["Indonesian Emotion Detection"]
)
async def health_check_id():
    """
    Health check for Indonesian IndoBERT model.
    """
    predictor = get_indobert_predictor()
    
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded(),
        emotions=predictor.get_emotions() if predictor.is_loaded() else [],
        version=settings.API_VERSION
    )


@router.post(
    "/predict/id",
    response_model=EmotionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary="Predict Emotion (Indonesian)",
    description="Analyze Indonesian text and predict its emotion using IndoBERT",
    tags=["Indonesian Emotion Detection"]
)
async def predict_emotion_id(request: EmotionRequest):
    """
    Predict emotion for Indonesian text using IndoBERT.
    
    Returns the predicted emotion, confidence score, and probability distribution.
    Supports 6 emotions: joy, sadness, anger, fear, love, neutral.
    """
    predictor = get_indobert_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indonesian model not loaded. Please try again later."
        )
    
    try:
        result = predictor.predict(request.text)
        
        return EmotionResponse(
            text=request.text,
            emotion=result['emotion'],
            confidence=result['confidence'],
            probabilities=result['probabilities']
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/id/batch",
    response_model=EmotionBatchResponse,
    responses={
        200: {"description": "Successful batch prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        503: {"model": ErrorResponse, "description": "Model not loaded"}
    },
    summary="Batch Predict Emotions (Indonesian)",
    description="Analyze multiple Indonesian texts using IndoBERT",
    tags=["Indonesian Emotion Detection"]
)
async def predict_emotions_batch_id(request: EmotionBatchRequest):
    """
    Predict emotions for multiple Indonesian texts using IndoBERT.
    """
    predictor = get_indobert_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indonesian model not loaded. Please try again later."
        )
    
    try:
        results = predictor.predict_batch(request.texts)
        
        responses = [
            EmotionResponse(
                text=text,
                emotion=result['emotion'],
                confidence=result['confidence'],
                probabilities=result['probabilities']
            )
            for text, result in zip(request.texts, results)
        ]
        
        return EmotionBatchResponse(
            results=responses,
            count=len(responses)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get(
    "/emotions/id",
    response_model=List[str],
    summary="Get Supported Emotions (Indonesian)",
    description="Get list of emotions for Indonesian model",
    tags=["Indonesian Emotion Detection"]
)
async def get_emotions_id():
    """
    Get list of all emotions the Indonesian model can detect.
    """
    predictor = get_indobert_predictor()
    
    if not predictor.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indonesian model not loaded. Please try again later."
        )
    
    return predictor.get_emotions()

