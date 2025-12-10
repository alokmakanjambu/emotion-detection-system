"""
FastAPI Main Application - Emotion Detection API.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.api.routes import router
from app.ml.model import get_predictor
from app.ml.model_indobert import get_indobert_predictor
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Loads both English and Indonesian ML models on startup.
    """
    # Startup: Load models
    print("🚀 Starting Emotion Detection API...")
    
    # Load English LSTM model
    predictor = get_predictor()
    try:
        predictor.load_model()
        print("✅ English LSTM model loaded!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load English model - {e}")
    
    # Load Indonesian IndoBERT model
    indobert = get_indobert_predictor()
    try:
        indobert.load_model()
        print("✅ Indonesian IndoBERT model loaded!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load Indonesian model - {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="""
    ## Emotion Detection API
    
    Analyze text and detect emotions using deep learning.
    
    ### 🌐 Supported Languages
    - **English** - LSTM model (`/api/v1/predict`)
    - **Indonesian** - IndoBERT model (`/api/v1/predict/id`) 🇮🇩
    
    ### Supported Emotions
    - **Joy** - Happiness, excitement
    - **Sadness** - Sorrow, grief
    - **Anger** - Frustration, irritation
    - **Fear** - Anxiety, worry
    - **Love** - Affection, care
    - **Neutral** - No strong emotion (Indonesian only)
    
    ### Features
    - Single text prediction
    - Batch prediction (up to 100 texts)
    - Confidence scores and probability distributions
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - redirects to documentation."""
    return {
        "message": "Emotion Detection API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
