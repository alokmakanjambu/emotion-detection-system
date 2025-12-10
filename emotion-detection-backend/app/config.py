"""
Configuration settings for the Emotion Detection ML System.
Uses pydantic-settings for environment variable management.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Model paths
    MODEL_PATH: str = "app/ml/saved_models/emotion_lstm.h5"
    TOKENIZER_PATH: str = "app/ml/saved_models/tokenizer.pkl"
    LABEL_ENCODER_PATH: str = "app/ml/saved_models/label_encoder.pkl"
    
    # Model hyperparameters
    MAX_SEQUENCE_LENGTH: int = 100
    MAX_WORDS: int = 10000
    EMBEDDING_DIM: int = 100
    
    # Training settings
    BATCH_SIZE: int = 32
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    
    # API settings
    API_TITLE: str = "Emotion Detection API"
    API_VERSION: str = "1.0.0"
    
    # Data paths
    DATA_DIR: str = "data"
    TRAIN_DATA: str = "data/train.txt"
    VAL_DATA: str = "data/val.txt"
    TEST_DATA: str = "data/test.txt"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Global settings instance
settings = Settings()
