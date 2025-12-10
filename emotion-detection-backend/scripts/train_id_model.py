"""
Train Indonesian Emotion Detection Model.
Uses the same architecture as the English model but with Indonesian dataset.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.train import EmotionModelTrainer

# Paths
DATASET_DIR = r"d:\project-emotion-detected-system\dataset-bahasa-processed"
SAVE_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_id"

def main():
    print("="*60)
    print("🇮🇩 TRAINING INDONESIAN EMOTION DETECTION MODEL")
    print("="*60)
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize trainer
    trainer = EmotionModelTrainer(
        max_words=10000,
        max_sequence_length=100,
        embedding_dim=100,
        model_type='lstm'
    )
    
    # Load data
    print("\n📂 Loading Indonesian dataset...")
    train_path = os.path.join(DATASET_DIR, "train.txt")
    val_path = os.path.join(DATASET_DIR, "val.txt")
    test_path = os.path.join(DATASET_DIR, "test.txt")
    
    print(f"   Train: {train_path}")
    train_df = trainer.load_data(train_path)
    
    print(f"   Validation: {val_path}")
    val_df = trainer.load_data(val_path)
    
    print(f"   Test: {test_path}")
    test_df = trainer.load_data(test_path)
    
    # Print statistics
    print(f"\n📊 Data Statistics:")
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")
    print(f"   Test samples: {len(test_df)}")
    print(f"\n   Emotion distribution (train):")
    print(train_df['label'].value_counts().to_string().replace('\n', '\n   '))
    
    # Train model
    history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=50,
        batch_size=32,
        use_class_weights=True
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_df)
    print(f"\n📈 Test Results:")
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save model artifacts
    trainer.save_model(
        model_path=str(SAVE_DIR / "emotion_lstm_id.h5"),
        tokenizer_path=str(SAVE_DIR / "tokenizer.pkl"),
        label_encoder_path=str(SAVE_DIR / "label_encoder.pkl")
    )
    
    # Plot training history
    trainer.plot_history(history, save_path=str(SAVE_DIR / "training_history.png"))
    
    print("\n" + "="*60)
    print("✅ INDONESIA MODEL TRAINING COMPLETED!")
    print("="*60)
    print(f"\n📁 Model saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
