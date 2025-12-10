"""
Emotion Detection Model Training Script.
Trains a Bidirectional LSTM/GRU model for text emotion classification.
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, GRU, Dense, Dropout,
    Bidirectional, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from app.ml.preprocessing import TextPreprocessor


class EmotionModelTrainer:
    """
    Trainer class for emotion detection LSTM/GRU model.
    
    Features:
    - Bidirectional LSTM or GRU architecture
    - Word embeddings
    - Dropout for regularization
    - Early stopping and learning rate scheduling
    - Class weight balancing for imbalanced data
    """
    
    def __init__(
        self,
        max_words: int = 10000,
        max_sequence_length: int = 100,
        embedding_dim: int = 100,
        model_type: str = 'lstm'  # 'lstm' or 'gru'
    ):
        """
        Initialize the trainer.
        
        Args:
            max_words: Maximum vocabulary size
            max_sequence_length: Maximum tokens per text
            embedding_dim: Dimension of word embeddings
            model_type: Type of RNN ('lstm' or 'gru')
        """
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.model_type = model_type.lower()
        
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.preprocessor = TextPreprocessor(remove_stopwords=False, lemmatize=True)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from semicolon-separated text file.
        
        Args:
            filepath: Path to data file
            
        Returns:
            DataFrame with 'text' and 'label' columns
        """
        # Read semicolon-separated file
        df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'label'])
        
        # Clean any NaN values
        df = df.dropna()
        
        # Preprocess text
        print(f"  Preprocessing {len(df)} texts...")
        df['text'] = df['text'].apply(self.preprocessor.preprocess)
        
        # Remove empty texts after preprocessing
        df = df[df['text'].str.len() > 0]
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = True):
        """
        Prepare and encode data for training.
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            fit: Whether to fit tokenizer and label encoder (True for training data)
            
        Returns:
            Tuple of (X_padded, y_encoded)
        """
        if fit:
            # Fit label encoder
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(df['label'])
            
            # Fit tokenizer
            self.tokenizer = Tokenizer(
                num_words=self.max_words,
                oov_token='<OOV>'
            )
            self.tokenizer.fit_on_texts(df['text'])
        else:
            # Transform using fitted encoders
            y = self.label_encoder.transform(df['label'])
        
        # Convert texts to sequences
        sequences = self.tokenizer.texts_to_sequences(df['text'])
        
        # Pad sequences
        X = pad_sequences(
            sequences,
            maxlen=self.max_sequence_length,
            padding='post',
            truncating='post'
        )
        
        return X, y
    
    def build_model(self, num_classes: int) -> Sequential:
        """
        Build the Bidirectional LSTM/GRU model.
        
        Args:
            num_classes: Number of emotion classes
            
        Returns:
            Compiled Keras model
        """
        # Select RNN layer type
        RNNLayer = LSTM if self.model_type == 'lstm' else GRU
        
        model = Sequential([
            # Embedding layer
            Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length
            ),
            SpatialDropout1D(0.2),
            
            # First Bidirectional RNN layer (with return_sequences=True)
            Bidirectional(RNNLayer(128, return_sequences=True)),
            Dropout(0.3),
            
            # Second Bidirectional RNN layer
            Bidirectional(RNNLayer(64)),
            Dropout(0.3),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def compute_class_weights(self, y: np.ndarray) -> dict:
        """
        Compute class weights for imbalanced data.
        
        Args:
            y: Label array
            
        Returns:
            Dictionary of class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 32,
        use_class_weights: bool = True
    ):
        """
        Train the emotion detection model.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            epochs: Maximum training epochs
            batch_size: Training batch size
            use_class_weights: Whether to use class weights for imbalanced data
            
        Returns:
            Training history
        """
        # Prepare data
        print("\n📊 Preparing training data...")
        X_train, y_train = self.prepare_data(train_df, fit=True)
        
        print("📊 Preparing validation data...")
        X_val, y_val = self.prepare_data(val_df, fit=False)
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        print(f"\n🏗️  Building {self.model_type.upper()} model...")
        print(f"   Number of classes: {num_classes}")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        print(f"   Vocabulary size: {min(len(self.tokenizer.word_index) + 1, self.max_words)}")
        
        self.model = self.build_model(num_classes)
        
        # Print model summary
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        # Compute class weights
        class_weights = None
        if use_class_weights:
            class_weights = self.compute_class_weights(y_train)
            print(f"\n⚖️  Class weights: {class_weights}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\n🚀 Starting training...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print("-" * 60)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate model on test data.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n📈 Evaluating on test data...")
        X_test, y_test = self.prepare_data(test_df, fit=False)
        
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def save_model(self, model_path: str, tokenizer_path: str, label_encoder_path: str):
        """
        Save model and preprocessing objects.
        
        Args:
            model_path: Path to save Keras model
            tokenizer_path: Path to save tokenizer
            label_encoder_path: Path to save label encoder
        """
        # Create directories if needed
        for path in [model_path, tokenizer_path, label_encoder_path]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(model_path)
        print(f"✅ Model saved to {model_path}")
        
        # Save tokenizer
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        print(f"✅ Tokenizer saved to {tokenizer_path}")
        
        # Save label encoder
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Label encoder saved to {label_encoder_path}")
    
    def plot_history(self, history, save_path: str = 'training_history.png'):
        """
        Plot training history.
        
        Args:
            history: Keras training history object
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to {save_path}")
        plt.close()


# ========== MAIN TRAINING SCRIPT ==========
def main():
    """Main training function."""
    print("=" * 60)
    print("🎭 EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Paths - adjust based on where script is run from
    base_path = Path(__file__).parent.parent.parent.parent  # Go to project root
    archive_path = base_path / "archive"
    
    train_path = archive_path / "train.txt"
    val_path = archive_path / "val.txt"
    test_path = archive_path / "test.txt"
    
    # Check if data exists
    if not train_path.exists():
        print(f"❌ Training data not found at {train_path}")
        print("   Please ensure the dataset is in the 'archive' folder.")
        return
    
    # Initialize trainer
    trainer = EmotionModelTrainer(
        max_words=10000,
        max_sequence_length=100,
        embedding_dim=100,
        model_type='lstm'  # Change to 'gru' for GRU model
    )
    
    # Load data
    print("\n📂 Loading data...")
    print(f"   Train: {train_path}")
    train_df = trainer.load_data(str(train_path))
    
    print(f"   Validation: {val_path}")
    val_df = trainer.load_data(str(val_path))
    
    print(f"   Test: {test_path}")
    test_df = trainer.load_data(str(test_path))
    
    # Print data statistics
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
    save_dir = Path(__file__).parent / "saved_models"
    trainer.save_model(
        model_path=str(save_dir / "emotion_lstm.h5"),
        tokenizer_path=str(save_dir / "tokenizer.pkl"),
        label_encoder_path=str(save_dir / "label_encoder.pkl")
    )
    
    # Plot training history
    trainer.plot_history(history, save_path=str(save_dir / "training_history.png"))
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 Model artifacts saved to: {save_dir}")
    print("   - emotion_lstm.h5 (trained model)")
    print("   - tokenizer.pkl (text tokenizer)")
    print("   - label_encoder.pkl (label encoder)")
    print("   - training_history.png (training curves)")


if __name__ == "__main__":
    main()
