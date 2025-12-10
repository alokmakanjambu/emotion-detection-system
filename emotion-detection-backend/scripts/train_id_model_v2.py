"""
Train Indonesian Emotion Model with Indonesian-specific preprocessing.
Uses Sastrawi stemmer and Indonesian stopwords.
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.ml.preprocessing_id import IndonesianTextPreprocessor

# Paths
DATASET_DIR = r"d:\project-emotion-detected-system\dataset-bahasa-processed"
SAVE_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_id_v2"

# Hyperparameters
MAX_WORDS = 15000  # Increased for Indonesian vocabulary
MAX_LEN = 100
EMBEDDING_DIM = 128  # Increased
BATCH_SIZE = 32
EPOCHS = 100  # More epochs, will use early stopping

def load_data(filepath, preprocessor):
    """Load and preprocess data."""
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'label'])
    df = df.dropna()
    
    print(f"  Preprocessing {len(df)} texts...")
    df['text'] = df['text'].apply(preprocessor.preprocess)
    df = df[df['text'].str.len() > 0]
    
    return df

def build_model(vocab_size, num_classes):
    """Build improved LSTM model."""
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.3),
        
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        
        Bidirectional(LSTM(64)),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("="*70)
    print("🇮🇩 TRAINING INDONESIAN MODEL v2 (with ID Preprocessing)")
    print("="*70)
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Initialize Indonesian preprocessor
    print("\n📝 Initializing Indonesian preprocessor...")
    preprocessor = IndonesianTextPreprocessor(
        remove_stopwords=False,  # Keep stopwords, let model learn
        use_stemming=True        # Use Sastrawi stemming
    )
    
    # Load data
    print("\n📂 Loading dataset...")
    train_df = load_data(os.path.join(DATASET_DIR, "train.txt"), preprocessor)
    val_df = load_data(os.path.join(DATASET_DIR, "val.txt"), preprocessor)
    test_df = load_data(os.path.join(DATASET_DIR, "test.txt"), preprocessor)
    
    print(f"\n📊 Data Statistics:")
    print(f"   Train: {len(train_df)}")
    print(f"   Val: {len(val_df)}")
    print(f"   Test: {len(test_df)}")
    print(f"\n   Distribution:\n{train_df['label'].value_counts().to_string()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df['label'])
    y_val = label_encoder.transform(val_df['label'])
    y_test = label_encoder.transform(test_df['label'])
    
    num_classes = len(label_encoder.classes_)
    print(f"\n   Classes ({num_classes}): {list(label_encoder.classes_)}")
    
    # Tokenize
    print("\n📝 Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['text'])
    
    X_train = pad_sequences(tokenizer.texts_to_sequences(train_df['text']), maxlen=MAX_LEN)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_df['text']), maxlen=MAX_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_df['text']), maxlen=MAX_LEN)
    
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS)
    print(f"   Vocabulary size: {vocab_size}")
    
    # Compute class weights
    class_weights = dict(zip(
        np.unique(y_train),
        compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    ))
    print(f"\n⚖️ Class weights: {class_weights}")
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_model(vocab_size, num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    ]
    
    # Train
    print("\n🚀 Training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluating...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {acc*100:.2f}%")
    print(f"   Test Loss: {loss:.4f}")
    
    # Save
    print("\n💾 Saving artifacts...")
    model.save(str(SAVE_DIR / "emotion_lstm_id_v2.h5"))
    
    with open(str(SAVE_DIR / "tokenizer.pkl"), 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open(str(SAVE_DIR / "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save preprocessor config
    with open(str(SAVE_DIR / "preprocessor_config.pkl"), 'wb') as f:
        pickle.dump({'remove_stopwords': False, 'use_stemming': True}, f)
    
    # Plot history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(SAVE_DIR / "training_history.png"), dpi=300)
    plt.close()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📁 Saved to: {SAVE_DIR}")
    print(f"   Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
