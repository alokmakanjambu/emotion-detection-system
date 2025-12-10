"""
Indonesian Emotion Detection using IndoBERT.
Fine-tune IndoBERT for emotion classification.
"""
import os
# Disable TensorFlow to avoid conflicts
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['USE_TF'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import pandas as pd
import pickle
import re
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

# Paths
DATASET_DIR = r"d:\project-emotion-detected-system\dataset-bahasa-processed"
SAVE_DIR = Path(__file__).parent.parent / "app" / "ml" / "saved_models_indobert"

# Model config
MODEL_NAME = "indobenchmark/indobert-base-p1"  # IndoBERT pre-trained
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

def preprocess_text(text):
    """Simple text cleaning."""
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data(filepath):
    """Load and preprocess data."""
    df = pd.read_csv(filepath, sep=';', header=None, names=['text', 'label'])
    df = df.dropna()
    df['text'] = df['text'].apply(preprocess_text)
    df = df[df['text'].str.len() > 0]
    return df

class EmotionDataset(Dataset):
    """Custom dataset for emotion classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def main():
    print("="*70)
    print("🇮🇩 TRAINING INDOBERT EMOTION MODEL")
    print("="*70)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    print("\n📂 Loading dataset...")
    train_df = load_data(os.path.join(DATASET_DIR, "train.txt"))
    val_df = load_data(os.path.join(DATASET_DIR, "val.txt"))
    test_df = load_data(os.path.join(DATASET_DIR, "test.txt"))
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['label'])
    val_labels = label_encoder.transform(val_df['label'])
    test_labels = label_encoder.transform(test_df['label'])
    
    num_classes = len(label_encoder.classes_)
    print(f"   Classes ({num_classes}): {list(label_encoder.classes_)}")
    
    # Save label encoder
    with open(str(SAVE_DIR / "label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Load tokenizer
    print(f"\n📝 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create datasets
    train_dataset = EmotionDataset(
        train_df['text'].values, train_labels, tokenizer, MAX_LEN
    )
    val_dataset = EmotionDataset(
        val_df['text'].values, val_labels, tokenizer, MAX_LEN
    )
    test_dataset = EmotionDataset(
        test_df['text'].values, test_labels, tokenizer, MAX_LEN
    )
    
    # Load model
    print(f"\n🏗️ Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(SAVE_DIR / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(SAVE_DIR / "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to="none",
        learning_rate=LEARNING_RATE,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print("\n🚀 Training...")
    trainer.train()
    
    # Evaluate on test
    print("\n📈 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"   Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"   Test F1: {test_results['eval_f1']*100:.2f}%")
    
    # Save model
    print("\n💾 Saving model...")
    model.save_pretrained(str(SAVE_DIR / "model"))
    tokenizer.save_pretrained(str(SAVE_DIR / "tokenizer"))
    
    print("\n" + "="*70)
    print(f"✅ TRAINING COMPLETE!")
    print(f"   Test Accuracy: {test_results['eval_accuracy']*100:.2f}%")
    print(f"   Saved to: {SAVE_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
