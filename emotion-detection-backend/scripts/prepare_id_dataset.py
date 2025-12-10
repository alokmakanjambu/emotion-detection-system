"""
Prepare Indonesian dataset for emotion detection training.
Combines multiple CSV files into train/val/test splits.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset paths
DATASET_DIR = r"d:\project-emotion-detected-system\dataset-bahasa"
OUTPUT_DIR = r"d:\project-emotion-detected-system\dataset-bahasa-processed"

# Label mapping (normalize labels to lowercase)
LABEL_MAP = {
    'Anger': 'anger',
    'Fear': 'fear', 
    'Joy': 'joy',
    'Love': 'love',
    'Sad': 'sadness',
    'Neutral': 'neutral'
}

def load_and_combine_data():
    """Load all CSV files and combine into single dataframe."""
    all_data = []
    
    files = {
        'AngerData.csv': 'anger',
        'FearData.csv': 'fear',
        'JoyData.csv': 'joy',
        'LoveData.csv': 'love',
        'SadData.csv': 'sadness',
        'NeutralData.csv': 'neutral'
    }
    
    for filename, label in files.items():
        filepath = os.path.join(DATASET_DIR, filename)
        if os.path.exists(filepath):
            # Read TSV (tab-separated)
            df = pd.read_csv(filepath, sep='\t', header=0, names=['text', 'original_label'])
            df['label'] = label
            all_data.append(df[['text', 'label']])
            print(f"✅ Loaded {filename}: {len(df)} samples")
        else:
            print(f"⚠️ File not found: {filename}")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total samples: {len(combined)}")
    return combined

def clean_text(text):
    """Basic text cleaning."""
    if pd.isna(text):
        return ""
    return str(text).strip()

def create_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/val/test splits."""
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: first train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_ratio, random_state=42, stratify=df['label']
    )
    
    # Split: train vs val
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=42, stratify=train_val['label']
    )
    
    return train, val, test

def save_splits(train, val, test, output_dir):
    """Save splits in text;label format."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in [('train', train), ('val', val), ('test', test)]:
        filepath = os.path.join(output_dir, f'{name}.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Format: text;label (semicolon separated like English dataset)
                text = row['text'].replace(';', ',')  # Escape semicolons in text
                f.write(f"{text};{row['label']}\n")
        print(f"✅ Saved {name}.txt: {len(df)} samples")

def print_stats(train, val, test):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("📊 DATASET STATISTICS")
    print("="*50)
    
    for name, df in [('Train', train), ('Val', val), ('Test', test)]:
        print(f"\n{name} ({len(df)} samples):")
        print(df['label'].value_counts().to_string())

def main():
    print("="*50)
    print("🇮🇩 INDONESIAN EMOTION DATASET PREPARATION")
    print("="*50)
    
    # Load and combine
    print("\n📂 Loading data...")
    df = load_and_combine_data()
    
    # Create splits
    print("\n✂️ Creating train/val/test splits...")
    train, val, test = create_splits(df)
    
    # Print stats
    print_stats(train, val, test)
    
    # Save
    print(f"\n💾 Saving to {OUTPUT_DIR}...")
    save_splits(train, val, test, OUTPUT_DIR)
    
    print("\n" + "="*50)
    print("✅ PREPARATION COMPLETE!")
    print("="*50)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_DIR}/train.txt")
    print(f"  - {OUTPUT_DIR}/val.txt")
    print(f"  - {OUTPUT_DIR}/test.txt")

if __name__ == "__main__":
    main()
