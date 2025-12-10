"""
Text Preprocessing Module for Emotion Detection.
Handles text cleaning, tokenization, stopword removal, and lemmatization.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import List, Optional


# Download required NLTK data (run once)
def download_nltk_data():
    """Download required NLTK data packages."""
    packages = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {package}: {e}")


class TextPreprocessor:
    """
    Text preprocessing pipeline for emotion detection.
    
    Features:
    - Text cleaning (lowercase, remove URLs, mentions, special chars)
    - Tokenization
    - Stopword removal (optional)
    - Lemmatization (optional)
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove English stopwords
            lemmatize: Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Download NLTK data if needed
        download_nltk_data()
        
        # Initialize stopwords and lemmatizer
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of tokens
        """
        try:
            return word_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            return word_tokenize(text)
    
    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Filtered list without stopwords
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Apply lemmatization to tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text string
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        if not text:
            return ""
        
        # Step 2: Tokenize
        tokens = self.tokenize(text)
        
        # Step 3: Remove stopwords (optional)
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Step 4: Lemmatize (optional)
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Step 5: Join back to string
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Example usage
if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "I'm so happy today! 😊 #blessed http://example.com",
        "This is TERRIBLE!!! I hate everything @user123",
        "What a surprising news about the election results!!!",
        "I feel so sad and lonely right now...",
        "OMG I'm scared of what might happen next :(",
    ]
    
    print("=" * 60)
    print("TEXT PREPROCESSING TEST")
    print("=" * 60)
    
    for text in test_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
    
    print("\n" + "=" * 60)
    print("✓ Preprocessing test completed!")
    print("=" * 60)
