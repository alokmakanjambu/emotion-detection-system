"""
Indonesian Text Preprocessing Module.
Uses Sastrawi for Indonesian stemming and custom stopwords.
"""
import re
from typing import List, Optional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class IndonesianTextPreprocessor:
    """
    Indonesian-specific text preprocessing pipeline.
    
    Features:
    - Text cleaning (URLs, mentions, hashtags, special chars)
    - Indonesian stopword removal (optional)
    - Sastrawi Indonesian stemming (optional)
    """
    
    # Common Indonesian stopwords (extended)
    STOPWORDS_ID = {
        'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'dengan', 'untuk',
        'pada', 'adalah', 'juga', 'tidak', 'akan', 'ada', 'saya', 'aku',
        'kamu', 'dia', 'kami', 'kita', 'mereka', 'atau', 'tetapi', 'tapi',
        'karena', 'jika', 'maka', 'saat', 'saja', 'hanya', 'bisa', 'dapat',
        'sudah', 'udah', 'belum', 'masih', 'lagi', 'nya', 'dalam', 'oleh',
        'seperti', 'bahwa', 'hal', 'sama', 'jadi', 'sih', 'dong', 'deh',
        'nih', 'gitu', 'gini', 'yg', 'dgn', 'utk', 'krn', 'tp', 'jg', 'sm',
        'aja', 'bgt', 'bngtt', 'bener', 'banget', 'sekali', 'sangat', 'amat',
        'para', 'lalu', 'kemudian', 'sebelum', 'sesudah', 'apabila', 'bila',
        # Keep emotion words - don't remove these!
    }
    
    def __init__(
        self,
        remove_stopwords: bool = True,
        use_stemming: bool = True,
        lowercase: bool = True
    ):
        """
        Initialize Indonesian preprocessor.
        
        Args:
            remove_stopwords: Whether to remove Indonesian stopwords
            use_stemming: Whether to apply Sastrawi stemming
            lowercase: Whether to convert to lowercase
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.lowercase = lowercase
        
        # Initialize Sastrawi stemmer (lazy loading)
        self._stemmer = None
        if use_stemming:
            factory = StemmerFactory()
            self._stemmer = factory.create_stemmer()
        
        # Sastrawi stopword remover
        self._stopword_remover = None
        if remove_stopwords:
            factory = StopWordRemoverFactory()
            self._stopword_remover = factory.create_stop_word_remover()
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing noise.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags (keep the text without #)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet indicator)
        text = re.sub(r'\brt\b', '', text)
        
        # Normalize repeated characters (e.g., "senaaang" -> "senang")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_slang(self, text: str) -> str:
        """
        Normalize common Indonesian slang/abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        slang_dict = {
            'gak': 'tidak',
            'ga': 'tidak',
            'gk': 'tidak',
            'ngga': 'tidak',
            'nggak': 'tidak',
            'enggak': 'tidak',
            'tdk': 'tidak',
            'yg': 'yang',
            'dgn': 'dengan',
            'utk': 'untuk',
            'krn': 'karena',
            'tp': 'tapi',
            'jg': 'juga',
            'sm': 'sama',
            'bgt': 'sekali',
            'bngt': 'sekali',
            'bngtt': 'sekali',
            'skrg': 'sekarang',
            'blm': 'belum',
            'sdh': 'sudah',
            'udh': 'sudah',
            'kmrn': 'kemarin',
            'bsk': 'besok',
            'org': 'orang',
            'lg': 'lagi',
            'dr': 'dari',
            'kl': 'kalau',
            'klo': 'kalau',
            'klu': 'kalau',
            'emg': 'memang',
            'emang': 'memang',
            'gmn': 'bagaimana',
            'gimana': 'bagaimana',
            'knp': 'kenapa',
            'knapa': 'kenapa',
            'spy': 'supaya',
            'bkn': 'bukan',
            'bnyk': 'banyak',
            'sy': 'saya',
            'ak': 'aku',
            'gue': 'aku',
            'gw': 'aku',
            'lo': 'kamu',
            'lu': 'kamu',
            'elu': 'kamu',
            'doi': 'dia',
            'dy': 'dia',
            'mrk': 'mereka',
        }
        
        words = text.split()
        normalized = []
        for word in words:
            normalized.append(slang_dict.get(word, word))
        
        return ' '.join(normalized)
    
    def remove_stopwords_func(self, text: str) -> str:
        """
        Remove Indonesian stopwords.
        
        Args:
            text: Input text
            
        Returns:
            Text with stopwords removed
        """
        if self._stopword_remover:
            return self._stopword_remover.remove(text)
        
        # Fallback to manual stopword removal
        words = text.split()
        filtered = [w for w in words if w not in self.STOPWORDS_ID]
        return ' '.join(filtered)
    
    def stem(self, text: str) -> str:
        """
        Apply Indonesian stemming using Sastrawi.
        
        Args:
            text: Input text
            
        Returns:
            Stemmed text
        """
        if self._stemmer:
            return self._stemmer.stem(text)
        return text
    
    def preprocess(self, text: str) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Normalize slang
        text = self.normalize_slang(text)
        
        # Step 3: Remove stopwords (optional)
        if self.remove_stopwords:
            text = self.remove_stopwords_func(text)
        
        # Step 4: Stemming (optional)
        if self.use_stemming:
            text = self.stem(text)
        
        return text.strip()
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Quick test
if __name__ == "__main__":
    preprocessor = IndonesianTextPreprocessor(
        remove_stopwords=True,
        use_stemming=True
    )
    
    test_texts = [
        "Aku senang banget hari ini! 😊",
        "Gue bgt sedih krn gagal ujian",
        "@user123 ini berita yg bikin marah bgt https://example.com",
        "RT: Takut bgt sama situasi skrg",
        "Kangen bgt sm doi 💕",
    ]
    
    print("Indonesian Preprocessor Test")
    print("="*50)
    
    for text in test_texts:
        result = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Processed: {result}")
        print("-"*50)
