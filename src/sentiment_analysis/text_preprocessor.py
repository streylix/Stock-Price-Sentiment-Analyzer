import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        """Initialize the text preprocessor with required NLTK resources."""
        # Download necessary NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            logger.info("NLTK resources downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK resources: {e}")
        
        # Initialize the lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Get English stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add finance-specific stopwords
        finance_stopwords = {
            'stock', 'stocks', 'market', 'markets', 'company', 'companies',
            'share', 'shares', 'price', 'prices', 'investor', 'investors',
            'trading', 'trader', 'traders', 'trade', 'trades', 'finance',
            'financial', 'investment', 'investments', 'buy', 'sell', 'hold'
        }
        self.stop_words.update(finance_stopwords)
    
    def clean_text(self, text):
        """
        Clean and normalize text data.
        
        Args:
            text (str): The input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text.
        
        Args:
            text (str): The input text
            
        Returns:
            str: Text with stopwords removed
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """
        Lemmatize text to reduce words to their base forms.
        
        Args:
            text (str): The input text
            
        Returns:
            str: Lemmatized text
        """
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def preprocess(self, text):
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text (str): The raw input text
            
        Returns:
            str: Fully preprocessed text
        """
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_text = "Just bought 100 shares of $AAPL at $150! The stock is trending upward after their Q3 earnings report. https://finance.example.com/apple-q3-earnings"
    
    cleaned_text = preprocessor.clean_text(sample_text)
    print(f"Cleaned text: {cleaned_text}")
    
    preprocessed_text = preprocessor.preprocess(sample_text)
    print(f"Preprocessed text: {preprocessed_text}") 