import os
import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import logging
import torch
from .text_preprocessor import TextPreprocessor
from ..data_collection.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, method='textblob', use_preprocessing=True):
        """
        Initialize the sentiment analyzer.
        
        Args:
            method (str): Sentiment analysis method ('textblob', 'vader', or 'transformers')
            use_preprocessing (bool): Whether to preprocess text before analysis
        """
        self.method = method
        self.use_preprocessing = use_preprocessing
        
        # Initialize text preprocessor
        if use_preprocessing:
            self.preprocessor = TextPreprocessor()
        
        # Initialize the sentiment analyzer based on the chosen method
        if method == 'transformers':
            try:
                # Load FinBERT model for financial sentiment analysis
                model_name = "ProsusAI/finbert"
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
                logger.info(f"Loaded transformer model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading transformer model: {e}")
                logger.info("Falling back to TextBlob")
                self.method = 'textblob'
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'sentiment'), exist_ok=True)
    
    def analyze_text_textblob(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores (polarity and subjectivity)
        """
        if not text or not isinstance(text, str):
            return {'polarity': 0.0, 'subjectivity': 0.0}
        
        analysis = TextBlob(text)
        
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity,
            'sentiment': 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
        }
    
    def analyze_text_transformers(self, text):
        """
        Analyze sentiment using a transformer model.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not text or not isinstance(text, str):
            return {'polarity': 0.0, 'label': 'neutral'}
        
        # Truncate text if too long for the model
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            result = self.nlp(text)[0]
            
            # Map the result to a polarity score
            label = result['label']
            score = result['score']
            
            if label == 'positive':
                polarity = score
            elif label == 'negative':
                polarity = -score
            else:
                polarity = 0.0
            
            return {
                'polarity': polarity,
                'score': score,
                'label': label,
                'sentiment': label
            }
        except Exception as e:
            logger.error(f"Error analyzing text with transformer: {e}")
            # Fall back to TextBlob if transformer analysis fails
            return self.analyze_text_textblob(text)
    
    def analyze_text(self, text):
        """
        Analyze sentiment of a text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if self.use_preprocessing:
            text = self.preprocessor.preprocess(text)
        
        if self.method == 'transformers':
            return self.analyze_text_transformers(text)
        else:
            return self.analyze_text_textblob(text)
    
    def analyze_dataframe(self, df, text_column, new_prefix='sentiment_'):
        """
        Analyze sentiment for all texts in a DataFrame column.
        
        Args:
            df (pandas.DataFrame): DataFrame containing text data
            text_column (str): Column name containing text to analyze
            new_prefix (str): Prefix for new sentiment columns
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment scores added
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for sentiment analysis")
            return df
        
        logger.info(f"Analyzing sentiment for {len(df)} entries using {self.method} method")
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Apply sentiment analysis to each text
        sentiments = []
        for text in df[text_column]:
            sentiments.append(self.analyze_text(text))
        
        # Convert results to DataFrame
        sentiment_df = pd.DataFrame(sentiments)
        
        # Add sentiment columns to the result DataFrame
        for column in sentiment_df.columns:
            result_df[f"{new_prefix}{column}"] = sentiment_df[column]
        
        return result_df
    
    def analyze_stock_data(self, stock_symbol, data_type='reddit'):
        """
        Analyze sentiment for a specific stock's data.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            
        Returns:
            pandas.DataFrame: DataFrame with sentiment scores added
        """
        # Load data
        if data_type == 'reddit':
            filename = os.path.join(RAW_DATA_PATH, 'reddit', f"{stock_symbol}_reddit_data.csv")
            text_column = 'title'  # Use post titles for Reddit data
            additional_text_column = 'selftext'  # Also analyze post content
        else:  # news
            filename = os.path.join(RAW_DATA_PATH, 'news', f"{stock_symbol}_news_data.csv")
            text_column = 'title'
            additional_text_column = 'description'
        
        try:
            df = pd.read_csv(filename)
            
            if df.empty:
                logger.warning(f"No {data_type} data found for {stock_symbol}")
                return df
            
            logger.info(f"Analyzing sentiment for {stock_symbol} {data_type} data ({len(df)} entries)")
            
            # Analyze title sentiment
            result_df = self.analyze_dataframe(df, text_column, f"{text_column}_sentiment_")
            
            # Also analyze additional text if available
            if additional_text_column in df.columns:
                result_df = self.analyze_dataframe(result_df, additional_text_column, f"{additional_text_column}_sentiment_")
                
                # Calculate combined sentiment (average of title and content)
                if f"{text_column}_sentiment_polarity" in result_df.columns and f"{additional_text_column}_sentiment_polarity" in result_df.columns:
                    result_df['combined_sentiment_polarity'] = result_df[[f"{text_column}_sentiment_polarity", f"{additional_text_column}_sentiment_polarity"]].mean(axis=1)
                    
                    # Map combined polarity to sentiment label
                    result_df['combined_sentiment'] = result_df['combined_sentiment_polarity'].apply(
                        lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
                    )
            
            # Save processed data
            output_dir = os.path.join(PROCESSED_DATA_PATH, 'sentiment', data_type)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{stock_symbol}_sentiment.csv")
            result_df.to_csv(output_file, index=False)
            
            logger.info(f"Saved sentiment analysis for {stock_symbol} {data_type} to {output_file}")
            
            return result_df
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error analyzing {data_type} data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_all_stocks(self, data_types=None):
        """
        Analyze sentiment for all stocks and data types.
        
        Args:
            data_types (list): List of data types to analyze ('reddit', 'news')
            
        Returns:
            dict: Dictionary of analyzed data for each stock and data type
        """
        if data_types is None:
            data_types = ['reddit', 'news']
        
        results = {}
        
        # Get list of stocks from directory contents
        stocks = set()
        for data_type in data_types:
            data_dir = os.path.join(RAW_DATA_PATH, data_type)
            if os.path.exists(data_dir):
                for filename in os.listdir(data_dir):
                    if filename.endswith("_data.csv"):
                        stock_symbol = filename.split('_')[0]
                        stocks.add(stock_symbol)
        
        for stock in stocks:
            results[stock] = {}
            for data_type in data_types:
                results[stock][data_type] = self.analyze_stock_data(stock, data_type)
        
        return results


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer(method='textblob')
    
    # Test with a sample text
    sample_text = "Apple's earnings exceeded expectations, showing strong growth in services."
    sentiment = analyzer.analyze_text(sample_text)
    print(f"Sample text: {sample_text}")
    print(f"Sentiment: {sentiment}")
    
    # Test with a stock
    # analyzer.analyze_stock_data('AAPL', 'reddit') 