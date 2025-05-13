import os
import logging
import pandas as pd
from datetime import datetime
import argparse

from .sentiment_analyzer import SentimentAnalyzer
from ..data_collection.config import TARGET_STOCKS, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'sentiment_analysis.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'sentiment'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'sentiment', 'reddit'), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'sentiment', 'news'), exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def run_sentiment_analysis(method='textblob', data_types=None, stocks=None):
    """
    Run sentiment analysis on collected data.
    
    Args:
        method (str): Sentiment analysis method ('textblob' or 'transformers')
        data_types (list): List of data types to analyze
        stocks (list): List of stock symbols to analyze
        
    Returns:
        dict: Results of sentiment analysis
    """
    create_directories()
    
    if data_types is None:
        data_types = ['reddit', 'news']
    
    if stocks is None:
        stocks = TARGET_STOCKS
    
    logger.info(f"Starting sentiment analysis using {method} method")
    logger.info(f"Analyzing data types: {', '.join(data_types)}")
    logger.info(f"Analyzing stocks: {', '.join(stocks)}")
    
    analyzer = SentimentAnalyzer(method=method)
    
    results = {}
    for stock in stocks:
        results[stock] = {}
        for data_type in data_types:
            logger.info(f"Analyzing {stock} {data_type} data")
            try:
                result_df = analyzer.analyze_stock_data(stock, data_type)
                results[stock][data_type] = result_df
                
                if not result_df.empty:
                    # Log sentiment distribution
                    if 'combined_sentiment' in result_df.columns:
                        sentiment_counts = result_df['combined_sentiment'].value_counts()
                    elif 'title_sentiment_sentiment' in result_df.columns:
                        sentiment_counts = result_df['title_sentiment_sentiment'].value_counts()
                    else:
                        sentiment_counts = pd.Series()
                    
                    logger.info(f"{stock} {data_type} sentiment distribution: {dict(sentiment_counts)}")
            except Exception as e:
                logger.error(f"Error analyzing {stock} {data_type} data: {e}")
    
    # Save a summary of the sentiment analysis
    summary_data = []
    
    for stock in results:
        for data_type in results[stock]:
            df = results[stock][data_type]
            
            if not df.empty:
                # Get sentiment columns
                sentiment_columns = [col for col in df.columns if col.endswith('_sentiment')]
                polarity_columns = [col for col in df.columns if col.endswith('_polarity')]
                
                if 'combined_sentiment_polarity' in df.columns:
                    avg_polarity = df['combined_sentiment_polarity'].mean()
                elif polarity_columns:
                    avg_polarity = df[polarity_columns[0]].mean()
                else:
                    avg_polarity = 0.0
                
                # Count sentiment categories
                if 'combined_sentiment' in df.columns:
                    positive_count = (df['combined_sentiment'] == 'positive').sum()
                    negative_count = (df['combined_sentiment'] == 'negative').sum()
                    neutral_count = (df['combined_sentiment'] == 'neutral').sum()
                elif sentiment_columns:
                    positive_count = (df[sentiment_columns[0]] == 'positive').sum()
                    negative_count = (df[sentiment_columns[0]] == 'negative').sum()
                    neutral_count = (df[sentiment_columns[0]] == 'neutral').sum()
                else:
                    positive_count = negative_count = neutral_count = 0
                
                total_count = len(df)
                
                summary_data.append({
                    'stock': stock,
                    'data_type': data_type,
                    'total_entries': total_count,
                    'positive_count': positive_count,
                    'negative_count': negative_count,
                    'neutral_count': neutral_count,
                    'positive_ratio': positive_count / total_count if total_count > 0 else 0,
                    'negative_ratio': negative_count / total_count if total_count > 0 else 0,
                    'neutral_ratio': neutral_count / total_count if total_count > 0 else 0,
                    'avg_polarity': avg_polarity
                })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(PROCESSED_DATA_PATH, 'sentiment', 'sentiment_analysis_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved sentiment analysis summary to {summary_file}")
    
    logger.info("Sentiment analysis completed")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run sentiment analysis on stock-related text data')
    
    parser.add_argument('--method', type=str, default='textblob', choices=['textblob', 'transformers'],
                        help='Sentiment analysis method (default: textblob)')
    
    parser.add_argument('--data-types', type=str, nargs='+', default=['reddit', 'news'],
                        help='Data types to analyze (default: reddit news)')
    
    parser.add_argument('--stocks', type=str, nargs='+',
                        help='Stock symbols to analyze (default: all stocks in config)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_sentiment_analysis(method=args.method, data_types=args.data_types, stocks=args.stocks) 