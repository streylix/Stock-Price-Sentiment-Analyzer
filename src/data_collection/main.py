import os
import logging
import pandas as pd
from datetime import datetime

os.makedirs('logs', exist_ok=True)

from .reddit_collector import RedditCollector
from .news_collector import NewsCollector
from .stock_collector import StockDataCollector
from .config import TARGET_STOCKS, RAW_DATA_PATH, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'data_collection.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def collect_all_data():
    """Collect all data from different sources."""
    create_directories()
    
    logger.info("Starting data collection process")
    collection_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Collect stock price data
    logger.info("Collecting stock price data")
    stock_collector = StockDataCollector()
    stock_data = stock_collector.collect_all_stocks()
    market_data = stock_collector.collect_market_data()
    
    # Collect Reddit data
    logger.info("Collecting Reddit data")
    reddit_collector = RedditCollector()
    reddit_data = reddit_collector.collect_all_stocks()
    
    # Collect news data
    logger.info("Collecting news data")
    news_collector = NewsCollector()
    news_data = news_collector.collect_all_stocks()
    
    # Create a summary of collected data
    summary = []
    for stock in TARGET_STOCKS:
        stock_summary = {
            'stock': stock,
            'price_data_rows': len(stock_data.get(stock, pd.DataFrame())),
            'reddit_posts': len(reddit_data.get(stock, pd.DataFrame())),
            'news_articles': len(news_data.get(stock, pd.DataFrame()))
        }
        summary.append(stock_summary)
    
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(RAW_DATA_PATH, f'data_collection_summary_{collection_time}.csv')
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Data collection completed. Summary saved to {summary_file}")
    logger.info(f"Summary:\n{summary_df.to_string()}")
    
    return {
        'stock_data': stock_data,
        'market_data': market_data,
        'reddit_data': reddit_data,
        'news_data': news_data
    }

if __name__ == "__main__":
    collect_all_data() 