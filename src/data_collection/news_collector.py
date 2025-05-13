import os
import pandas as pd
import requests
from datetime import datetime
import logging
import time
from .config import (
    NEWS_API_KEY,
    TARGET_STOCKS,
    START_DATE,
    END_DATE,
    RAW_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        """Initialize the News API collector."""
        self.api_key = NEWS_API_KEY
        
        if not self.api_key:
            logger.warning("News API key not found. Please check your .env file.")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(RAW_DATA_PATH, 'news'), exist_ok=True)
        
        # Map stock symbols to company names for better search results
        self.company_mapping = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'NVDA': 'Nvidia',
            'GOOGL': 'Google OR Alphabet',
            'AMZN': 'Amazon',
            # Add more mappings as needed
        }
    
    def fetch_news(self, query, start_date=None, end_date=None, page_size=100, page=1):
        """
        Fetch news articles from News API.
        
        Args:
            query (str): The search query
            start_date (datetime): The start date for news articles
            end_date (datetime): The end date for news articles
            page_size (int): Number of results per request (max 100)
            page (int): Page number
            
        Returns:
            list: News articles
        """
        if not self.api_key:
            logger.error("News API key not available")
            return []
        
        if start_date is None:
            start_date = START_DATE
        if end_date is None:
            end_date = END_DATE
        
        # Format dates for News API
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching news for query: '{query}' from {from_date} to {to_date}")
        
        url = 'https://newsapi.org/v2/everything'
        
        params = {
            'q': query,
            'apiKey': self.api_key,
            'from': from_date,
            'to': to_date,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"API Error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            total_results = data.get('totalResults', 0)
            
            logger.info(f"Retrieved {len(articles)} articles out of {total_results} total results")
            
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def collect_stock_news(self, stock_symbol, max_pages=5):
        """
        Collect news articles related to a specific stock.
        
        Args:
            stock_symbol (str): The stock ticker symbol
            max_pages (int): Maximum number of pages to retrieve
            
        Returns:
            pandas.DataFrame: News articles
        """
        all_articles = []
        
        # Get the company name for the stock symbol
        company_name = self.company_mapping.get(stock_symbol, stock_symbol)
        
        # Create queries for both stock symbol and company name
        queries = [
            f"{stock_symbol} stock",
            f"{company_name} stock",
            f"{company_name} company"
        ]
        
        for query in queries:
            page = 1
            while page <= max_pages:
                articles = self.fetch_news(query, page=page)
                
                if not articles:
                    break
                
                all_articles.extend(articles)
                
                # News API has rate limits, so we need to sleep between requests
                time.sleep(1)
                page += 1
        
        # Convert to DataFrame
        if all_articles:
            df = pd.DataFrame(all_articles)
            
            # Add stock symbol and query columns
            df['stock_symbol'] = stock_symbol
            
            # Parse publishedAt to datetime
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            
            # Remove duplicates based on title and URL
            df = df.drop_duplicates(subset=['title', 'url'])
            
            # Save the data
            filename = os.path.join(RAW_DATA_PATH, 'news', f"{stock_symbol}_news_data.csv")
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(df)} news articles for {stock_symbol} to {filename}")
            
            return df
        else:
            logger.warning(f"No news articles found for {stock_symbol}")
            return pd.DataFrame()
    
    def collect_all_stocks(self):
        """Collect news for all target stocks."""
        all_stocks_news = {}
        
        for stock in TARGET_STOCKS:
            logger.info(f"Collecting news for {stock}")
            stock_news = self.collect_stock_news(stock)
            all_stocks_news[stock] = stock_news
            
            # Sleep between stocks to respect API rate limits
            time.sleep(2)
        
        return all_stocks_news


if __name__ == "__main__":
    # Test the collector
    collector = NewsCollector()
    all_news = collector.collect_all_stocks()
    print(f"Collected news for {len(all_news)} stocks") 