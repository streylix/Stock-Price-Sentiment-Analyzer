import os
import pandas as pd
import praw
import datetime
from tqdm import tqdm
import logging
from .config import (
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
    TARGET_STOCKS,
    START_DATE,
    END_DATE,
    RAW_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedditCollector:
    def __init__(self):
        """Initialize the Reddit API client."""
        if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
            logger.warning("Reddit API credentials not found. Please check your .env file.")
            self.reddit = None
        else:
            self.reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(RAW_DATA_PATH, 'reddit'), exist_ok=True)
    
    def _timestamp_to_datetime(self, timestamp):
        """Convert Unix timestamp to datetime."""
        return datetime.datetime.fromtimestamp(timestamp)
    
    def collect_subreddit_data(self, subreddit_name, query, limit=1000):
        """Collect posts from a subreddit matching the query."""
        if not self.reddit:
            logger.error("Reddit client not initialized")
            return pd.DataFrame()
        
        logger.info(f"Collecting posts from r/{subreddit_name} with query: {query}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []
        
        try:
            # Search for posts matching the query
            for submission in tqdm(subreddit.search(query, sort='new', time_filter='all', limit=limit)):
                # Convert timestamp to datetime
                created_utc = self._timestamp_to_datetime(submission.created_utc)
                
                # Filter by date range
                if START_DATE <= created_utc <= END_DATE:
                    posts.append({
                        'id': submission.id,
                        'title': submission.title,
                        'created_utc': created_utc,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': submission.upvote_ratio,
                        'url': submission.url,
                        'selftext': submission.selftext,
                        'subreddit': subreddit_name,
                        'query': query
                    })
        except Exception as e:
            logger.error(f"Error collecting data from r/{subreddit_name}: {e}")
        
        return pd.DataFrame(posts)
    
    def collect_stock_data(self, stock_symbol, subreddits=None):
        """Collect data for a specific stock from multiple subreddits."""
        if subreddits is None:
            subreddits = ['wallstreetbets', 'stocks', 'investing', f"{stock_symbol.lower()}"]
        
        all_data = pd.DataFrame()
        
        for subreddit in subreddits:
            try:
                # Collect posts that mention the stock symbol
                df = self.collect_subreddit_data(subreddit, stock_symbol)
                all_data = pd.concat([all_data, df])
                
                # Additional search for full company name if needed
                # E.g., for AAPL, also search "Apple"
                # This would require a mapping of symbols to company names
            except Exception as e:
                logger.error(f"Error collecting {stock_symbol} data from r/{subreddit}: {e}")
        
        # Save the data
        if not all_data.empty:
            filename = os.path.join(RAW_DATA_PATH, 'reddit', f"{stock_symbol}_reddit_data.csv")
            all_data.to_csv(filename, index=False)
            logger.info(f"Saved {len(all_data)} Reddit posts for {stock_symbol} to {filename}")
        
        return all_data
    
    def collect_all_stocks(self):
        """Collect data for all target stocks."""
        all_stocks_data = {}
        
        for stock in TARGET_STOCKS:
            logger.info(f"Collecting Reddit data for {stock}")
            stock_data = self.collect_stock_data(stock)
            all_stocks_data[stock] = stock_data
        
        return all_stocks_data


if __name__ == "__main__":
    # Test the collector
    collector = RedditCollector()
    all_data = collector.collect_all_stocks()
    print(f"Collected data for {len(all_data)} stocks") 