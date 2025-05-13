import os
from dotenv import load_dotenv
import datetime

# Load environment variables
load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# Twitter API credentials
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')

# News API credentials
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Target stocks to analyze
TARGET_STOCKS = os.getenv('TARGET_STOCKS', 'AAPL,MSFT,NVDA,GOOGL,AMZN').split(',')

# Time period for analysis
LOOKBACK_PERIOD = int(os.getenv('LOOKBACK_PERIOD', 30))

# Calculate date range for data collection
END_DATE = datetime.datetime.now()
START_DATE = END_DATE - datetime.timedelta(days=LOOKBACK_PERIOD)

# Data storage paths
RAW_DATA_PATH = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed') 