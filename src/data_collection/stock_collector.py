import os
import pandas as pd
import yfinance as yf
import logging
from datetime import timedelta
from .config import (
    TARGET_STOCKS,
    START_DATE,
    END_DATE,
    RAW_DATA_PATH
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self):
        """Initialize the stock data collector."""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(RAW_DATA_PATH, 'stocks'), exist_ok=True)
    
    def collect_stock_data(self, ticker, start_date=None, end_date=None):
        """
        Collect historical stock data for a given ticker symbol.
        
        Args:
            ticker (str): The stock ticker symbol
            start_date (datetime): The start date for data collection
            end_date (datetime): The end date for data collection
            
        Returns:
            pandas.DataFrame: Historical stock data
        """
        if start_date is None:
            start_date = START_DATE
        if end_date is None:
            end_date = END_DATE
            
        # Add a buffer of a few days before the start date to calculate accurate returns
        buffered_start = start_date - timedelta(days=5)
        
        logger.info(f"Collecting stock data for {ticker} from {buffered_start.date()} to {end_date.date()}")
        
        try:
            # Get stock data from Yahoo Finance
            stock_data = yf.download(
                ticker,
                start=buffered_start,
                end=end_date + timedelta(days=1),  # Add one day to include the end date
                progress=False
            )
            
            if stock_data.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
                
            # Calculate daily returns and volatility
            stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
            
            # Calculate moving averages
            stock_data['MA5'] = stock_data['Adj Close'].rolling(window=5).mean()
            stock_data['MA20'] = stock_data['Adj Close'].rolling(window=20).mean()
            
            # Calculate trading volume moving average
            stock_data['Volume_MA5'] = stock_data['Volume'].rolling(window=5).mean()
            
            # Save the data
            filename = os.path.join(RAW_DATA_PATH, 'stocks', f"{ticker}_price_data.csv")
            stock_data.to_csv(filename)
            logger.info(f"Saved stock data for {ticker} to {filename}")
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {e}")
            return pd.DataFrame()
    
    def collect_all_stocks(self):
        """Collect data for all target stocks."""
        all_stocks_data = {}
        
        for stock in TARGET_STOCKS:
            logger.info(f"Collecting price data for {stock}")
            stock_data = self.collect_stock_data(stock)
            all_stocks_data[stock] = stock_data
        
        return all_stocks_data
    
    def collect_market_data(self):
        """Collect market index data for reference (S&P 500)."""
        logger.info("Collecting market index data (S&P 500)")
        market_data = self.collect_stock_data('^GSPC')  # S&P 500 index
        
        return market_data


if __name__ == "__main__":
    # Test the collector
    collector = StockDataCollector()
    stock_data = collector.collect_all_stocks()
    market_data = collector.collect_market_data()
    print(f"Collected price data for {len(stock_data)} stocks and market index") 