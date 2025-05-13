import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from ..data_collection.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeSeriesProcessor:
    def __init__(self):
        """Initialize the time series processor."""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'timeseries'), exist_ok=True)
    
    def load_stock_data(self, stock_symbol):
        """
        Load stock price data for a given symbol.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            
        Returns:
            pandas.DataFrame: Stock price data
        """
        filename = os.path.join(RAW_DATA_PATH, 'stocks', f"{stock_symbol}_price_data.csv")
        
        try:
            df = pd.read_csv(filename)
            
            # Convert Date to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Set Date as index
            df = df.set_index('Date')
            
            logger.info(f"Loaded stock data for {stock_symbol} with {len(df)} entries")
            return df
        except FileNotFoundError:
            logger.error(f"Stock data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading stock data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, stock_symbol, data_type='reddit'):
        """
        Load sentiment data for a given symbol and data type.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            
        Returns:
            pandas.DataFrame: Sentiment data
        """
        filename = os.path.join(PROCESSED_DATA_PATH, 'sentiment', data_type, f"{stock_symbol}_sentiment.csv")
        
        try:
            df = pd.read_csv(filename)
            
            # Convert date column to datetime
            if data_type == 'reddit':
                date_column = 'created_utc'
            else:  # news
                date_column = 'publishedAt'
            
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column])
            
            logger.info(f"Loaded {data_type} sentiment data for {stock_symbol} with {len(df)} entries")
            return df
        except FileNotFoundError:
            logger.error(f"Sentiment data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading sentiment data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def aggregate_daily_sentiment(self, df, date_column, sentiment_column='combined_sentiment_polarity'):
        """
        Aggregate sentiment data to daily frequencies.
        
        Args:
            df (pandas.DataFrame): Sentiment data
            date_column (str): Column name containing dates
            sentiment_column (str): Column name containing sentiment scores
            
        Returns:
            pandas.DataFrame: Daily aggregated sentiment
        """
        if df.empty:
            return pd.DataFrame()
        
        # Find the appropriate sentiment column
        if sentiment_column not in df.columns:
            sentiment_columns = [col for col in df.columns if col.endswith('_polarity')]
            if sentiment_columns:
                sentiment_column = sentiment_columns[0]
                logger.info(f"Using {sentiment_column} for sentiment aggregation")
            else:
                logger.error("No sentiment polarity column found")
                return pd.DataFrame()
        
        # Create a date column without time
        df['date'] = df[date_column].dt.date
        
        # Calculate daily sentiment metrics
        daily_sentiment = df.groupby('date').agg({
            sentiment_column: ['mean', 'median', 'std', 'count']
        })
        
        # Flatten column names
        daily_sentiment.columns = [f"sentiment_{col[1]}" for col in daily_sentiment.columns]
        
        # Reset index and convert date to datetime
        daily_sentiment = daily_sentiment.reset_index()
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        
        return daily_sentiment
    
    def merge_price_and_sentiment(self, stock_data, sentiment_data, lag_days=0):
        """
        Merge stock price data with sentiment data.
        
        Args:
            stock_data (pandas.DataFrame): Stock price data
            sentiment_data (pandas.DataFrame): Sentiment data
            lag_days (int): Number of days to lag sentiment (negative for lead)
            
        Returns:
            pandas.DataFrame: Merged dataframe
        """
        if stock_data.empty or sentiment_data.empty:
            return pd.DataFrame()
        
        # Reset index to get Date as a column
        stock_data_reset = stock_data.reset_index()
        
        # Create a date column in sentiment_data that's adjusted for lag
        sentiment_data['adjusted_date'] = sentiment_data['date'] + timedelta(days=lag_days)
        
        # Merge the dataframes
        merged_df = pd.merge(
            stock_data_reset,
            sentiment_data,
            left_on='Date',
            right_on='adjusted_date',
            how='left'
        )
        
        # Fill NaN sentiment values with 0 (neutral)
        sentiment_columns = [col for col in merged_df.columns if col.startswith('sentiment_')]
        for col in sentiment_columns:
            merged_df[col] = merged_df[col].fillna(0)
        
        return merged_df
    
    def create_lagged_features(self, df, sentiment_cols, lag_days=5):
        """
        Create lagged features for sentiment analysis.
        
        Args:
            df (pandas.DataFrame): Combined price and sentiment data
            sentiment_cols (list): List of sentiment columns to lag
            lag_days (int): Maximum number of days to lag
            
        Returns:
            pandas.DataFrame: DataFrame with lagged features
        """
        result_df = df.copy()
        
        # Create lagged features
        for col in sentiment_cols:
            for lag in range(1, lag_days + 1):
                result_df[f"{col}_lag{lag}"] = result_df[col].shift(lag)
        
        # Drop rows with NaN values
        result_df = result_df.dropna()
        
        return result_df
    
    def process_stock_data(self, stock_symbol, data_types=None, max_lag_days=5):
        """
        Process stock and sentiment data for correlation analysis.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_types (list): List of data types to process
            max_lag_days (int): Maximum number of days to lag
            
        Returns:
            dict: Dictionary of processed data for each data type
        """
        if data_types is None:
            data_types = ['reddit', 'news']
        
        results = {}
        
        # Load stock price data
        stock_data = self.load_stock_data(stock_symbol)
        
        if stock_data.empty:
            logger.error(f"No stock price data available for {stock_symbol}")
            return results
        
        for data_type in data_types:
            # Load sentiment data
            sentiment_df = self.load_sentiment_data(stock_symbol, data_type)
            
            if sentiment_df.empty:
                logger.warning(f"No {data_type} sentiment data available for {stock_symbol}")
                continue
            
            # Determine date column based on data type
            if data_type == 'reddit':
                date_column = 'created_utc'
            else:  # news
                date_column = 'publishedAt'
            
            # Aggregate sentiment data to daily frequency
            daily_sentiment = self.aggregate_daily_sentiment(sentiment_df, date_column)
            
            if daily_sentiment.empty:
                logger.warning(f"Failed to aggregate {data_type} sentiment for {stock_symbol}")
                continue
            
            # Create different lag versions for correlation analysis
            lag_results = {}
            
            for lag in range(-max_lag_days, max_lag_days + 1):
                # Merge price and sentiment data with lag
                merged_df = self.merge_price_and_sentiment(stock_data, daily_sentiment, lag)
                
                if not merged_df.empty:
                    lag_results[lag] = merged_df
                    
                    # Save processed data
                    output_dir = os.path.join(PROCESSED_DATA_PATH, 'timeseries', data_type)
                    os.makedirs(output_dir, exist_ok=True)
                    output_file = os.path.join(output_dir, f"{stock_symbol}_lag{lag}.csv")
                    merged_df.to_csv(output_file, index=False)
            
            results[data_type] = lag_results
        
        return results
    
    def process_all_stocks(self, stock_symbols, data_types=None, max_lag_days=5):
        """
        Process all stocks for correlation analysis.
        
        Args:
            stock_symbols (list): List of stock symbols to process
            data_types (list): List of data types to process
            max_lag_days (int): Maximum number of days to lag
            
        Returns:
            dict: Dictionary of processed data for each stock
        """
        all_results = {}
        
        for stock in stock_symbols:
            logger.info(f"Processing time series data for {stock}")
            stock_results = self.process_stock_data(stock, data_types, max_lag_days)
            all_results[stock] = stock_results
        
        return all_results


if __name__ == "__main__":
    # Test the processor
    processor = TimeSeriesProcessor()
    
    # Test with a stock
    # processor.process_stock_data('AAPL') 