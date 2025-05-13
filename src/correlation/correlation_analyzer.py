import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from ..data_collection.config import TARGET_STOCKS, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    def __init__(self):
        """Initialize the correlation analyzer."""
        # Create output directories if they don't exist
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'correlation'), exist_ok=True)
        os.makedirs(os.path.join(PROCESSED_DATA_PATH, 'correlation', 'figures'), exist_ok=True)
    
    def load_processed_data(self, stock_symbol, data_type, lag_days):
        """
        Load processed data for correlation analysis.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            lag_days (int): Number of days lagged
            
        Returns:
            pandas.DataFrame: Processed data
        """
        filename = os.path.join(PROCESSED_DATA_PATH, 'timeseries', data_type, f"{stock_symbol}_lag{lag_days}.csv")
        
        try:
            df = pd.read_csv(filename)
            
            # Convert Date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            logger.info(f"Loaded {data_type} data for {stock_symbol} with lag {lag_days}")
            return df
        except FileNotFoundError:
            logger.warning(f"Processed data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return pd.DataFrame()
    
    def calculate_correlation(self, df, price_col='Daily_Return', sentiment_col='sentiment_mean', method='pearson'):
        """
        Calculate correlation between price and sentiment.
        
        Args:
            df (pandas.DataFrame): Data frame with price and sentiment data
            price_col (str): Column name for price/return data
            sentiment_col (str): Column name for sentiment data
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            tuple: Correlation coefficient and p-value
        """
        if df.empty or price_col not in df.columns or sentiment_col not in df.columns:
            return (0, 1.0)  # No correlation
        
        # Remove rows with NaN values
        valid_data = df[[price_col, sentiment_col]].dropna()
        
        if len(valid_data) < 3:  # Need at least 3 points for correlation
            return (0, 1.0)
        
        # Calculate correlation
        if method == 'pearson':
            corr, p_value = pearsonr(valid_data[price_col], valid_data[sentiment_col])
        else:  # spearman
            corr, p_value = spearmanr(valid_data[price_col], valid_data[sentiment_col])
        
        return (corr, p_value)
    
    def plot_correlation(self, df, stock_symbol, data_type, lag_days, 
                         price_col='Daily_Return', sentiment_col='sentiment_mean'):
        """
        Create a scatter plot of price vs sentiment.
        
        Args:
            df (pandas.DataFrame): Data frame with price and sentiment data
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            lag_days (int): Number of days lagged
            price_col (str): Column name for price/return data
            sentiment_col (str): Column name for sentiment data
            
        Returns:
            str: Path to saved figure
        """
        if df.empty or price_col not in df.columns or sentiment_col not in df.columns:
            logger.warning(f"Cannot create plot: missing data or columns")
            return None
        
        # Remove rows with NaN values
        valid_data = df[[price_col, sentiment_col, 'Date']].dropna()
        
        if len(valid_data) < 3:
            logger.warning(f"Not enough valid data points for plotting")
            return None
        
        # Calculate correlation for the plot title
        corr, p_value = self.calculate_correlation(df, price_col, sentiment_col)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with regression line
        sns.regplot(x=sentiment_col, y=price_col, data=valid_data, scatter_kws={'alpha':0.6})
        
        # Add title and labels
        lag_desc = "lead" if lag_days < 0 else "lag" if lag_days > 0 else "same day"
        abs_lag = abs(lag_days)
        lag_str = f"{abs_lag} day {lag_desc}" if abs_lag > 0 else "same day"
        
        plt.title(f"{stock_symbol}: {data_type.capitalize()} Sentiment vs. Daily Returns ({lag_str})\nCorrelation: {corr:.3f} (p-value: {p_value:.3f})")
        plt.xlabel(f"{sentiment_col.replace('_', ' ').title()}")
        plt.ylabel(f"{price_col.replace('_', ' ').title()}")
        
        # Add a horizontal line at y=0
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Save the figure
        output_dir = os.path.join(PROCESSED_DATA_PATH, 'correlation', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock_symbol}_{data_type}_lag{lag_days}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation plot to {output_file}")
        
        return output_file
    
    def analyze_all_lags(self, stock_symbol, data_type, price_col='Daily_Return', 
                        sentiment_col='sentiment_mean', max_lag_days=5, create_plots=True):
        """
        Analyze correlation for all lag periods.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            price_col (str): Column name for price/return data
            sentiment_col (str): Column name for sentiment data
            max_lag_days (int): Maximum number of days to lag
            create_plots (bool): Whether to create plots
            
        Returns:
            pandas.DataFrame: Correlation results for all lags
        """
        results = []
        
        for lag in range(-max_lag_days, max_lag_days + 1):
            # Load data for the current lag
            df = self.load_processed_data(stock_symbol, data_type, lag)
            
            if df.empty:
                continue
            
            # Calculate correlation
            corr, p_value = self.calculate_correlation(df, price_col, sentiment_col)
            
            # Create plot if needed
            plot_file = None
            if create_plots:
                plot_file = self.plot_correlation(df, stock_symbol, data_type, lag, price_col, sentiment_col)
            
            # Save results
            results.append({
                'stock': stock_symbol,
                'data_type': data_type,
                'lag': lag,
                'price_column': price_col,
                'sentiment_column': sentiment_col,
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'plot_file': plot_file,
                'data_points': len(df.dropna(subset=[price_col, sentiment_col]))
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            # Save results
            output_file = os.path.join(PROCESSED_DATA_PATH, 'correlation', f"{stock_symbol}_{data_type}_correlations.csv")
            results_df.to_csv(output_file, index=False)
            logger.info(f"Saved correlation results to {output_file}")
            
            # Plot correlation vs lag
            if create_plots and len(results_df) > 1:
                self.plot_correlation_vs_lag(results_df, stock_symbol, data_type)
        
        return results_df
    
    def plot_correlation_vs_lag(self, results_df, stock_symbol, data_type):
        """
        Plot correlation vs lag days.
        
        Args:
            results_df (pandas.DataFrame): DataFrame with correlation results
            stock_symbol (str): Stock ticker symbol
            data_type (str): Type of data ('reddit' or 'news')
            
        Returns:
            str: Path to saved figure
        """
        plt.figure(figsize=(12, 6))
        
        # Main plot
        ax = plt.subplot(111)
        
        # Plot correlation vs lag
        ax.plot(results_df['lag'], results_df['correlation'], marker='o', linestyle='-', color='blue')
        
        # Add horizontal line at correlation = 0
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add vertical line at lag = 0
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight significant correlations
        significant = results_df[results_df['significant']]
        if not significant.empty:
            ax.scatter(significant['lag'], significant['correlation'], color='red', s=100, alpha=0.7, 
                       label='Significant (p < 0.05)')
        
        # Add title and labels
        ax.set_title(f"{stock_symbol}: Correlation between {data_type.capitalize()} Sentiment and Daily Returns")
        ax.set_xlabel('Lag Days (Negative = Sentiment leads Returns, Positive = Returns lead Sentiment)')
        ax.set_ylabel('Correlation Coefficient')
        
        # Show only integer ticks for lag
        ax.set_xticks(sorted(results_df['lag'].unique()))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend if needed
        if not significant.empty:
            ax.legend()
        
        # Save the figure
        output_dir = os.path.join(PROCESSED_DATA_PATH, 'correlation', 'figures')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{stock_symbol}_{data_type}_correlation_vs_lag.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved correlation vs lag plot to {output_file}")
        
        return output_file
    
    def analyze_stock(self, stock_symbol, data_types=None, max_lag_days=5):
        """
        Analyze correlation for a single stock.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            data_types (list): List of data types to analyze
            max_lag_days (int): Maximum number of days to lag
            
        Returns:
            dict: Correlation results for each data type
        """
        if data_types is None:
            data_types = ['reddit', 'news']
        
        results = {}
        
        for data_type in data_types:
            logger.info(f"Analyzing correlation for {stock_symbol} {data_type} data")
            
            # Analyze with daily returns
            results_returns = self.analyze_all_lags(
                stock_symbol, data_type, 
                price_col='Daily_Return', 
                sentiment_col='sentiment_mean',
                max_lag_days=max_lag_days
            )
            
            # Analyze with trading volume
            results_volume = self.analyze_all_lags(
                stock_symbol, data_type, 
                price_col='Volume', 
                sentiment_col='sentiment_mean',
                max_lag_days=max_lag_days
            )
            
            results[data_type] = {
                'returns': results_returns,
                'volume': results_volume
            }
        
        return results
    
    def analyze_all_stocks(self, stocks=None, data_types=None, max_lag_days=5):
        """
        Analyze correlation for all stocks.
        
        Args:
            stocks (list): List of stock symbols to analyze
            data_types (list): List of data types to analyze
            max_lag_days (int): Maximum number of days to lag
            
        Returns:
            dict: Correlation results for all stocks
        """
        if stocks is None:
            stocks = TARGET_STOCKS
        
        if data_types is None:
            data_types = ['reddit', 'news']
        
        all_results = {}
        all_correlations = []
        
        for stock in stocks:
            logger.info(f"Analyzing correlation for {stock}")
            stock_results = self.analyze_stock(stock, data_types, max_lag_days)
            all_results[stock] = stock_results
            
            # Collect all correlation results for summary
            for data_type in stock_results:
                for metric_type, corr_df in stock_results[data_type].items():
                    if not corr_df.empty:
                        corr_df['metric_type'] = metric_type
                        all_correlations.append(corr_df)
        
        # Create overall summary
        if all_correlations:
            all_correlations_df = pd.concat(all_correlations)
            summary_file = os.path.join(PROCESSED_DATA_PATH, 'correlation', 'all_correlations_summary.csv')
            all_correlations_df.to_csv(summary_file, index=False)
            logger.info(f"Saved all correlations summary to {summary_file}")
        
        return all_results


if __name__ == "__main__":
    # Test the analyzer
    analyzer = CorrelationAnalyzer()
    
    # Test with a stock
    # analyzer.analyze_stock('AAPL') 