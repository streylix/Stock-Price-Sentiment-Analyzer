import os
import logging
import pandas as pd
import argparse

from .time_series_processor import TimeSeriesProcessor
from .correlation_analyzer import CorrelationAnalyzer
from ..data_collection.config import TARGET_STOCKS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'correlation_analysis.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_correlation_analysis(stocks=None, data_types=None, max_lag_days=5, skip_processing=False):
    """
    Run correlation analysis on processed data.
    
    Args:
        stocks (list): List of stock symbols to analyze
        data_types (list): List of data types to analyze
        max_lag_days (int): Maximum number of days to lag
        skip_processing (bool): Skip time series processing if already done
        
    Returns:
        dict: Results of correlation analysis
    """
    if stocks is None:
        stocks = TARGET_STOCKS
    
    if data_types is None:
        data_types = ['reddit', 'news']
    
    logger.info("Starting correlation analysis")
    logger.info(f"Analyzing stocks: {', '.join(stocks)}")
    logger.info(f"Analyzing data types: {', '.join(data_types)}")
    
    os.makedirs('logs', exist_ok=True)
    
    # Step 1: Process time series data (combine stock prices with sentiment)
    if not skip_processing:
        logger.info("Processing time series data")
        processor = TimeSeriesProcessor()
        processor.process_all_stocks(stocks, data_types, max_lag_days)
    else:
        logger.info("Skipping time series processing")
    
    # Step 2: Analyze correlations
    logger.info("Analyzing correlations")
    analyzer = CorrelationAnalyzer()
    results = analyzer.analyze_all_stocks(stocks, data_types, max_lag_days)
    
    # Step 3: Find significant correlations
    significant_correlations = find_significant_correlations(results)
    
    # Log significant findings
    if significant_correlations:
        logger.info("Significant correlations found:")
        for finding in significant_correlations:
            logger.info(f"  - {finding}")
    else:
        logger.info("No significant correlations found.")
    
    logger.info("Correlation analysis completed")
    
    return results

def find_significant_correlations(results):
    """
    Extract significant correlations from results.
    
    Args:
        results (dict): Correlation analysis results
        
    Returns:
        list: List of significant correlation findings
    """
    findings = []
    
    for stock in results:
        for data_type in results[stock]:
            for metric_type, corr_df in results[stock][data_type].items():
                if not corr_df.empty:
                    # Get significant correlations
                    significant = corr_df[corr_df['significant']]
                    
                    if not significant.empty:
                        # Find the strongest correlation
                        strongest = significant.loc[significant['correlation'].abs().idxmax()]
                        
                        # Create finding description
                        lag = strongest['lag']
                        corr = strongest['correlation']
                        direction = "positive" if corr > 0 else "negative"
                        
                        if lag < 0:
                            timing = f"{abs(lag)} day(s) before"
                        elif lag > 0:
                            timing = f"{lag} day(s) after"
                        else:
                            timing = "on the same day as"
                        
                        finding = (
                            f"{stock}: {direction} correlation ({corr:.3f}) between {data_type} sentiment "
                            f"and {metric_type} {timing} sentiment (p={strongest['p_value']:.3f})"
                        )
                        
                        findings.append(finding)
    
    return findings

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run correlation analysis between sentiment and stock prices')
    
    parser.add_argument('--stocks', type=str, nargs='+',
                        help='Stock symbols to analyze (default: all stocks in config)')
    
    parser.add_argument('--data-types', type=str, nargs='+', default=['reddit', 'news'],
                        help='Data types to analyze (default: reddit news)')
    
    parser.add_argument('--max-lag', type=int, default=5,
                        help='Maximum number of days to lag (default: 5)')
    
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip time series processing if already done')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    run_correlation_analysis(
        stocks=args.stocks,
        data_types=args.data_types,
        max_lag_days=args.max_lag,
        skip_processing=args.skip_processing
    ) 