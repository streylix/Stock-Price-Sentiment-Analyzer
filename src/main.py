import os
import logging
import argparse
from datetime import datetime

from src.data_collection import collect_all_data
from src.sentiment_analysis import run_sentiment_analysis
from src.correlation import run_correlation_analysis
from src.visualization import run_visualization
from src.data_collection.config import TARGET_STOCKS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(stocks=None, data_types=None, steps=None):
    """
    Run the complete stock sentiment analysis pipeline.
    
    Args:
        stocks (list): List of stock symbols to analyze
        data_types (list): List of data types to analyze
        steps (list): List of pipeline steps to run
        
    Returns:
        bool: Success flag
    """
    if stocks is None:
        stocks = TARGET_STOCKS
    
    if data_types is None:
        data_types = ['reddit', 'news']
    
    if steps is None:
        steps = ['collect', 'analyze', 'correlate', 'visualize']
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger.info("Starting Stock Price Sentiment Analysis Pipeline")
    logger.info(f"Analyzing stocks: {', '.join(stocks)}")
    logger.info(f"Using data sources: {', '.join(data_types)}")
    logger.info(f"Running steps: {', '.join(steps)}")
    
    start_time = datetime.now()
    
    try:
        # Step 1: Collect data
        if 'collect' in steps:
            logger.info("Step 1: Data Collection")
            collect_all_data()
            logger.info("Data collection completed")
        else:
            logger.info("Skipping data collection step")
        
        # Step 2: Sentiment analysis
        if 'analyze' in steps:
            logger.info("Step 2: Sentiment Analysis")
            run_sentiment_analysis(method='textblob', data_types=data_types, stocks=stocks)
            logger.info("Sentiment analysis completed")
        else:
            logger.info("Skipping sentiment analysis step")
        
        # Step 3: Correlation analysis
        if 'correlate' in steps:
            logger.info("Step 3: Correlation Analysis")
            run_correlation_analysis(stocks=stocks, data_types=data_types, max_lag_days=5)
            logger.info("Correlation analysis completed")
        else:
            logger.info("Skipping correlation analysis step")
        
        # Step 4: Visualization
        if 'visualize' in steps:
            logger.info("Step 4: Visualization")
            run_visualization(stocks=stocks)
            logger.info("Visualization completed")
        else:
            logger.info("Skipping visualization step")
        
        end_time = datetime.now()
        run_time = end_time - start_time
        
        logger.info(f"Pipeline completed successfully in {run_time}")
        logger.info("Find results in the 'data/processed' directory")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Stock Price Sentiment Analysis Pipeline')
    
    parser.add_argument('--stocks', type=str, nargs='+',
                       help='Stock symbols to analyze (default: all stocks in config)')
    
    parser.add_argument('--data-types', type=str, nargs='+', default=['reddit', 'news'],
                       help='Data types to analyze (default: reddit news)')
    
    parser.add_argument('--steps', type=str, nargs='+', 
                       default=['collect', 'analyze', 'correlate', 'visualize'],
                       choices=['collect', 'analyze', 'correlate', 'visualize', 'all'],
                       help='Pipeline steps to run (default: all steps)')
    
    args = parser.parse_args()
    
    # If 'all' is in steps, run all steps
    if 'all' in args.steps:
        args.steps = ['collect', 'analyze', 'correlate', 'visualize']
    
    return args

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(stocks=args.stocks, data_types=args.data_types, steps=args.steps) 