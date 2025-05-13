#!/usr/bin/env python3
"""
Stock Price Sentiment Analyzer - Main Runner Script

This script runs the complete stock sentiment analysis pipeline:
1. Data collection (Reddit, News API, Yahoo Finance)
2. Sentiment analysis
3. Correlation analysis between sentiment and stock movements
4. Data visualization and reporting
"""

import argparse
import logging
from datetime import datetime
import os

from src.main import run_pipeline, parse_arguments

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Set up logging
    log_file = os.path.join('logs', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Stock Price Sentiment Analyzer")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the pipeline
    success = run_pipeline(stocks=args.stocks, data_types=args.data_types, steps=args.steps)
    
    if success:
        logger.info("Pipeline execution completed successfully")
        print("\nPipeline execution completed successfully!")
        print("Check the 'data/processed' directory for analysis results.")
        print(f"Log file saved to: {log_file}")
    else:
        logger.error("Pipeline execution failed")
        print("\nPipeline execution failed.")
        print(f"Check the log file for details: {log_file}") 