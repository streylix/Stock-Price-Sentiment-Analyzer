# Stock Price Sentiment Analyzer

An AI/ML project that analyzes sentiment trends from social media and news sources and correlates them with stock price movements.

## Project Overview

This project uses Natural Language Processing (NLP) to analyze sentiment from Reddit, Twitter, or news articles and correlates these sentiment trends with stock price movements to discover potential relationships between public sentiment and market performance.

## Features

- Data collection from social media platforms and financial news
- Stock price data retrieval using Yahoo Finance
- Text preprocessing and sentiment analysis using NLP techniques
- Time series correlation analysis between sentiment and stock prices
- Visualization and reporting of findings

## Project Structure

```
stock/
├── data/                # Data storage
│   ├── raw/             # Raw collected data
│   └── processed/       # Processed and cleaned data
├── notebooks/           # Jupyter notebooks for exploration and analysis
├── src/                 # Source code
│   ├── data_collection/ # Scripts for collecting social media and stock data
│   ├── sentiment_analysis/ # Sentiment analysis models and utilities
│   ├── correlation/     # Correlation analysis between sentiment and prices
│   └── visualization/   # Visualization and reporting tools
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys for data sources (see below)

## .env File Configuration

Create a `.env` file in the project root with the following variables:

```
# Reddit API credentials (required)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent

# Twitter API credentials (optional, not yet used)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret

# News API credentials (required)
NEWS_API_KEY=your_newsapi_key

# Stocks to track (optional, comma-separated, defaults to AAPL,MSFT,NVDA,GOOGL,AMZN)
TARGET_STOCKS=AAPL,MSFT,NVDA,GOOGL,AMZN

# Time period for analysis in days (optional, defaults to 30)
LOOKBACK_PERIOD=30
```

- **REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT**: Get these from your Reddit app settings at https://www.reddit.com/prefs/apps
- **NEWS_API_KEY**: Get this from https://newsapi.org/
- **TWITTER_API_***: For future use (not required for current pipeline)
- **TARGET_STOCKS**: Comma-separated list of stock tickers to analyze
- **LOOKBACK_PERIOD**: Number of days to look back for data collection

## Usage

1. Collect data:
   ```
   python src/data_collection/main.py
   ```
2. Run sentiment analysis:
   ```
   python src/sentiment_analysis/analyze.py
   ```
3. Perform correlation analysis:
   ```
   python src/correlation/analyze.py
   ```
4. Generate visualizations:
   ```
   python src/visualization/visualize.py
   ```

## License

MIT 