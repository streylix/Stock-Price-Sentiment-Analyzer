import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import argparse
from datetime import datetime

from ..data_collection.config import TARGET_STOCKS, PROCESSED_DATA_PATH, RAW_DATA_PATH

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'visualization.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockSentimentVisualizer:
    def __init__(self):
        """Initialize the visualizer."""
        # Create output directory if it doesn't exist
        self.output_dir = os.path.join(PROCESSED_DATA_PATH, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Set default plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def load_stock_data(self, stock_symbol):
        """Load stock price data."""
        try:
            filename = os.path.join(RAW_DATA_PATH, 'stocks', f"{stock_symbol}_price_data.csv")
            df = pd.read_csv(filename)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.error(f"Error loading stock data for {stock_symbol}: {e}")
            return pd.DataFrame()
    
    def load_sentiment_summary(self):
        """Load sentiment analysis summary."""
        try:
            filename = os.path.join(PROCESSED_DATA_PATH, 'sentiment', 'sentiment_analysis_summary.csv')
            df = pd.read_csv(filename)
            return df
        except Exception as e:
            logger.error(f"Error loading sentiment summary: {e}")
            return pd.DataFrame()
    
    def load_correlation_results(self):
        """Load correlation analysis results."""
        try:
            filename = os.path.join(PROCESSED_DATA_PATH, 'correlation', 'all_correlations_summary.csv')
            df = pd.read_csv(filename)
            return df
        except Exception as e:
            logger.error(f"Error loading correlation results: {e}")
            return pd.DataFrame()
    
    def plot_stock_price_history(self, stock_symbol):
        """
        Plot stock price history with volume.
        
        Args:
            stock_symbol (str): Stock ticker symbol
            
        Returns:
            str: Path to saved figure
        """
        stock_data = self.load_stock_data(stock_symbol)
        
        if stock_data.empty:
            logger.warning(f"No stock data available for {stock_symbol}")
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        
        # Plot price
        ax1.plot(stock_data['Date'], stock_data['Close'], label='Close Price', color='blue')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f"{stock_symbol} Stock Price History")
        ax1.grid(True, alpha=0.3)
        
        # Add moving averages
        if 'MA5' in stock_data.columns:
            ax1.plot(stock_data['Date'], stock_data['MA5'], label='5-Day MA', color='orange', alpha=0.7)
        if 'MA20' in stock_data.columns:
            ax1.plot(stock_data['Date'], stock_data['MA20'], label='20-Day MA', color='green', alpha=0.7)
        
        ax1.legend()
        
        # Plot volume
        ax2.bar(stock_data['Date'], stock_data['Volume'], color='darkblue', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"{stock_symbol}_price_history.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved price history plot for {stock_symbol} to {output_file}")
        
        return output_file
    
    def plot_sentiment_distribution(self, stock_symbol=None):
        """
        Plot sentiment distribution across data sources.
        
        Args:
            stock_symbol (str): Stock ticker symbol (None for all stocks)
            
        Returns:
            str: Path to saved figure
        """
        sentiment_df = self.load_sentiment_summary()
        
        if sentiment_df.empty:
            logger.warning("No sentiment data available")
            return None
        
        # Filter by stock if specified
        if stock_symbol:
            sentiment_df = sentiment_df[sentiment_df['stock'] == stock_symbol]
            if sentiment_df.empty:
                logger.warning(f"No sentiment data available for {stock_symbol}")
                return None
        
        # Create stacked bar plot
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        plot_data = sentiment_df.copy()
        
        # Create unique labels for each stock and data type
        plot_data['label'] = plot_data['stock'] + ' (' + plot_data['data_type'] + ')'
        
        # Sort by positive ratio for better visualization
        plot_data = plot_data.sort_values('positive_ratio', ascending=False)
        
        # Create stacked bars
        ax = plt.subplot(111)
        
        # Plot stacked bars
        bar_width = 0.8
        labels = plot_data['label']
        
        # Create bars
        ax.bar(labels, plot_data['positive_ratio'], bar_width, 
               label='Positive', color='green', alpha=0.7)
        ax.bar(labels, plot_data['neutral_ratio'], bar_width, 
               bottom=plot_data['positive_ratio'], label='Neutral', color='gray', alpha=0.7)
        ax.bar(labels, plot_data['negative_ratio'], bar_width, 
               bottom=plot_data['positive_ratio'] + plot_data['neutral_ratio'], 
               label='Negative', color='red', alpha=0.7)
        
        # Add title and labels
        title = f"Sentiment Distribution for {stock_symbol}" if stock_symbol else "Sentiment Distribution Across Stocks"
        ax.set_title(title)
        ax.set_ylabel('Proportion')
        ax.set_ylim(0, 1.0)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add value labels on bars
        for i, label in enumerate(labels):
            positive = plot_data['positive_ratio'].iloc[i]
            neutral = plot_data['neutral_ratio'].iloc[i]
            negative = plot_data['negative_ratio'].iloc[i]
            
            # Add count information
            pos_count = plot_data['positive_count'].iloc[i]
            neu_count = plot_data['neutral_count'].iloc[i]
            neg_count = plot_data['negative_count'].iloc[i]
            
            # Only add labels if there's enough space
            if positive > 0.1:
                ax.text(i, positive/2, f"{positive:.2f}\n({pos_count})", 
                       ha='center', va='center', color='white', fontweight='bold')
            
            if neutral > 0.1:
                ax.text(i, positive + neutral/2, f"{neutral:.2f}\n({neu_count})", 
                       ha='center', va='center', color='black', fontweight='bold')
            
            if negative > 0.1:
                ax.text(i, positive + neutral + negative/2, f"{negative:.2f}\n({neg_count})", 
                       ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        filename_part = stock_symbol if stock_symbol else "all_stocks"
        output_file = os.path.join(self.output_dir, f"{filename_part}_sentiment_distribution.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sentiment distribution plot to {output_file}")
        
        return output_file
    
    def plot_best_correlations(self, top_n=10):
        """
        Plot top correlations.
        
        Args:
            top_n (int): Number of top correlations to plot
            
        Returns:
            str: Path to saved figure
        """
        corr_df = self.load_correlation_results()
        
        if corr_df.empty:
            logger.warning("No correlation data available")
            return None
        
        # Get strongest correlations (by absolute value)
        corr_df['abs_correlation'] = corr_df['correlation'].abs()
        
        # Filter for significant correlations
        significant_corr = corr_df[corr_df['significant']]
        
        if significant_corr.empty:
            logger.warning("No significant correlations available")
            return None
        
        # Get top N by absolute correlation
        top_corr = significant_corr.nlargest(top_n, 'abs_correlation')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)
        
        # Create labels for bars
        top_corr['label'] = top_corr['stock'] + ' (' + top_corr['data_type'] + ', ' + top_corr['metric_type'] + ', lag=' + top_corr['lag'].astype(str) + ')'
        
        # Sort by correlation value for better visualization
        top_corr = top_corr.sort_values('correlation')
        
        # Create horizontal bar chart
        bars = ax.barh(top_corr['label'], top_corr['correlation'], color=top_corr['correlation'].map(
            lambda x: 'green' if x > 0 else 'red'), alpha=0.7)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width + 0.01 if width > 0 else width - 0.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                   va='center', ha='left' if width > 0 else 'right')
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add title and labels
        ax.set_title(f"Top {top_n} Significant Correlations Between Sentiment and Stock Movements")
        ax.set_xlabel('Correlation Coefficient')
        
        # Add gridlines
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        output_file = os.path.join(self.output_dir, f"top_{top_n}_correlations.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved top correlations plot to {output_file}")
        
        return output_file
    
    def create_lag_heatmap(self):
        """
        Create a heatmap showing correlation strength by lag for each stock and data type.
        
        Returns:
            str: Path to saved figure
        """
        corr_df = self.load_correlation_results()
        
        if corr_df.empty:
            logger.warning("No correlation data available")
            return None
        
        # Filter for returns metric type and merge data types
        returns_corr = corr_df[corr_df['metric_type'] == 'returns']
        
        if returns_corr.empty:
            logger.warning("No returns correlation data available")
            return None
        
        # Filter for Reddit data
        reddit_corr = returns_corr[returns_corr['data_type'] == 'reddit']
        
        # Filter for news data
        news_corr = returns_corr[returns_corr['data_type'] == 'news']
        
        # Create pivot tables for heatmaps
        reddit_pivot = None
        if not reddit_corr.empty:
            reddit_pivot = reddit_corr.pivot_table(
                values='correlation', 
                index='stock', 
                columns='lag',
                aggfunc='mean'
            )
        
        news_pivot = None
        if not news_corr.empty:
            news_pivot = news_corr.pivot_table(
                values='correlation', 
                index='stock', 
                columns='lag',
                aggfunc='mean'
            )
        
        # Create plot
        if reddit_pivot is not None or news_pivot is not None:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            # Set up color map with centered colorbar
            cmap = sns.diverging_palette(240, 10, as_cmap=True)
            vmin = -0.5
            vmax = 0.5
            
            # Plot Reddit heatmap
            if reddit_pivot is not None:
                sns.heatmap(reddit_pivot, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax, 
                            center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
                axes[0].set_title('Reddit Sentiment Correlation with Daily Returns by Lag')
                axes[0].set_ylabel('Stock')
            else:
                axes[0].text(0.5, 0.5, 'No Reddit data available', ha='center', va='center')
                axes[0].set_title('Reddit Sentiment (No Data)')
            
            # Plot News heatmap
            if news_pivot is not None:
                sns.heatmap(news_pivot, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax, 
                            center=0, annot=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
                axes[1].set_title('News Sentiment Correlation with Daily Returns by Lag')
                axes[1].set_ylabel('Stock')
                axes[1].set_xlabel('Lag (Negative = Sentiment Leads Returns, Positive = Returns Lead Sentiment)')
            else:
                axes[1].text(0.5, 0.5, 'No News data available', ha='center', va='center')
                axes[1].set_title('News Sentiment (No Data)')
            
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.output_dir, "lag_correlation_heatmap.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved lag correlation heatmap to {output_file}")
            
            return output_file
        else:
            logger.warning("No data available for heatmap")
            return None
    
    def visualize_all(self, stocks=None):
        """
        Create all visualizations.
        
        Args:
            stocks (list): List of stock symbols to visualize
            
        Returns:
            list: Paths to all generated figures
        """
        if stocks is None:
            stocks = TARGET_STOCKS
        
        logger.info("Starting data visualization")
        output_files = []
        
        # Create individual stock price plots
        for stock in stocks:
            output_file = self.plot_stock_price_history(stock)
            if output_file:
                output_files.append(output_file)
        
        # Create sentiment distribution plots
        # Overall sentiment distribution
        output_file = self.plot_sentiment_distribution()
        if output_file:
            output_files.append(output_file)
        
        # Individual stock sentiment distributions
        for stock in stocks:
            output_file = self.plot_sentiment_distribution(stock)
            if output_file:
                output_files.append(output_file)
        
        # Create correlation plots
        output_file = self.plot_best_correlations()
        if output_file:
            output_files.append(output_file)
        
        # Create lag heatmap
        output_file = self.create_lag_heatmap()
        if output_file:
            output_files.append(output_file)
        
        logger.info(f"Data visualization completed. Created {len(output_files)} figures.")
        
        # Generate a report with all figures
        self.generate_html_report(output_files)
        
        return output_files
    
    def generate_html_report(self, figure_paths):
        """
        Generate an HTML report with all figures.
        
        Args:
            figure_paths (list): Paths to all generated figures
            
        Returns:
            str: Path to HTML report
        """
        if not figure_paths:
            logger.warning("No figures to include in the report")
            return None
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Price Sentiment Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .figure {{ margin: 20px 0; border: 1px solid #ddd; padding: 10px; }}
                .figure img {{ max-width: 100%; height: auto; }}
                .figure-caption {{ margin-top: 10px; font-style: italic; color: #555; }}
            </style>
        </head>
        <body>
            <h1>Stock Price Sentiment Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Analysis Results</h2>
        """
        
        # Add figures to the report
        for fig_path in figure_paths:
            if fig_path:
                # Get relative path for HTML
                rel_path = os.path.relpath(fig_path, start=PROCESSED_DATA_PATH)
                
                # Get figure name without extension
                fig_name = os.path.basename(fig_path).replace('.png', '').replace('_', ' ').title()
                
                # Add figure to HTML
                html_content += f"""
                <div class="figure">
                    <h3>{fig_name}</h3>
                    <img src="../{rel_path}" alt="{fig_name}">
                    <div class="figure-caption">Figure: {fig_name}</div>
                </div>
                """
        
        # Close HTML
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        report_file = os.path.join(self.output_dir, "sentiment_analysis_report.html")
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at {report_file}")
        
        return report_file

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize stock sentiment analysis results')
    
    parser.add_argument('--stocks', type=str, nargs='+',
                       help='Stock symbols to visualize (default: all stocks in config)')
    
    return parser.parse_args()

def run_visualization(stocks=None):
    """Run visualization"""
    visualizer = StockSentimentVisualizer()
    visualizer.visualize_all(stocks)

if __name__ == "__main__":
    args = parse_arguments()
    run_visualization(args.stocks) 