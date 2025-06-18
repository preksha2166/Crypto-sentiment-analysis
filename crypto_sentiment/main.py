"""
Main entry point for the cryptocurrency sentiment analysis system.
"""
import asyncio
import logging
from datetime import datetime
import pandas as pd
from pathlib import Path

from .config import (
    API_KEYS,
    COLLECTION_SETTINGS,
    SENTIMENT_SETTINGS,
    ANALYTICS_SETTINGS,
    DASHBOARD_SETTINGS,
    LOGGING_SETTINGS
)
from .data_collection.collectors import (
    TwitterCollector,
    RedditCollector,
    TelegramCollector,
    GitHubCollector
)
from .sentiment_analysis.analyzer import SentimentAnalyzer
from .analytics.analyzer import SentimentAnalytics
from .utils.helpers import (
    setup_logging,
    create_directory_structure,
    save_json_file
)

# Set up logging
setup_logging("logs/app.log")
logger = logging.getLogger(__name__)

class CryptoSentimentSystem:
    """Main system class that coordinates all components."""
    
    def __init__(self):
        """Initialize the system."""
        # Create necessary directories
        create_directory_structure(".")
        
        # Initialize components
        self.collectors = [
            TwitterCollector(),
            RedditCollector(),
            TelegramCollector(),
            GitHubCollector()
        ]
        self.sentiment_analyzer = SentimentAnalyzer()
        self.analytics = SentimentAnalytics()
        
        # Initialize data storage
        self.data = {
            "sentiment": pd.DataFrame(),
            "price": pd.DataFrame(),
            "last_update": None
        }
    
    async def collect_data(self):
        """Collect data from all sources."""
        all_data = []
        
        for collector in self.collectors:
            try:
                data = await collector.collect()
                if data:
                    all_data.extend(data)
                    logger.info(f"Collected {len(data)} items from {collector.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error collecting data from {collector.__class__.__name__}: {str(e)}")
        
        if all_data:
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Analyze sentiment
            analyses = self.sentiment_analyzer.analyze_batch(df["text"].tolist())
            df["sentiment_score"] = [a["sentiment"]["score"] for a in analyses]
            df["sentiment_label"] = [a["sentiment"]["label"] for a in analyses]
            
            self.data["sentiment"] = df
            self.data["last_update"] = datetime.now()
            
            # Save data
            self._save_data()
    
    def analyze_data(self):
        """Perform advanced analytics on collected data."""
        if self.data["sentiment"].empty:
            logger.warning("No data available for analysis")
            return
        
        try:
            # Calculate time buckets
            self.data["sentiment"] = self.data["sentiment"].assign(
                time_bucket=pd.to_datetime(self.data["sentiment"]["created_at"]).dt.floor("1H")
            )
            
            # Calculate sentiment metrics
            metrics = self.analytics.calculate_sentiment_metrics(
                self.data["sentiment"],
                "sentiment_score",
                "time_bucket"
            )
            
            # Detect anomalies
            anomalies = self.analytics.detect_anomalies(
                self.data["sentiment"],
                "sentiment_score"
            )
            
            # Save analysis results
            self._save_analysis_results(metrics, anomalies)
            
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
    
    def _save_data(self):
        """Save collected data to disk."""
        try:
            # Save sentiment data
            if not self.data["sentiment"].empty:
                self.data["sentiment"].to_csv(
                    "data/processed/sentiment_data.csv",
                    index=False
                )
            
            # Save price data
            if not self.data["price"].empty:
                self.data["price"].to_csv(
                    "data/processed/price_data.csv",
                    index=False
                )
            
            logger.info("Data saved successfully")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def _save_analysis_results(self, metrics: pd.DataFrame, anomalies: pd.DataFrame):
        """Save analysis results to disk."""
        try:
            # Save metrics
            metrics.to_csv("data/processed/sentiment_metrics.csv", index=False)
            
            # Save anomalies
            anomalies.to_csv("data/processed/anomalies.csv", index=False)
            
            logger.info("Analysis results saved successfully")
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")

async def main():
    """Main entry point."""
    try:
        # Initialize system
        system = CryptoSentimentSystem()
        
        # Collect data
        await system.collect_data()
        
        # Analyze data
        system.analyze_data()
        
        logger.info("System run completed successfully")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    # Run the system
    asyncio.run(main()) 