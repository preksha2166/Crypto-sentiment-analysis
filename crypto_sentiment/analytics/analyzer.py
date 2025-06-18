"""
Advanced analytics module for cryptocurrency sentiment analysis.
"""
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from sklearn.ensemble import IsolationForest
from ..config import ANALYTICS_SETTINGS

logger = logging.getLogger(__name__)

class SentimentAnalytics:
    """Advanced analytics for sentiment data."""
    
    def __init__(self):
        self.settings = ANALYTICS_SETTINGS
    
    def analyze_correlation(
        self,
        sentiment_data: pd.DataFrame,
        price_data: pd.DataFrame,
        max_lag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze time-lagged correlation between sentiment and price data.
        
        Args:
            sentiment_data: DataFrame with sentiment metrics
            price_data: DataFrame with price data
            max_lag: Maximum lag to consider (in hours)
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            max_lag = max_lag or self.settings["correlation"]["max_lag"]
            correlations = []
            
            # Ensure data is aligned by timestamp
            merged_data = pd.merge(
                sentiment_data,
                price_data,
                on="timestamp",
                how="inner"
            )
            
            # Calculate correlations for different lags
            for lag in range(max_lag + 1):
                if lag > 0:
                    sentiment_shifted = merged_data["sentiment_score"].shift(lag)
                else:
                    sentiment_shifted = merged_data["sentiment_score"]
                
                correlation = sentiment_shifted.corr(merged_data["price"])
                correlations.append({
                    "lag": lag,
                    "correlation": correlation
                })
            
            # Find best lag
            best_lag = max(correlations, key=lambda x: abs(x["correlation"]))
            
            return {
                "correlations": correlations,
                "best_lag": best_lag,
                "min_correlation": self.settings["correlation"]["min_correlation"]
            }
        except Exception as e:
            logger.error(f"Error in correlation analysis: {str(e)}")
            return {}
    
    def granger_causality_test(
        self,
        sentiment_data: pd.DataFrame,
        price_data: pd.DataFrame,
        max_lag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Granger causality test between sentiment and price data.
        
        Args:
            sentiment_data: DataFrame with sentiment metrics
            price_data: DataFrame with price data
            max_lag: Maximum lag to consider
            
        Returns:
            Dictionary containing Granger causality test results
        """
        try:
            max_lag = max_lag or self.settings["granger"]["max_lag"]
            
            # Prepare data for Granger test
            merged_data = pd.merge(
                sentiment_data,
                price_data,
                on="timestamp",
                how="inner"
            )
            
            # Perform Granger causality test
            data = np.column_stack((
                merged_data["sentiment_score"].values,
                merged_data["price"].values
            ))
            
            results = grangercausalitytests(
                data,
                maxlag=max_lag,
                verbose=False
            )
            
            # Extract significant results
            significant_lags = []
            for lag, result in results.items():
                p_value = result[0]["ssr_chi2test"][1]
                if p_value < self.settings["granger"]["significance_level"]:
                    significant_lags.append({
                        "lag": lag,
                        "p_value": p_value
                    })
            
            return {
                "significant_lags": significant_lags,
                "significance_level": self.settings["granger"]["significance_level"]
            }
        except Exception as e:
            logger.error(f"Error in Granger causality test: {str(e)}")
            return {}
    
    def detect_anomalies(
        self,
        sentiment_data: pd.DataFrame,
        window_size: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Detect anomalies in sentiment data using Isolation Forest.
        
        Args:
            sentiment_data: DataFrame with sentiment metrics
            window_size: Size of the rolling window
            threshold: Anomaly threshold
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            window_size = window_size or self.settings["anomaly_detection"]["window_size"]
            threshold = threshold or self.settings["anomaly_detection"]["threshold"]
            
            # Calculate rolling statistics
            rolling_mean = sentiment_data["sentiment_score"].rolling(
                window=window_size
            ).mean()
            rolling_std = sentiment_data["sentiment_score"].rolling(
                window=window_size
            ).std()
            
            # Calculate z-scores
            z_scores = (sentiment_data["sentiment_score"] - rolling_mean) / rolling_std
            
            # Detect anomalies
            anomalies = sentiment_data[abs(z_scores) > threshold].copy()
            anomalies["z_score"] = z_scores[abs(z_scores) > threshold]
            
            return {
                "anomalies": anomalies.to_dict(orient="records"),
                "total_anomalies": len(anomalies),
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {}
    
    def analyze_sentiment_trends(
        self,
        sentiment_data: pd.DataFrame,
        window_size: int = 24
    ) -> Dict[str, Any]:
        """
        Analyze sentiment trends over time.
        
        Args:
            sentiment_data: DataFrame with sentiment metrics
            window_size: Size of the rolling window (in hours)
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            # Calculate rolling statistics
            rolling_stats = sentiment_data["sentiment_score"].rolling(
                window=window_size
            ).agg(["mean", "std", "min", "max"])
            
            # Calculate trend direction
            trend = np.polyfit(
                range(len(sentiment_data)),
                sentiment_data["sentiment_score"],
                1
            )[0]
            
            return {
                "rolling_stats": rolling_stats.to_dict(),
                "trend_direction": "increasing" if trend > 0 else "decreasing",
                "trend_strength": abs(trend)
            }
        except Exception as e:
            logger.error(f"Error in trend analysis: {str(e)}")
            return {}
    
    def detect_sentiment_events(
        self,
        sentiment_data: pd.DataFrame,
        price_data: pd.DataFrame,
        threshold: float = 2.0
    ) -> Dict[str, Any]:
        """
        Detect significant sentiment events and their impact on price.
        
        Args:
            sentiment_data: DataFrame with sentiment metrics
            price_data: DataFrame with price data
            threshold: Threshold for event detection
            
        Returns:
            Dictionary containing event detection results
        """
        try:
            # Calculate sentiment changes
            sentiment_changes = sentiment_data["sentiment_score"].diff()
            
            # Detect significant changes
            significant_changes = sentiment_data[abs(sentiment_changes) > threshold].copy()
            significant_changes["change"] = sentiment_changes[abs(sentiment_changes) > threshold]
            
            # Analyze price impact
            events = []
            for idx, row in significant_changes.iterrows():
                # Get price data around the event
                event_time = row.name
                price_window = price_data.loc[
                    event_time - pd.Timedelta(hours=24):
                    event_time + pd.Timedelta(hours=24)
                ]
                
                if not price_window.empty:
                    price_change = (
                        price_window["price"].iloc[-1] -
                        price_window["price"].iloc[0]
                    ) / price_window["price"].iloc[0] * 100
                    
                    events.append({
                        "timestamp": event_time,
                        "sentiment_change": row["change"],
                        "price_change": price_change,
                        "sentiment_score": row["sentiment_score"]
                    })
            
            return {
                "events": events,
                "total_events": len(events),
                "threshold": threshold
            }
        except Exception as e:
            logger.error(f"Error in event detection: {str(e)}")
            return {} 