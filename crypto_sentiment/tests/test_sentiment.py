"""
Test cases for the sentiment analysis module.
"""
import unittest
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from ..sentiment_analysis.analyzer import SentimentAnalyzer
from ..analytics.analyzer import SentimentAnalytics
from ..utils.helpers import (
    clean_text,
    calculate_time_buckets,
    detect_bots,
    calculate_sentiment_metrics,
    detect_anomalies
)

class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for the SentimentAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()
        self.test_texts = [
            "Bitcoin is going to the moon! 🚀",
            "This is a scam, stay away from crypto.",
            "Just another day in the crypto market.",
            "HODL your coins, don't sell!",
            "FUD is spreading in the market."
        ]
    
    def test_analyze_text(self):
        """Test basic text analysis."""
        for text in self.test_texts:
            result = self.analyzer.analyze_text(text)
            
            # Check result structure
            self.assertIn("sentiment", result)
            self.assertIn("emotions", result)
            self.assertIn("crypto_analysis", result)
            
            # Check sentiment
            self.assertIn(result["sentiment"]["label"], ["positive", "negative", "neutral"])
            self.assertGreaterEqual(result["sentiment"]["score"], 0)
            self.assertLessEqual(result["sentiment"]["score"], 1)
    
    def test_analyze_batch(self):
        """Test batch analysis."""
        results = self.analyzer.analyze_batch(self.test_texts)
        
        # Check results
        self.assertEqual(len(results), len(self.test_texts))
        for result in results:
            self.assertIn("sentiment", result)
            self.assertIn("emotions", result)
            self.assertIn("crypto_analysis", result)
    
    def test_crypto_terms(self):
        """Test crypto-specific term analysis."""
        result = self.analyzer.analyze_text("HODL and moon! FUD is spreading.")
        
        # Check crypto terms
        self.assertIn("HODL", result["crypto_analysis"]["positive_terms"])
        self.assertIn("moon", result["crypto_analysis"]["positive_terms"])
        self.assertIn("FUD", result["crypto_analysis"]["negative_terms"])

class TestSentimentAnalytics(unittest.TestCase):
    """Test cases for the SentimentAnalytics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analytics = SentimentAnalytics()
        
        # Create test data
        dates = pd.date_range(start="2023-01-01", periods=100, freq="H")
        self.sentiment_data = pd.DataFrame({
            "timestamp": dates,
            "sentiment_score": np.random.normal(0.5, 0.2, 100)
        })
        
        self.price_data = pd.DataFrame({
            "timestamp": dates,
            "price": np.random.normal(100, 10, 100)
        })
    
    def test_analyze_correlation(self):
        """Test correlation analysis."""
        result = self.analytics.analyze_correlation(
            self.sentiment_data,
            self.price_data
        )
        
        # Check result structure
        self.assertIn("correlations", result)
        self.assertIn("best_lag", result)
        self.assertIn("min_correlation", result)
        
        # Check correlations
        self.assertGreater(len(result["correlations"]), 0)
        for corr in result["correlations"]:
            self.assertIn("lag", corr)
            self.assertIn("correlation", corr)
            self.assertGreaterEqual(corr["correlation"], -1)
            self.assertLessEqual(corr["correlation"], 1)
    
    def test_granger_causality(self):
        """Test Granger causality analysis."""
        result = self.analytics.granger_causality_test(
            self.sentiment_data,
            self.price_data
        )
        
        # Check result structure
        self.assertIn("significant_lags", result)
        self.assertIn("significance_level", result)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        result = self.analytics.detect_anomalies(self.sentiment_data)
        
        # Check result structure
        self.assertIn("anomalies", result)
        self.assertIn("total_anomalies", result)
        self.assertIn("threshold", result)
        
        # Check anomalies
        self.assertIsInstance(result["anomalies"], list)
        self.assertGreaterEqual(result["total_anomalies"], 0)

class TestHelpers(unittest.TestCase):
    """Test cases for helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = "Check out https://example.com Bitcoin is going to the moon! 🚀"
        self.test_data = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="H"),
            "text": ["test"] * 10,
            "author": ["user1"] * 10,
            "created_at": [datetime.now()] * 10,
            "account_created_at": [datetime.now() - timedelta(days=30)] * 10
        })
    
    def test_clean_text(self):
        """Test text cleaning."""
        cleaned = clean_text(self.test_text)
        
        # Check cleaning
        self.assertNotIn("https://", cleaned)
        self.assertEqual(cleaned, cleaned.lower())
        self.assertNotIn("  ", cleaned)  # No double spaces
    
    def test_calculate_time_buckets(self):
        """Test time bucket calculation."""
        result = calculate_time_buckets(
            self.test_data,
            "timestamp",
            "1H"
        )
        
        # Check time buckets
        self.assertIn("time_bucket", result.columns)
        self.assertEqual(len(result), len(self.test_data))
    
    def test_detect_bots(self):
        """Test bot detection."""
        metrics = {
            "max_posts_per_hour": 10,
            "min_content_similarity": 0.5,
            "min_account_age_days": 7
        }
        
        result = detect_bots(self.test_data, metrics)
        
        # Check bot detection
        self.assertIn("bot_probability", result.columns)
        self.assertIn("posts_per_hour", result.columns)
        self.assertIn("content_similarity", result.columns)
        self.assertIn("account_age_days", result.columns)
    
    def test_calculate_sentiment_metrics(self):
        """Test sentiment metrics calculation."""
        result = calculate_sentiment_metrics(
            self.test_data,
            "text",
            "timestamp",
            "1H"
        )
        
        # Check metrics
        self.assertIn("mean", result.columns)
        self.assertIn("std", result.columns)
        self.assertIn("min", result.columns)
        self.assertIn("max", result.columns)
        self.assertIn("count", result.columns)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        result = detect_anomalies(
            self.test_data,
            "text",
            window_size=3,
            threshold=2.0
        )
        
        # Check anomaly detection
        self.assertIn("is_anomaly", result.columns)
        self.assertIn("anomaly_score", result.columns)

if __name__ == "__main__":
    unittest.main() 