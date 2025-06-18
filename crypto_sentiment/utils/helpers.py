"""
Utility functions for the cryptocurrency sentiment analysis system.
"""
import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def load_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {str(e)}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {str(e)}")
        return False

def clean_text(text: str) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith(('http://', 'https://')))
    
    # Remove special characters and extra whitespace
    text = ' '.join(text.split())
    
    return text

def calculate_time_buckets(
    data: pd.DataFrame,
    time_column: str,
    bucket_size: str = '1H'
) -> pd.DataFrame:
    """
    Calculate time-based buckets for data aggregation.
    
    Args:
        data: Input DataFrame
        time_column: Name of the time column
        bucket_size: Size of time buckets (e.g., '1H' for 1 hour)
        
    Returns:
        DataFrame with time buckets
    """
    try:
        # Ensure time column is datetime
        data[time_column] = pd.to_datetime(data[time_column])
        
        # Create time buckets
        data['time_bucket'] = data[time_column].dt.floor(bucket_size)
        
        return data
    except Exception as e:
        logger.error(f"Error calculating time buckets: {str(e)}")
        return data

def detect_bots(
    data: pd.DataFrame,
    metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Detect potential bot accounts based on activity metrics.
    
    Args:
        data: DataFrame containing user data
        metrics: Dictionary of metric thresholds
        
    Returns:
        DataFrame with bot detection results
    """
    try:
        # Calculate bot probability based on metrics
        data['bot_probability'] = 0.0
        
        # Check post frequency
        if 'max_posts_per_hour' in metrics:
            data['posts_per_hour'] = data.groupby('author')['created_at'].transform(
                lambda x: len(x) / ((x.max() - x.min()).total_seconds() / 3600)
            )
            data.loc[data['posts_per_hour'] > metrics['max_posts_per_hour'], 'bot_probability'] += 0.3
        
        # Check content similarity
        if 'min_content_similarity' in metrics:
            data['content_similarity'] = data.groupby('author')['text'].transform(
                lambda x: x.str.len().std() / x.str.len().mean() if len(x) > 1 else 0
            )
            data.loc[data['content_similarity'] < metrics['min_content_similarity'], 'bot_probability'] += 0.3
        
        # Check account age
        if 'min_account_age_days' in metrics:
            data['account_age_days'] = (
                datetime.now() - pd.to_datetime(data['account_created_at'])
            ).dt.total_seconds() / (24 * 3600)
            data.loc[data['account_age_days'] < metrics['min_account_age_days'], 'bot_probability'] += 0.4
        
        return data
    except Exception as e:
        logger.error(f"Error detecting bots: {str(e)}")
        return data

def calculate_sentiment_metrics(
    data: pd.DataFrame,
    sentiment_column: str,
    time_column: str,
    window_size: str = '1H'
) -> pd.DataFrame:
    """
    Calculate sentiment metrics over time windows.
    
    Args:
        data: DataFrame containing sentiment data
        sentiment_column: Name of the sentiment column
        time_column: Name of the time column
        window_size: Size of time windows
        
    Returns:
        DataFrame with sentiment metrics
    """
    try:
        # Ensure time column is datetime
        data[time_column] = pd.to_datetime(data[time_column])
        
        # Calculate rolling metrics
        metrics = data.set_index(time_column)[sentiment_column].rolling(
            window=window_size,
            min_periods=1
        ).agg(['mean', 'std', 'min', 'max', 'count'])
        
        # Reset index to keep time column
        metrics = metrics.reset_index()
        
        return metrics
    except Exception as e:
        logger.error(f"Error calculating sentiment metrics: {str(e)}")
        return pd.DataFrame()

def detect_anomalies(
    data: pd.DataFrame,
    column: str,
    window_size: int = 24,
    threshold: float = 2.0
) -> pd.DataFrame:
    """
    Detect anomalies in time series data.
    
    Args:
        data: DataFrame containing time series data
        column: Name of the column to analyze
        window_size: Size of the rolling window
        threshold: Number of standard deviations for anomaly detection
        
    Returns:
        DataFrame with anomaly detection results
    """
    try:
        # Calculate rolling statistics
        rolling_mean = data[column].rolling(window=window_size).mean()
        rolling_std = data[column].rolling(window=window_size).std()
        
        # Calculate z-scores
        z_scores = (data[column] - rolling_mean) / rolling_std
        
        # Detect anomalies
        data['is_anomaly'] = abs(z_scores) > threshold
        data['anomaly_score'] = abs(z_scores)
        
        return data
    except Exception as e:
        logger.error(f"Error detecting anomalies: {str(e)}")
        return data

def create_directory_structure(base_path: str) -> None:
    """
    Create necessary directory structure for the project.
    
    Args:
        base_path: Base path for the project
    """
    try:
        directories = [
            'data/raw',
            'data/processed',
            'models',
            'logs',
            'cache'
        ]
        
        for directory in directories:
            Path(base_path).joinpath(directory).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory structure: {str(e)}")

def setup_logging(log_file: str) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to the log file
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        # Fallback to basic console logging
        logging.basicConfig(level=logging.INFO) 