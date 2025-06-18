"""
Configuration settings for the Crypto Sentiment Analysis System.
"""
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# API Keys (to be loaded from environment variables)
API_KEYS = {
    "twitter": os.getenv("TWITTER_API_KEY"),
    "reddit": {
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
        "user_agent": "CryptoSentimentAnalysis/1.0"
    },
    "telegram": os.getenv("TELEGRAM_API_KEY"),
    "discord": os.getenv("DISCORD_API_KEY"),
    "github": os.getenv("GITHUB_API_KEY"),
    "etherscan": os.getenv("ETHERSCAN_API_KEY"),
}

# Data Collection Settings
COLLECTION_SETTINGS = {
    "twitter": {
        "track_keywords": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft"],
        "languages": ["en"],
        "max_tweets_per_request": 100,
        "rate_limit_pause": 60,  # seconds
    },
    "reddit": {
        "subreddits": ["CryptoCurrency", "CryptoMarkets", "Bitcoin", "Ethereum"],
        "max_posts_per_request": 100,
        "rate_limit_pause": 60,
    },
    "telegram": {
        "channels": ["binanceupdates", "cryptosignals"],
        "max_messages_per_request": 100,
    },
    "github": {
        "repositories": [
            "bitcoin/bitcoin",
            "ethereum/go-ethereum",
            "binance-chain/bsc",
        ],
        "max_commits_per_request": 100,
    }
}

# Sentiment Analysis Settings
SENTIMENT_SETTINGS = {
    "model": {
        "name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "epochs": 3,
    },
    "emotions": [
        "fear", "excitement", "optimism", "uncertainty", "anger",
        "trust", "surprise", "disgust", "joy", "sadness"
    ],
    "crypto_specific_terms": {
        "positive": ["HODL", "moon", "bullish", "diamond hands", "to the moon"],
        "negative": ["FUD", "rekt", "bearish", "dump", "scam"],
        "neutral": ["DYOR", "NGU", "NGMI", "wagmi"],
    }
}

# Analytics Settings
ANALYTICS_SETTINGS = {
    "correlation": {
        "max_lag": 24,  # hours
        "min_correlation": 0.3,
    },
    "granger": {
        "max_lag": 12,
        "significance_level": 0.05,
    },
    "anomaly_detection": {
        "window_size": 24,  # hours
        "threshold": 2.0,  # standard deviations
    }
}

# Dashboard Settings
DASHBOARD_SETTINGS = {
    "update_interval": 60,  # seconds
    "max_data_points": 1000,
    "default_time_range": "24h",
    "available_time_ranges": ["1h", "24h", "7d", "30d", "90d"],
}

# Database Settings
DATABASE_SETTINGS = {
    "url": os.getenv("DATABASE_URL", "sqlite:///crypto_sentiment.db"),
    "pool_size": 5,
    "max_overflow": 10,
}

# Logging Settings
LOGGING_SETTINGS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
} 