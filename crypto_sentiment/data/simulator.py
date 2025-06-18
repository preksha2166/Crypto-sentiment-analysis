"""
Data simulator for generating realistic mock data for testing.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta

class DataSimulator:
    """Class for generating realistic mock data."""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """Initialize the data simulator."""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        # Base prices for different cryptocurrencies
        self.base_prices = {
            'BTC': 50000.0,
            'ETH': 3000.0,
            'BNB': 400.0,
            'SOL': 100.0
        }
        
        # Volatility levels for different cryptocurrencies
        self.volatility_levels = {
            'BTC': 0.02,  # 2% daily volatility
            'ETH': 0.025,  # 2.5% daily volatility
            'BNB': 0.03,  # 3% daily volatility
            'SOL': 0.035  # 3.5% daily volatility
        }
        
        # Correlation levels for different cryptocurrencies
        self.correlation_levels = {
            'BTC': 0.3,  # 30% correlation with sentiment
            'ETH': 0.25,  # 25% correlation with sentiment
            'BNB': 0.2,  # 20% correlation with sentiment
            'SOL': 0.15  # 15% correlation with sentiment
        }
    
    def generate_price_data(self, coin: str, base_price: float = None, volatility: float = None) -> pd.Series:
        """Generate realistic price data with random walk."""
        if base_price is None:
            base_price = self.base_prices.get(coin, 100.0)
        if volatility is None:
            volatility = self.volatility_levels.get(coin, 0.02)
            
        # Use coin-specific seed for reproducibility
        np.random.seed(hash(coin) % 2**32)
        
        # Generate returns with some autocorrelation
        returns = np.random.normal(0, volatility, len(self.date_range))
        for i in range(1, len(returns)):
            returns[i] = 0.7 * returns[i-1] + 0.3 * returns[i]
        
        # Add some trend component
        trend = np.linspace(0, np.random.normal(0, volatility*2), len(self.date_range))
        returns = returns + trend
        
        # Generate price series
        price_series = base_price * (1 + returns).cumprod()
        
        # Add some periodic components
        days = np.arange(len(self.date_range))
        weekly = 0.1 * np.sin(2 * np.pi * days / 7)
        monthly = 0.05 * np.sin(2 * np.pi * days / 30)
        price_series = price_series * (1 + weekly + monthly)
        
        return pd.Series(price_series, index=self.date_range, name=f'{coin}_price')
    
    def generate_sentiment_data(self, coin: str, price_data: pd.Series, 
                              correlation: float = None,
                              noise_level: float = 0.5) -> pd.Series:
        """Generate sentiment data with some correlation to price."""
        if correlation is None:
            correlation = self.correlation_levels.get(coin, 0.3)
            
        # Use coin-specific seed for reproducibility
        np.random.seed(hash(coin + '_sentiment') % 2**32)
        
        # Generate base sentiment with some correlation to price
        price_normalized = (price_data - price_data.mean()) / price_data.std()
        base_sentiment = correlation * price_normalized + \
                        (1 - correlation) * np.random.normal(0, 1, len(price_data))
        
        # Add some random spikes and noise
        noise = np.random.normal(0, noise_level, len(price_data))
        sentiment = base_sentiment + noise
        
        # Add some periodic components
        days = np.arange(len(price_data))
        weekly = 0.2 * np.sin(2 * np.pi * days / 7)
        monthly = 0.1 * np.sin(2 * np.pi * days / 30)
        sentiment = sentiment + weekly + monthly
        
        # Add some sentiment shocks
        shock_days = np.random.choice(len(sentiment), size=3, replace=False)
        for day in shock_days:
            sentiment[day] += np.random.choice([-1, 1]) * np.random.uniform(0.5, 1.0)
        
        # Add some trend components
        trend = np.linspace(0, np.random.normal(0, 0.5), len(sentiment))
        sentiment = sentiment + trend
        
        # Add some market-specific sentiment factors
        if coin == 'BTC':
            # Bitcoin tends to have more extreme sentiment
            sentiment = sentiment * 1.2
        elif coin == 'ETH':
            # Ethereum sentiment is more stable
            sentiment = sentiment * 0.8
        elif coin == 'BNB':
            # BNB sentiment is more volatile
            sentiment = sentiment * 1.5
        elif coin == 'SOL':
            # Solana sentiment is more cyclical
            sentiment = sentiment + 0.3 * np.sin(2 * np.pi * days / 14)
        
        # Normalize to [-1, 1] range
        sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min()) * 2 - 1
        
        return pd.Series(sentiment, index=price_data.index, name=f'{coin}_sentiment')
    
    def generate_sample_tweets(self, coin: str, date: datetime, sentiment: float) -> List[Dict[str, Any]]:
        """Generate sample tweets for a given date and sentiment."""
        if sentiment > 0.3:
            tweets = [
                {"text": f"🚀 ${coin} looking bullish! Great technical setup!", "sentiment": 0.8},
                {"text": f"Just bought more {coin}, feeling confident about the market", "sentiment": 0.6},
                {"text": f"Strong fundamentals for {coin}, this is just the beginning", "sentiment": 0.7}
            ]
        elif sentiment < -0.3:
            tweets = [
                {"text": f"😱 {coin} market looking scary, might be time to exit", "sentiment": -0.7},
                {"text": f"Not a good time to buy {coin}, wait for better entry", "sentiment": -0.5},
                {"text": f"Too much uncertainty in the {coin} market right now", "sentiment": -0.6}
            ]
        else:
            tweets = [
                {"text": f"{coin} market seems stable, waiting for clear direction", "sentiment": 0.1},
                {"text": f"No strong opinion on {coin}, just watching the charts", "sentiment": -0.1},
                {"text": f"Mixed signals for {coin}, staying on the sidelines", "sentiment": 0.0}
            ]
        
        return tweets
    
    def calculate_metrics(self, price_data: pd.Series, sentiment_data: pd.Series) -> Dict:
        """Calculate key metrics from price and sentiment data."""
        # Calculate price metrics
        price_volatility = price_data.pct_change().std()
        price_volatility_change = price_volatility - price_data.pct_change().std()  # Change from previous period
        
        # Calculate sentiment metrics
        sentiment_volatility = sentiment_data.std()
        sentiment_volatility_change = sentiment_volatility - sentiment_data.std()  # Change from previous period
        
        # Calculate correlation
        correlation = price_data.corr(sentiment_data)
        correlation_change = correlation - price_data.corr(sentiment_data)  # Change from previous period
        
        # Find most volatile day
        daily_volatility = price_data.pct_change().abs()
        most_volatile_day = daily_volatility.idxmax()
        most_volatile_day_change = daily_volatility[most_volatile_day] - daily_volatility.mean()
        
        return {
            'price_volatility': price_volatility,
            'price_volatility_change': price_volatility_change,
            'sentiment_volatility': sentiment_volatility,
            'sentiment_volatility_change': sentiment_volatility_change,
            'correlation': correlation,
            'correlation_change': correlation_change,
            'most_volatile_day': most_volatile_day,
            'most_volatile_day_change': most_volatile_day_change
        }
    
    def generate_cross_correlation(self, price_data: pd.Series, 
                                 sentiment_data: pd.Series,
                                 max_lag: int = 7) -> pd.Series:
        """Calculate cross-correlation between price and sentiment."""
        correlations = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = price_data.corr(sentiment_data.shift(-lag))
            else:
                corr = price_data.corr(sentiment_data.shift(lag))
            correlations.append(corr)
        
        return pd.Series(correlations, index=range(-max_lag, max_lag + 1))
    
    def generate_insights(self, coin: str, price_data: pd.Series, 
                         sentiment_data: pd.Series) -> List[str]:
        """Generate insights from the data."""
        insights = []
        
        # Check for sentiment leading price
        sentiment_lead = self.generate_cross_correlation(price_data, sentiment_data)
        max_corr_lag = sentiment_lead.idxmax()
        if max_corr_lag > 0:
            insights.append(f"{coin} sentiment leads price by {max_corr_lag} days (correlation: {sentiment_lead[max_corr_lag]:.2f})")
        
        # Check for sentiment spikes
        sentiment_std = sentiment_data.std()
        recent_sentiment = sentiment_data.tail(5)
        if (recent_sentiment > 2 * sentiment_std).any():
            insights.append(f"Recent strong positive sentiment detected for {coin}")
        elif (recent_sentiment < -2 * sentiment_std).any():
            insights.append(f"Recent strong negative sentiment detected for {coin}")
        
        # Check for volatility
        price_volatility = price_data.pct_change().std()
        if price_volatility > 0.03:  # 3% daily volatility threshold
            insights.append(f"High price volatility detected for {coin}")
        
        return insights 