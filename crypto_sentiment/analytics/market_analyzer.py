"""
Market analyzer for analyzing cryptocurrency market data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple

class MarketAnalyzer:
    """Class for analyzing market data."""
    
    def __init__(self):
        """Initialize the market analyzer."""
        pass
    
    def calculate_moving_averages(self, prices: pd.Series, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
        """Calculate moving averages for different windows."""
        ma_df = pd.DataFrame(index=prices.index)
        
        for window in windows:
            ma_df[f'MA_{window}'] = prices.rolling(window=window).mean()
        
        return ma_df
    
    def calculate_volatility(self, prices: pd.Series, window: int = 30) -> pd.Series:
        """Calculate rolling volatility."""
        returns = prices.pct_change()
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def calculate_momentum(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate momentum indicator."""
        return prices / prices.shift(window) - 1
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        return pd.DataFrame({
            'middle_band': ma,
            'upper_band': upper_band,
            'lower_band': lower_band
        })
    
    def calculate_support_resistance(self, prices: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate support and resistance levels."""
        rolling_min = prices.rolling(window=window).min()
        rolling_max = prices.rolling(window=window).max()
        
        return rolling_min, rolling_max
    
    def calculate_volume_profile(self, prices: pd.Series, volume: pd.Series, bins: int = 50) -> pd.DataFrame:
        """Calculate volume profile."""
        price_bins = pd.qcut(prices, bins)
        volume_profile = volume.groupby(price_bins).sum()
        
        return pd.DataFrame({
            'price_level': volume_profile.index.astype(str),
            'volume': volume_profile.values
        })
    
    def get_market_summary(self, prices: pd.Series, volume: pd.Series = None) -> Dict:
        """Get a summary of market metrics."""
        returns = prices.pct_change()
        
        summary = {
            'current_price': prices.iloc[-1],
            'price_change_24h': returns.iloc[-1] * 100,
            'price_change_7d': (prices.iloc[-1] / prices.iloc[-8] - 1) * 100,
            'price_change_30d': (prices.iloc[-1] / prices.iloc[-31] - 1) * 100,
            'volatility_30d': returns.std() * np.sqrt(252) * 100,
            'rsi_14': self.calculate_rsi(prices).iloc[-1],
            'momentum_14': self.calculate_momentum(prices).iloc[-1] * 100
        }
        
        if volume is not None:
            summary.update({
                'volume_24h': volume.iloc[-1],
                'volume_change_24h': (volume.iloc[-1] / volume.iloc[-2] - 1) * 100
            })
        
        return summary 