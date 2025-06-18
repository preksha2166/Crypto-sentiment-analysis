import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats
from sklearn.linear_model import LinearRegression

class TrendAnalyzer:
    def __init__(self):
        """Initialize the trend analyzer."""
        pass
        
    def calculate_trend(self, series: pd.Series) -> Dict[str, float]:
        """
        Calculate linear trend of a series.
        
        Args:
            series (pd.Series): Series to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing trend parameters
        """
        x = np.arange(len(series)).reshape(-1, 1)
        y = series.values.reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return {
            'slope': model.coef_[0][0],
            'intercept': model.intercept_[0],
            'r2_score': model.score(x, y)
        }
    
    def calculate_moving_trend(self, series: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate moving trend of a series.
        
        Args:
            series (pd.Series): Series to analyze
            window (int): Window size
            
        Returns:
            pd.Series: Series of trend slopes
        """
        slopes = []
        for i in range(len(series) - window + 1):
            window_data = series.iloc[i:i+window]
            trend = self.calculate_trend(window_data)
            slopes.append(trend['slope'])
        
        return pd.Series(slopes, index=series.index[window-1:])
    
    def detect_trend_changes(self, series: pd.Series, 
                           window: int = 20, 
                           threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect significant trend changes in a series.
        
        Args:
            series (pd.Series): Series to analyze
            window (int): Window size
            threshold (float): Threshold for significant change
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing trend change points
        """
        moving_trend = self.calculate_moving_trend(series, window)
        changes = []
        
        for i in range(1, len(moving_trend)):
            if abs(moving_trend.iloc[i] - moving_trend.iloc[i-1]) > threshold:
                changes.append({
                    'index': moving_trend.index[i],
                    'value': series.iloc[i],
                    'trend_before': moving_trend.iloc[i-1],
                    'trend_after': moving_trend.iloc[i],
                    'change_magnitude': moving_trend.iloc[i] - moving_trend.iloc[i-1]
                })
        
        return changes
    
    def calculate_seasonality(self, series: pd.Series, period: int) -> Dict[str, Any]:
        """
        Calculate seasonality of a series.
        
        Args:
            series (pd.Series): Series to analyze
            period (int): Period of seasonality
            
        Returns:
            Dict[str, Any]: Dictionary containing seasonality analysis results
        """
        # Calculate seasonal components
        seasonal_components = []
        for i in range(period):
            seasonal_components.append(series[i::period].mean())
        
        # Calculate seasonal indices
        seasonal_indices = np.array(seasonal_components) / np.mean(seasonal_components)
        
        return {
            'seasonal_components': seasonal_components,
            'seasonal_indices': seasonal_indices,
            'strength': 1 - np.var(series - np.repeat(seasonal_components, len(series)//period + 1)[:len(series)]) / np.var(series)
        }
    
    def decompose_trend(self, series: pd.Series, period: int = None) -> Dict[str, pd.Series]:
        """
        Decompose a series into trend, seasonal, and residual components.
        
        Args:
            series (pd.Series): Series to decompose
            period (int): Period of seasonality (if known)
            
        Returns:
            Dict[str, pd.Series]: Dictionary containing decomposed components
        """
        # Calculate trend
        trend = self.calculate_moving_trend(series, window=len(series)//10)
        
        # Calculate seasonal component if period is provided
        if period:
            seasonal = self.calculate_seasonality(series, period)
            seasonal_series = pd.Series(
                np.repeat(seasonal['seasonal_components'], len(series)//period + 1)[:len(series)],
                index=series.index
            )
        else:
            seasonal_series = pd.Series(0, index=series.index)
        
        # Calculate residual
        residual = series - trend - seasonal_series
        
        return {
            'trend': trend,
            'seasonal': seasonal_series,
            'residual': residual
        }
    
    def get_trend_summary(self, series: pd.Series) -> Dict[str, Any]:
        """
        Get a summary of trend analysis.
        
        Args:
            series (pd.Series): Series to analyze
            
        Returns:
            Dict[str, Any]: Dictionary containing trend analysis summary
        """
        # Calculate overall trend
        overall_trend = self.calculate_trend(series)
        
        # Detect trend changes
        changes = self.detect_trend_changes(series)
        
        # Decompose series
        decomposition = self.decompose_trend(series)
        
        return {
            'overall_trend': overall_trend,
            'trend_changes': changes,
            'decomposition': decomposition,
            'trend_strength': abs(overall_trend['slope']) / series.std(),
            'trend_direction': 'upward' if overall_trend['slope'] > 0 else 'downward',
            'trend_significance': overall_trend['r2_score']
        } 