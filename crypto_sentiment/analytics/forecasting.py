"""
Forecasting module for time series prediction.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ForecastingModel:
    """Class for time series forecasting."""
    
    def __init__(self):
        """Initialize the forecasting model."""
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'arima': None,  # Will be initialized when needed
            'sarima': None  # Will be initialized when needed
        }
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.Series, n_lags: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for forecasting."""
        # Create lag features
        df = pd.DataFrame({'value': data})  # Use a default column name
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['value'].shift(i)
        
        # Drop NaN values
        df = df.dropna()
        
        # Prepare X and y
        X = df[[f'lag_{i}' for i in range(1, n_lags + 1)]].values
        y = df['value'].values
        
        return X, y
    
    def fit(self, data: pd.Series, method: str = 'linear', **kwargs) -> None:
        """Fit the forecasting model."""
        if method not in self.models:
            raise ValueError(f"Unknown method: {method}")
        
        if method in ['linear', 'random_forest']:
            X, y = self.prepare_features(data, kwargs.get('n_lags', 5))
            self.models[method].fit(X, y)
        elif method == 'arima':
            self.models[method] = ARIMA(data, order=kwargs.get('order', (1, 1, 1))).fit()
        elif method == 'sarima':
            self.models[method] = SARIMAX(
                data,
                order=kwargs.get('order', (1, 1, 1)),
                seasonal_order=kwargs.get('seasonal_order', (1, 1, 1, 12))
            ).fit()
    
    def predict(self, data: pd.Series, method: str = 'linear', **kwargs) -> np.ndarray:
        """Make predictions using the fitted model."""
        if method not in self.models:
            raise ValueError(f"Unknown method: {method}")
        
        if method in ['linear', 'random_forest']:
            X, _ = self.prepare_features(data, kwargs.get('n_lags', 5))
            return self.models[method].predict(X)
        elif method == 'arima':
            return self.models[method].predict(start=len(data), end=len(data) + kwargs.get('steps', 1) - 1)
        elif method == 'sarima':
            return self.models[method].predict(start=len(data), end=len(data) + kwargs.get('steps', 1) - 1)
    
    def forecast(self, data: pd.Series, steps: int = 30, method: str = 'linear', **kwargs) -> Dict[str, Any]:
        """Generate forecast for future values."""
        # Fit the model
        self.fit(data, method, **kwargs)
        
        # Generate forecast
        if method in ['linear', 'random_forest']:
            # For ML models, we need to generate forecasts iteratively
            forecast_values = []
            last_values = data.iloc[-kwargs.get('n_lags', 5):].values
            
            for _ in range(steps):
                # Prepare features for next prediction
                X = last_values.reshape(1, -1)
                # Make prediction
                pred = self.models[method].predict(X)[0]
                forecast_values.append(pred)
                # Update last values
                last_values = np.roll(last_values, -1)
                last_values[-1] = pred
            
            forecast_values = np.array(forecast_values)
        else:
            # For ARIMA/SARIMA, we can get all forecasts at once
            forecast_values = self.predict(data, method, steps=steps)
        
        # Create forecast index
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            if pd.infer_freq(data.index) == 'D':  # Daily data
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
            else:  # Default to business days
                forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='B')
        else:
            # If index is not datetime, use simple integer index
            forecast_index = range(len(data), len(data) + steps)
        
        # Create forecast series
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        # Calculate confidence intervals
        if method in ['arima', 'sarima']:
            forecast_ci = self.models[method].get_forecast(steps=steps).conf_int()
            return {
                'forecast': forecast_series,
                'lower_ci': forecast_ci.iloc[:, 0],
                'upper_ci': forecast_ci.iloc[:, 1]
            }
        
        return {'forecast': forecast_series}
    
    def evaluate(self, data: pd.Series, method: str = 'linear', **kwargs) -> Dict[str, Any]:
        """Evaluate the forecasting model."""
        # Split data into train and test
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Fit model and make predictions
        self.fit(train_data, method, **kwargs)
        predictions = self.predict(train_data, method, **kwargs)
        
        # Calculate metrics
        mse = np.mean((test_data.values - predictions) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_data.values - predictions))
        mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def get_forecast_summary(self, series: pd.Series, 
                           steps: int = 10,
                           methods: List[str] = None) -> Dict[str, Any]:
        """
        Get a summary of forecasts from multiple methods.
        
        Args:
            series (pd.Series): Time series data
            steps (int): Number of steps to forecast
            methods (List[str]): List of forecasting methods to use
            
        Returns:
            Dict[str, Any]: Dictionary containing forecast summary
        """
        if methods is None:
            methods = ['linear', 'random_forest', 'arima', 'sarima']
        
        forecasts = {}
        for method in methods:
            try:
                forecasts[method] = self.forecast(series, steps=steps, method=method)
            except Exception as e:
                forecasts[method] = {'error': str(e)}
        
        return {
            'forecasts': forecasts,
            'ensemble_forecast': pd.concat([f['forecast'] for f in forecasts.values() if 'forecast' in f], axis=1).mean(axis=1),
            'methods_used': methods,
            'steps': steps
        } 