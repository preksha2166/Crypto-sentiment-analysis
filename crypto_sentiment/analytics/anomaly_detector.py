"""
Anomaly detector for identifying anomalies in time series data.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    """Class for detecting anomalies in time series data."""
    
    def __init__(self):
        """Initialize the anomaly detector."""
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
    
    def detect_anomalies(self, data: Union[pd.Series, pd.DataFrame], 
                        method: str = 'isolation_forest',
                        threshold: float = 2.0) -> Dict:
        """
        Detect anomalies in the data.
        
        Args:
            data: Time series data
            method: Detection method ('isolation_forest' or 'zscore')
            threshold: Threshold for z-score method
            
        Returns:
            Dictionary containing anomaly information
        """
        if method == 'isolation_forest':
            return self._detect_using_isolation_forest(data)
        elif method == 'zscore':
            return self._detect_using_zscore(data, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_using_isolation_forest(self, data: Union[pd.Series, pd.DataFrame]) -> Dict:
        """Detect anomalies using Isolation Forest."""
        # Prepare data
        if isinstance(data, pd.Series):
            X = data.values.reshape(-1, 1)
        else:
            X = data.values
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model and predict
        self.model.fit(X_scaled)
        scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Convert predictions to boolean (1 for normal, -1 for anomaly)
        is_anomaly = predictions == -1
        
        # Create results
        results = {
            'is_anomaly': is_anomaly,
            'anomaly_scores': scores,
            'anomaly_indices': np.where(is_anomaly)[0],
            'anomaly_count': np.sum(is_anomaly)
        }
        
        if isinstance(data, pd.Series):
            results['anomaly_values'] = data[is_anomaly]
        else:
            results['anomaly_values'] = data.iloc[is_anomaly]
        
        return results
    
    def _detect_using_zscore(self, data: Union[pd.Series, pd.DataFrame], 
                           threshold: float) -> Dict:
        """Detect anomalies using z-score method."""
        # Calculate z-scores
        if isinstance(data, pd.Series):
            z_scores = np.abs((data - data.mean()) / data.std())
        else:
            z_scores = np.abs((data - data.mean()) / data.std())
        
        # Identify anomalies
        is_anomaly = z_scores > threshold
        
        # Create results
        results = {
            'is_anomaly': is_anomaly,
            'z_scores': z_scores,
            'anomaly_indices': np.where(is_anomaly)[0],
            'anomaly_count': np.sum(is_anomaly)
        }
        
        if isinstance(data, pd.Series):
            results['anomaly_values'] = data[is_anomaly]
        else:
            results['anomaly_values'] = data.iloc[is_anomaly]
        
        return results
    
    def get_anomaly_summary(self, data: Union[pd.Series, pd.DataFrame],
                          method: str = 'isolation_forest',
                          threshold: float = 2.0) -> Dict:
        """Get a summary of detected anomalies."""
        results = self.detect_anomalies(data, method, threshold)
        
        summary = {
            'total_points': len(data),
            'anomaly_count': results['anomaly_count'],
            'anomaly_percentage': (results['anomaly_count'] / len(data)) * 100
        }
        
        if isinstance(data, pd.Series):
            if results['anomaly_count'] > 0:
                summary.update({
                    'min_anomaly_value': results['anomaly_values'].min(),
                    'max_anomaly_value': results['anomaly_values'].max(),
                    'mean_anomaly_value': results['anomaly_values'].mean()
                })
        else:
            if results['anomaly_count'] > 0:
                summary['anomaly_statistics'] = results['anomaly_values'].describe().to_dict()
        
        return summary 