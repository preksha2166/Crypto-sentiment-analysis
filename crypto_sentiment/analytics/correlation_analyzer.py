import pandas as pd
import numpy as np
from typing import Dict, List, Any
from scipy import stats

class CorrelationAnalyzer:
    def __init__(self):
        """Initialize the correlation analyzer."""
        pass
        
    def calculate_correlation(self, series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
        """
        Calculate correlation between two series.
        
        Args:
            series1 (pd.Series): First series
            series2 (pd.Series): Second series
            
        Returns:
            Dict[str, float]: Dictionary containing correlation coefficient and p-value
        """
        correlation, p_value = stats.pearsonr(series1.dropna(), series2.dropna())
        return {
            'correlation': correlation,
            'p_value': p_value
        }
    
    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing the series
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        return df.corr()
    
    def calculate_rolling_correlation(self, series1: pd.Series, 
                                    series2: pd.Series, 
                                    window: int = 20) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            series1 (pd.Series): First series
            series2 (pd.Series): Second series
            window (int): Rolling window size
            
        Returns:
            pd.Series: Series of rolling correlations
        """
        return series1.rolling(window=window).corr(series2)
    
    def find_highly_correlated_pairs(self, df: pd.DataFrame, 
                                   threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find pairs of highly correlated variables.
        
        Args:
            df (pd.DataFrame): DataFrame containing the variables
            threshold (float): Correlation threshold
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing correlated pairs
        """
        corr_matrix = self.calculate_correlation_matrix(df)
        highly_correlated = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    highly_correlated.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return highly_correlated
    
    def calculate_partial_correlation(self, df: pd.DataFrame, 
                                    var1: str, 
                                    var2: str, 
                                    control_vars: List[str]) -> Dict[str, float]:
        """
        Calculate partial correlation between two variables controlling for others.
        
        Args:
            df (pd.DataFrame): DataFrame containing the variables
            var1 (str): First variable name
            var2 (str): Second variable name
            control_vars (List[str]): List of control variable names
            
        Returns:
            Dict[str, float]: Dictionary containing partial correlation and p-value
        """
        from scipy import stats
        
        # Calculate residuals
        def get_residuals(y, X):
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return y - model.predict(X)
        
        # Get residuals for both variables
        X = df[control_vars]
        res1 = get_residuals(df[var1], X)
        res2 = get_residuals(df[var2], X)
        
        # Calculate correlation between residuals
        correlation, p_value = stats.pearsonr(res1, res2)
        
        return {
            'partial_correlation': correlation,
            'p_value': p_value
        }
    
    def get_correlation_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of correlation analysis.
        
        Args:
            df (pd.DataFrame): DataFrame containing the variables
            
        Returns:
            Dict[str, Any]: Dictionary containing correlation summary statistics
        """
        corr_matrix = self.calculate_correlation_matrix(df)
        
        return {
            'correlation_matrix': corr_matrix,
            'highly_correlated_pairs': self.find_highly_correlated_pairs(df),
            'average_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean(),
            'max_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max(),
            'min_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min()
        } 