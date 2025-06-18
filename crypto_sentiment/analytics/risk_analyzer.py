import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy import stats

class RiskAnalyzer:
    def __init__(self):
        """Initialize the risk analyzer."""
        self.risk_metrics = {}
        
    def calculate_volatility(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            returns (pd.Series): Series of returns
            window (int): Rolling window size
            
        Returns:
            pd.Series: Rolling volatility
        """
        return returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns (pd.Series): Series of returns
            confidence_level (float): Confidence level for VaR calculation
            
        Returns:
            float: Value at Risk
        """
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns (pd.Series): Series of returns
            confidence_level (float): Confidence level for CVaR calculation
            
        Returns:
            float: Conditional Value at Risk
        """
        var = self.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sharpe ratio
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio.
        
        Args:
            returns (pd.Series): Series of returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            float: Sortino ratio
        """
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2))
        return np.sqrt(252) * excess_returns.mean() / downside_std
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            prices (pd.Series): Series of prices
            
        Returns:
            float: Maximum drawdown
        """
        rolling_max = prices.expanding().max()
        drawdowns = prices / rolling_max - 1
        return drawdowns.min()
    
    def calculate_beta(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta relative to market.
        
        Args:
            returns (pd.Series): Series of asset returns
            market_returns (pd.Series): Series of market returns
            
        Returns:
            float: Beta coefficient
        """
        covariance = returns.cov(market_returns)
        market_variance = market_returns.var()
        return covariance / market_variance
    
    def calculate_correlation(self, returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate correlation with market.
        
        Args:
            returns (pd.Series): Series of asset returns
            market_returns (pd.Series): Series of market returns
            
        Returns:
            float: Correlation coefficient
        """
        return returns.corr(market_returns)
    
    def calculate_tracking_error(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate tracking error.
        
        Args:
            returns (pd.Series): Series of asset returns
            benchmark_returns (pd.Series): Series of benchmark returns
            
        Returns:
            float: Tracking error
        """
        return np.sqrt(252) * (returns - benchmark_returns).std()
    
    def calculate_information_ratio(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate information ratio.
        
        Args:
            returns (pd.Series): Series of asset returns
            benchmark_returns (pd.Series): Series of benchmark returns
            
        Returns:
            float: Information ratio
        """
        excess_returns = returns - benchmark_returns
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0) -> float:
        """
        Calculate Omega ratio.
        
        Args:
            returns (pd.Series): Series of returns
            threshold (float): Return threshold
            
        Returns:
            float: Omega ratio
        """
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        return gains / losses if losses != 0 else float('inf')
    
    def calculate_calmar_ratio(self, returns: pd.Series, prices: pd.Series) -> float:
        """
        Calculate Calmar ratio.
        
        Args:
            returns (pd.Series): Series of returns
            prices (pd.Series): Series of prices
            
        Returns:
            float: Calmar ratio
        """
        annual_return = returns.mean() * 252
        max_drawdown = self.calculate_max_drawdown(prices)
        return annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
    
    def get_risk_metrics(self, 
                        prices: pd.Series,
                        returns: pd.Series,
                        market_returns: Optional[pd.Series] = None,
                        benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            prices (pd.Series): Series of prices
            returns (pd.Series): Series of returns
            market_returns (pd.Series, optional): Series of market returns
            benchmark_returns (pd.Series, optional): Series of benchmark returns
            
        Returns:
            Dict[str, float]: Dictionary of risk metrics
        """
        metrics = {
            'volatility': self.calculate_volatility(returns).iloc[-1],
            'var_95': self.calculate_var(returns),
            'cvar_95': self.calculate_cvar(returns),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(prices),
            'omega_ratio': self.calculate_omega_ratio(returns),
            'calmar_ratio': self.calculate_calmar_ratio(returns, prices)
        }
        
        if market_returns is not None:
            metrics.update({
                'beta': self.calculate_beta(returns, market_returns),
                'correlation': self.calculate_correlation(returns, market_returns)
            })
            
        if benchmark_returns is not None:
            metrics.update({
                'tracking_error': self.calculate_tracking_error(returns, benchmark_returns),
                'information_ratio': self.calculate_information_ratio(returns, benchmark_returns)
            })
            
        return metrics
    
    def get_risk_summary(self, 
                        prices: pd.Series,
                        returns: pd.Series,
                        market_returns: Optional[pd.Series] = None,
                        benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Get a comprehensive risk analysis summary.
        
        Args:
            prices (pd.Series): Series of prices
            returns (pd.Series): Series of returns
            market_returns (pd.Series, optional): Series of market returns
            benchmark_returns (pd.Series, optional): Series of benchmark returns
            
        Returns:
            Dict[str, Any]: Dictionary containing risk metrics and analysis
        """
        metrics = self.get_risk_metrics(prices, returns, market_returns, benchmark_returns)
        
        # Calculate rolling metrics
        rolling_volatility = self.calculate_volatility(returns)
        rolling_sharpe = pd.Series(index=returns.index)
        for i in range(len(returns)):
            if i >= 252:  # Use 1 year of data
                rolling_sharpe.iloc[i] = self.calculate_sharpe_ratio(returns.iloc[i-252:i])
        
        return {
            'metrics': metrics,
            'rolling_metrics': {
                'volatility': rolling_volatility,
                'sharpe_ratio': rolling_sharpe
            },
            'risk_level': self._assess_risk_level(metrics),
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _assess_risk_level(self, metrics: Dict[str, float]) -> str:
        """
        Assess overall risk level based on metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of risk metrics
            
        Returns:
            str: Risk level assessment
        """
        # Simple risk assessment based on volatility and drawdown
        if metrics['volatility'] > 0.5 or metrics['max_drawdown'] < -0.4:
            return 'High'
        elif metrics['volatility'] > 0.3 or metrics['max_drawdown'] < -0.25:
            return 'Medium'
        else:
            return 'Low'
    
    def _generate_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """
        Generate risk management recommendations.
        
        Args:
            metrics (Dict[str, float]): Dictionary of risk metrics
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if metrics['volatility'] > 0.4:
            recommendations.append("Consider implementing position sizing limits")
            
        if metrics['max_drawdown'] < -0.3:
            recommendations.append("Implement stop-loss mechanisms")
            
        if metrics['sharpe_ratio'] < 1:
            recommendations.append("Review strategy for better risk-adjusted returns")
            
        if metrics.get('beta', 1) > 1.2:
            recommendations.append("Consider reducing market exposure")
            
        if metrics.get('tracking_error', 0) > 0.1:
            recommendations.append("Review portfolio diversification")
            
        return recommendations 