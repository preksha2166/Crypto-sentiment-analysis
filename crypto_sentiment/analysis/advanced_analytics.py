import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def detect_sentiment_lags(self, price_data: pd.Series, sentiment_data: pd.Series, 
                            max_lag: int = 7, threshold: float = 0.05) -> List[Dict]:
        """Detect significant sentiment changes that preceded price movements."""
        insights = []
        price_changes = price_data.pct_change()
        sentiment_changes = sentiment_data.diff()
        
        # Find significant price movements
        significant_moves = price_changes[abs(price_changes) > threshold]
        
        for date, price_change in significant_moves.items():
            # Look back max_lag days for sentiment changes
            sentiment_window = sentiment_changes[date - pd.Timedelta(days=max_lag):date]
            
            if len(sentiment_window) > 0:
                # Find most significant sentiment change
                max_sentiment_change = sentiment_window.idxmax()
                min_sentiment_change = sentiment_window.idxmin()
                
                # Check if sentiment change preceded price move
                if sentiment_window[max_sentiment_change] > 0.2:
                    days_before = (date - max_sentiment_change).days
                    if days_before > 0:
                        insights.append({
                            'date': date,
                            'type': 'positive',
                            'sentiment_change': sentiment_window[max_sentiment_change],
                            'price_change': price_change,
                            'days_before': days_before
                        })
                
                if sentiment_window[min_sentiment_change] < -0.2:
                    days_before = (date - min_sentiment_change).days
                    if days_before > 0:
                        insights.append({
                            'date': date,
                            'type': 'negative',
                            'sentiment_change': sentiment_window[min_sentiment_change],
                            'price_change': price_change,
                            'days_before': days_before
                        })
        
        return insights
    
    def prepare_features(self, sentiment_data: pd.Series, window: int = 7) -> pd.DataFrame:
        """Prepare features for predictive modeling."""
        features = pd.DataFrame()
        
        # Rolling statistics
        features['sentiment_mean'] = sentiment_data.rolling(window).mean()
        features['sentiment_std'] = sentiment_data.rolling(window).std()
        features['sentiment_min'] = sentiment_data.rolling(window).min()
        features['sentiment_max'] = sentiment_data.rolling(window).max()
        
        # Momentum indicators
        features['sentiment_momentum'] = sentiment_data.diff(window)
        features['sentiment_acceleration'] = features['sentiment_momentum'].diff()
        
        # Volatility
        features['sentiment_volatility'] = features['sentiment_std'] / features['sentiment_mean'].abs()
        
        return features
    
    def train_predictive_model(self, features: pd.DataFrame, 
                             price_data: pd.Series) -> Tuple[float, float]:
        """Train a model to predict price direction based on sentiment features."""
        # Prepare target variable (1 for price increase, 0 for decrease)
        target = (price_data.shift(-1) > price_data).astype(int)
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        X = features[valid_idx]
        y = target[valid_idx]
        
        if len(X) < 10:  # Not enough data
            return 0.0, 0.0
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Get latest prediction
        latest_features = self.scaler.transform(features.iloc[[-1]])
        prediction = self.model.predict(latest_features)[0]
        confidence = self.model.predict_proba(latest_features)[0].max()
        
        return prediction, confidence
    
    def backtest_strategy(self, price_data: pd.Series, sentiment_data: pd.Series,
                         threshold: float = 0.3) -> Dict:
        """Backtest a simple sentiment-based trading strategy."""
        # Generate signals
        signals = pd.Series(0, index=price_data.index)
        signals[sentiment_data > threshold] = 1  # Buy signal
        signals[sentiment_data < -threshold] = -1  # Sell signal
        
        # Calculate returns
        price_returns = price_data.pct_change()
        strategy_returns = signals.shift(1) * price_returns
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (strategy_returns > 0).mean()
        }
    
    def plot_sentiment_price_overlay(self, price_data: pd.Series, 
                                   sentiment_data: pd.Series,
                                   insights: List[Dict]) -> go.Figure:
        """Create an enhanced overlay plot of price and sentiment with alert windows."""
        fig = make_subplots(rows=2, cols=1,
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=("Price", "Sentiment"),
                           row_heights=[0.7, 0.3])
        
        # Price plot
        fig.add_trace(
            go.Scatter(
                x=price_data.index,
                y=price_data.values,
                name="Price",
                line=dict(color="#00ff00", width=2)
            ),
            row=1, col=1
        )
        
        # Sentiment plot
        fig.add_trace(
            go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data.values,
                name="Sentiment",
                line=dict(color="#ffffff", width=1)
            ),
            row=2, col=1
        )
        
        # Add alert windows
        for insight in insights:
            date = insight['date']
            days_before = insight['days_before']
            start_date = date - pd.Timedelta(days=days_before)
            
            # Add shaded region
            fig.add_vrect(
                x0=start_date,
                x1=date,
                fillcolor="red" if insight['type'] == 'negative' else "green",
                opacity=0.2,
                layer="below",
                line_width=0,
                row=1, col=1
            )
            
            # Add annotation
            fig.add_annotation(
                x=date,
                y=price_data[date],
                text=f"{insight['price_change']:.1%}",
                showarrow=True,
                arrowhead=1,
                row=1, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_dark",
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode="x unified"
        )
        
        return fig 