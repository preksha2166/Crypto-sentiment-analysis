"""
Visualization manager for creating common charts and plots.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union

class VisualizationManager:
    """Manager class for creating visualizations."""
    
    def __init__(self):
        """Initialize the visualization manager."""
        self.default_colors = px.colors.qualitative.Set1
        self.default_template = 'plotly_white'
    
    def plot_forecast(self, 
                     actual: pd.Series,
                     forecast: Union[pd.Series, Dict[str, pd.Series]],
                     title: str = "Time Series Forecast",
                     show_confidence: bool = True) -> go.Figure:
        """
        Create a plot showing actual values and forecast.
        
        Args:
            actual (pd.Series): Actual time series data
            forecast (Union[pd.Series, Dict[str, pd.Series]]): Forecast data or dictionary of forecasts
            title (str): Plot title
            show_confidence (bool): Whether to show confidence intervals
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=actual.index,
            y=actual.values,
            name='Actual',
            line=dict(color=self.default_colors[0])
        ))
        
        # Plot forecast(s)
        if isinstance(forecast, pd.Series):
            fig.add_trace(go.Scatter(
                x=forecast.index,
                y=forecast.values,
                name='Forecast',
                line=dict(color=self.default_colors[1], dash='dash')
            ))
        else:
            for i, (method, series) in enumerate(forecast.items()):
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    name=f'{method} Forecast',
                    line=dict(color=self.default_colors[i+1], dash='dash')
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.default_template,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_correlation_matrix(self, 
                              corr_matrix: pd.DataFrame,
                              title: str = "Correlation Matrix") -> go.Figure:
        """
        Create a heatmap of the correlation matrix.
        
        Args:
            corr_matrix (pd.DataFrame): Correlation matrix
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title=title,
            template=self.default_template
        )
        
        return fig
    
    def plot_trend_analysis(self,
                          series: pd.Series,
                          trend: Dict[str, Any],
                          title: str = "Trend Analysis") -> go.Figure:
        """
        Create a plot showing the time series and its trend.
        
        Args:
            series (pd.Series): Time series data
            trend (Dict[str, Any]): Trend analysis results
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name='Actual',
            line=dict(color=self.default_colors[0])
        ))
        
        # Plot trend line
        if 'trend_line' in trend:
            fig.add_trace(go.Scatter(
                x=series.index,
                y=trend['trend_line'],
                name='Trend',
                line=dict(color=self.default_colors[1], dash='dash')
            ))
        
        # Add annotations for trend statistics
        annotations = [
            dict(
                x=0.02,
                y=0.98,
                xref='paper',
                yref='paper',
                text=f"Slope: {trend.get('slope', 'N/A'):.4f}<br>R²: {trend.get('r2_score', 'N/A'):.4f}",
                showarrow=False,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        ]
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.default_template,
            annotations=annotations
        )
        
        return fig
    
    def plot_rolling_statistics(self,
                              series: pd.Series,
                              window: int = 20,
                              title: str = "Rolling Statistics") -> go.Figure:
        """
        Create a plot showing rolling mean and standard deviation.
        
        Args:
            series (pd.Series): Time series data
            window (int): Rolling window size
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        
        fig = go.Figure()
        
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            name='Actual',
            line=dict(color=self.default_colors[0])
        ))
        
        # Plot rolling mean
        fig.add_trace(go.Scatter(
            x=rolling_mean.index,
            y=rolling_mean.values,
            name=f'{window}-period Rolling Mean',
            line=dict(color=self.default_colors[1])
        ))
        
        # Plot rolling standard deviation
        fig.add_trace(go.Scatter(
            x=rolling_std.index,
            y=rolling_std.values,
            name=f'{window}-period Rolling Std',
            line=dict(color=self.default_colors[2])
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.default_template
        )
        
        return fig
    
    def plot_forecast_comparison(self,
                               forecasts: Dict[str, Any],
                               actual: pd.Series = None,
                               title: str = "Forecast Comparison") -> go.Figure:
        """
        Create a plot comparing different forecasting methods.
        
        Args:
            forecasts (Dict[str, Any]): Dictionary of forecasts from different methods
            actual (pd.Series): Actual values (optional)
            title (str): Plot title
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        # Plot actual values if provided
        if actual is not None:
            fig.add_trace(go.Scatter(
                x=actual.index,
                y=actual.values,
                name='Actual',
                line=dict(color=self.default_colors[0])
            ))
        
        # Plot individual forecasts
        for i, (method, forecast) in enumerate(forecasts['forecasts'].items()):
            if 'forecast' in forecast:
                fig.add_trace(go.Scatter(
                    x=forecast['forecast'].index,
                    y=forecast['forecast'].values,
                    name=f'{method} Forecast',
                    line=dict(color=self.default_colors[i+1], dash='dash')
                ))
        
        # Plot ensemble forecast
        if 'ensemble_forecast' in forecasts:
            fig.add_trace(go.Scatter(
                x=forecasts['ensemble_forecast'].index,
                y=forecasts['ensemble_forecast'].values,
                name='Ensemble Forecast',
                line=dict(color='black', width=2, dash='dot')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            template=self.default_template,
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_time_series(data, title, x_label="Time", y_label="Value"):
        """Create a time series plot."""
        fig = go.Figure()
        
        if isinstance(data, pd.Series):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data.values,
                mode="lines",
                name=data.name if data.name else "Value"
            ))
        elif isinstance(data, pd.DataFrame):
            for column in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode="lines",
                    name=column
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            hovermode="x unified"
        )
        
        return fig
    
    @staticmethod
    def create_histogram(data, title, x_label="Value", y_label="Count"):
        """Create a histogram."""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data,
            name="Distribution"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        return fig
    
    @staticmethod
    def create_scatter(x, y, title, x_label="X", y_label="Y"):
        """Create a scatter plot."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Data Points"
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(data, title, x_label="X", y_label="Y"):
        """Create a heatmap."""
        fig = go.Figure()
        
        fig.add_trace(go.Heatmap(
            z=data,
            x=data.columns if isinstance(data, pd.DataFrame) else None,
            y=data.index if isinstance(data, pd.DataFrame) else None
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        return fig
    
    @staticmethod
    def create_box_plot(data, title, x_label="Category", y_label="Value"):
        """Create a box plot."""
        fig = go.Figure()
        
        if isinstance(data, pd.DataFrame):
            for column in data.columns:
                fig.add_trace(go.Box(
                    y=data[column],
                    name=column
                ))
        else:
            fig.add_trace(go.Box(
                y=data,
                name="Data"
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        return fig 