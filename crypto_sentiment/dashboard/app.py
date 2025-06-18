"""
Crypto Sentiment Dashboard - Main application file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import sys
import os
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.forecasting import ForecastingModel
from analytics.sentiment_analyzer import SentimentAnalyzer
from data.simulator import DataSimulator
from analysis.advanced_analytics import AdvancedAnalytics
from utils.export import export_to_csv, export_to_pdf

# Initialize session state
if 'simulator' not in st.session_state:
    st.session_state.simulator = DataSimulator()

def setup_page():
    """Configure the Streamlit page settings."""
    st.set_page_config(
        page_title="Crypto Sentiment Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for dark mode
    st.markdown("""
        <style>
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #262730;
            color: #ffffff;
            border: 1px solid #4a4a4a;
        }
        .stButton>button:hover {
            background-color: #363940;
            border-color: #6a6a6a;
        }
        </style>
    """, unsafe_allow_html=True)

def export_to_csv(price_data, sentiment_data, metrics, insights, tweets):
    """Export data to CSV format."""
    # Create a buffer to store the CSV data
    buffer = io.StringIO()
    
    # Write price data
    buffer.write("Price Data\n")
    price_data.to_csv(buffer)
    buffer.write("\n")
    
    # Write sentiment data
    buffer.write("Sentiment Data\n")
    sentiment_data.to_csv(buffer)
    buffer.write("\n")
    
    # Write metrics
    buffer.write("Metrics\n")
    pd.Series(metrics).to_csv(buffer)
    buffer.write("\n")
    
    # Write insights
    buffer.write("Insights\n")
    pd.Series(insights).to_csv(buffer)
    buffer.write("\n")
    
    # Write tweets
    buffer.write("Recent Tweets\n")
    pd.DataFrame(tweets).to_csv(buffer)
    
    return buffer.getvalue()

def export_to_pdf(price_data, sentiment_data, metrics, insights, tweets, selected_crypto):
    """Export data to PDF format."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph(f"{selected_crypto} Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Metrics
    elements.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics_data = [[k, f"{v:.2%}" if isinstance(v, float) else str(v)] for k, v in metrics.items()]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 20))
    
    # Insights
    elements.append(Paragraph("Market Insights", styles['Heading2']))
    for insight in insights:
        elements.append(Paragraph(f"• {insight}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Recent Tweets
    elements.append(Paragraph("Recent Social Media Activity", styles['Heading2']))
    tweets_data = [[tweet['text'], f"{tweet['sentiment']:.2f}"] for tweet in tweets]
    tweets_table = Table(tweets_data, colWidths=[4*inch, 1*inch])
    tweets_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(tweets_table)
    
    # Build PDF
    doc.build(elements)
    return buffer.getvalue()

def create_sidebar() -> Tuple[str, Tuple[datetime, datetime], bool, str]:
    """Create the sidebar with controls."""
    st.sidebar.title("Controls")
    
    # Cryptocurrency selection
    selected_crypto = st.sidebar.selectbox(
        "Select Cryptocurrency",
        ["BTC", "ETH", "BNB", "SOL"],
        index=0
    )
    
    # Date range selection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(start_date, end_date),
        max_value=end_date
    )
    
    # Analysis options
    analyze = st.sidebar.checkbox("Run Advanced Analysis", value=True)
    
    # Export options
    export_format = st.sidebar.selectbox(
        "Export Format",
        ["None", "CSV", "PDF"],
        index=0
    )
    
    return selected_crypto, date_range, analyze, export_format

def plot_price_and_sentiment(price_data, sentiment_data, metrics):
    """Create an interactive plot of price and sentiment with enhanced features."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Price", "Sentiment"),
        row_heights=[0.7, 0.3]
    )
    
    # Price plot with most volatile day annotation
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data.values,
            name="Price",
            line=dict(color="#00ff00", width=2)
        ),
        row=1, col=1
    )
    
    # Add most volatile day marker
    if metrics['most_volatile_day']:
        fig.add_trace(
            go.Scatter(
                x=[metrics['most_volatile_day']],
                y=[price_data[metrics['most_volatile_day']]],
                mode='markers',
                name='Most Volatile Day',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='star'
                )
            ),
            row=1, col=1
        )
    
    # Enhanced sentiment plot with color-coded markers
    sentiment_colors = np.where(
        sentiment_data > 0.3, "#00ff00",  # Positive
        np.where(sentiment_data < -0.3, "#ff0000", "#ffff00")  # Negative, Neutral
    )
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data.values,
            name="Sentiment",
            mode="lines",
            line=dict(color="#ffffff", width=1)
        ),
        row=2, col=1
    )
    
    # Add sentiment markers
    fig.add_trace(
        go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data.values,
            name="Sentiment Points",
            mode="markers",
            marker=dict(
                color=sentiment_colors,
                size=8,
                symbol="circle"
            )
        ),
        row=2, col=1
    )
    
    # Add sentiment thresholds
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-0.3, line_dash="dash", line_color="red", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        template="plotly_dark",
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment Score", row=2, col=1)
    
    return fig

def generate_insights(price_data, sentiment_data, metrics):
    """Generate detailed insights from the data."""
    insights = []
    
    # Calculate price changes
    price_changes = price_data.pct_change()
    sentiment_changes = sentiment_data.diff()
    
    # Find significant price movements
    significant_moves = price_changes[abs(price_changes) > 0.05]  # 5% threshold
    
    # Analyze sentiment leading price
    for date, change in significant_moves.items():
        # Look back 7 days for sentiment changes
        sentiment_window = sentiment_changes[date - pd.Timedelta(days=7):date]
        if len(sentiment_window) > 0:
            max_sentiment_change = sentiment_window.idxmax()
            if sentiment_window[max_sentiment_change] > 0.2:  # Significant positive sentiment
                days_before = (date - max_sentiment_change).days
                if days_before > 0:
                    insights.append(
                        f"Positive sentiment rose {days_before} days before a {change:.1%} price move"
                    )
    
    # Analyze sentiment shocks
    sentiment_std = sentiment_data.std()
    sentiment_shocks = sentiment_data[abs(sentiment_data) > 2 * sentiment_std]
    for date, shock in sentiment_shocks.items():
        price_change = price_changes[date:date + pd.Timedelta(days=3)].sum()
        if abs(price_change) > 0.05:  # 5% price change
            direction = "rise" if price_change > 0 else "fall"
            insights.append(
                f"Extreme {'positive' if shock > 0 else 'negative'} sentiment ({shock:.2f}) "
                f"preceded a {abs(price_change):.1%} price {direction}"
            )
    
    # Add volatility insights
    if metrics['price_volatility'] > 0.03:  # 3% daily volatility
        insights.append(
            f"High price volatility detected ({metrics['price_volatility']:.1%} daily)"
        )
    
    # Add sentiment trend insights
    sentiment_trend = sentiment_data.rolling(7).mean()
    if (sentiment_trend > 0.2).any():
        insights.append("Sustained positive sentiment trend detected")
    elif (sentiment_trend < -0.2).any():
        insights.append("Sustained negative sentiment trend detected")
    
    return insights[:5]  # Return top 5 insights

def display_metrics(metrics: Dict):
    """Display key metrics in a grid layout."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Price Volatility",
            f"{metrics['price_volatility']:.1%}",
            delta=f"{metrics['price_volatility_change']:.1%}"
        )
    
    with col2:
        st.metric(
            "Sentiment Volatility",
            f"{metrics['sentiment_volatility']:.1%}",
            delta=f"{metrics['sentiment_volatility_change']:.1%}"
        )
    
    with col3:
        st.metric(
            "Correlation",
            f"{metrics['correlation']:.2f}",
            delta=f"{metrics['correlation_change']:.2f}"
        )
    
    with col4:
        st.metric(
            "Most Volatile Day",
            metrics['most_volatile_day'].strftime('%Y-%m-%d'),
            delta=f"{metrics['most_volatile_day_change']:.1%}"
        )

def display_insights(insights: List[str]):
    """Display insights in a formatted way."""
    st.markdown("### 📊 Key Insights")
    
    for insight in insights:
        st.markdown(f"""
            <div style="background-color: #262730; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                {insight}
            </div>
        """, unsafe_allow_html=True)

def display_sample_tweets(tweets: List[Dict], selected_date=None):
    """Display sample tweets with enhanced formatting."""
    st.markdown("### 🐦 Sample Tweets")
    
    for tweet in tweets:
        # Determine emoji based on sentiment
        if tweet["sentiment"] > 0.3:
            emoji = "🚀"  # Bullish
        elif tweet["sentiment"] < -0.3:
            emoji = "🐻"  # Bearish
        else:
            emoji = "➡️"  # Neutral
            
        # Add additional emojis based on sentiment strength
        if abs(tweet["sentiment"]) > 0.7:
            emoji += " 🔥"  # Very strong sentiment
        elif abs(tweet["sentiment"]) > 0.5:
            emoji += " ⚡"  # Strong sentiment
            
        sentiment_color = "green" if tweet["sentiment"] > 0 else "red" if tweet["sentiment"] < 0 else "yellow"
        
        st.markdown(f"""
            <div style="background-color: #262730; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                <p style="color: {sentiment_color}; margin: 0;">
                    {emoji} {tweet['text']}
                </p>
                <small style="color: {sentiment_color};">
                    Sentiment: {tweet['sentiment']:.2f}
                </small>
            </div>
        """, unsafe_allow_html=True)

def display_backtest_results(results: Dict):
    """Display backtest results in a formatted way."""
    st.markdown("### 📈 Backtest Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{results['total_return']:.1%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
    
    with col4:
        st.metric("Win Rate", f"{results['win_rate']:.1%}")

def plot_cross_correlation(price_data, sentiment_data, max_lag=7):
    """Create an enhanced cross-correlation plot."""
    # Calculate correlations for different lags
    correlations = []
    lags = list(range(-max_lag, max_lag + 1))  # Convert range to list
    
    for lag in lags:
        if lag < 0:
            corr = price_data.corr(sentiment_data.shift(-lag))
        else:
            corr = price_data.corr(sentiment_data.shift(lag))
        correlations.append(corr)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=np.array(correlations).reshape(1, -1),
        x=lags,
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Correlation")
    ))
    
    # Add annotations for significant correlations
    for i, corr in enumerate(correlations):
        if abs(corr) > 0.3:  # Significant correlation threshold
            fig.add_annotation(
                x=lags[i],
                y=0,
                text=f"{corr:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(corr) < 0.5 else "black")
            )
    
    fig.update_layout(
        title="Price-Sentiment Cross Correlation",
        xaxis_title="Lag (days)",
        yaxis_title="",
        template="plotly_dark",
        height=200,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def main():
    """Main function to run the dashboard."""
    setup_page()
    
    # Initialize analytics
    analytics = AdvancedAnalytics()
    
    # Header
    st.title("Crypto Sentiment Dashboard")
    st.markdown("Analyze cryptocurrency price movements and social sentiment")
    
    # Sidebar controls
    selected_crypto, date_range, analyze, export_format = create_sidebar()
    
    # Generate data for selected cryptocurrency
    price_data = st.session_state.simulator.generate_price_data(selected_crypto)
    sentiment_data = st.session_state.simulator.generate_sentiment_data(selected_crypto, price_data)
    
    # Filter data based on selected date range
    start_date = pd.Timestamp(date_range[0])
    end_date = pd.Timestamp(date_range[1])

    # Ensure we have data for the selected range
    if start_date > end_date:
        st.error("Start date cannot be after end date")
        st.stop()

    if start_date < price_data.index.min():
        start_date = price_data.index.min()
        st.warning(f"Adjusted start date to {start_date.date()} (earliest available data)")

    if end_date > price_data.index.max():
        end_date = price_data.index.max()
        st.warning(f"Adjusted end date to {end_date.date()} (latest available data)")

    mask = (price_data.index >= start_date) & (price_data.index <= end_date)
    price_data = price_data[mask]
    sentiment_data = sentiment_data[mask]

    if len(price_data) == 0:
        st.error("No data available for the selected date range")
        st.stop()

    # Calculate basic metrics
    metrics = st.session_state.simulator.calculate_metrics(price_data, sentiment_data)
    display_metrics(metrics)

    # Advanced analysis
    if analyze:
        # Detect sentiment lags
        insights = analytics.detect_sentiment_lags(price_data, sentiment_data)
        
        # Prepare features and train predictive model
        features = analytics.prepare_features(sentiment_data)
        prediction, confidence = analytics.train_predictive_model(features, price_data)
        
        # Display prediction
        st.markdown("### 🔮 Price Prediction")
        direction = "up" if prediction == 1 else "down"
        st.markdown(f"""
            <div style="background-color: #262730; padding: 1rem; border-radius: 5px; margin: 0.5rem 0;">
                <h3 style="margin: 0;">Predicted Direction: {direction.upper()} (Confidence: {confidence:.1%})</h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Plot enhanced overlay
        st.plotly_chart(
            analytics.plot_sentiment_price_overlay(price_data, sentiment_data, insights),
            use_container_width=True
        )
        
        # Display insights
        display_insights([
            f"{'Positive' if i['type'] == 'positive' else 'Negative'} sentiment spiked {i['days_before']} days before a {abs(i['price_change']):.1%} {'rise' if i['price_change'] > 0 else 'drop'}"
            for i in insights
        ])
        
        # Run backtest
        backtest_results = analytics.backtest_strategy(price_data, sentiment_data)
        display_backtest_results(backtest_results)
    
    # Generate and display sample tweets
    latest_date = sentiment_data.index[-1]
    latest_sentiment = sentiment_data.iloc[-1]
    tweets = st.session_state.simulator.generate_sample_tweets(selected_crypto, latest_date, latest_sentiment)
    display_sample_tweets(tweets)
    
    # Handle export
    if export_format:
        if export_format == "CSV":
            csv_data = export_to_csv(price_data, sentiment_data, metrics, insights, tweets)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{selected_crypto}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:  # PDF
            pdf_data = export_to_pdf(price_data, sentiment_data, metrics, insights, tweets, selected_crypto)
            st.download_button(
                label="Download PDF",
                data=pdf_data,
                file_name=f"{selected_crypto}_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

if __name__ == "__main__":
    main() 