"""
Sentiment analyzer for analyzing text sentiment.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any
from textblob import TextBlob

class SentimentAnalyzer:
    """Class for analyzing text sentiment."""
    
    def __init__(self):
        """Initialize the sentiment analyzer."""
        pass
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of a single text."""
        blob = TextBlob(text)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment label
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'text': text,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'label': label
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze_text(text) for text in texts]
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Analyze sentiment of texts in a DataFrame."""
        # Create a copy of the input DataFrame
        result_df = df.copy()
        
        # Analyze each text
        analyses = self.analyze_batch(df[text_column].tolist())
        
        # Add sentiment columns
        result_df['sentiment_score'] = [a['polarity'] for a in analyses]
        result_df['sentiment_subjectivity'] = [a['subjectivity'] for a in analyses]
        result_df['sentiment_label'] = [a['label'] for a in analyses]
        
        return result_df
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get a summary of sentiment analysis results."""
        if 'sentiment_label' not in df.columns:
            raise ValueError("DataFrame must contain 'sentiment_label' column")
        
        # Calculate sentiment distribution
        sentiment_counts = df['sentiment_label'].value_counts()
        total = len(df)
        
        summary = {
            'total_texts': total,
            'sentiment_distribution': {
                label: count / total * 100
                for label, count in sentiment_counts.items()
            },
            'average_sentiment': df['sentiment_score'].mean() if 'sentiment_score' in df.columns else None,
            'average_subjectivity': df['sentiment_subjectivity'].mean() if 'sentiment_subjectivity' in df.columns else None
        }
        
        return summary 