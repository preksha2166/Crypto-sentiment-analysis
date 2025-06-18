"""
Sentiment analysis module for cryptocurrency-related text data.
"""
import logging
from typing import Dict, List, Any, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import numpy as np
from ..config import SENTIMENT_SETTINGS

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Main sentiment analysis class using transformer models."""
    
    def __init__(self):
        self.settings = SENTIMENT_SETTINGS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize sentiment model
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(
            self.settings["model"]["name"]
        )
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            self.settings["model"]["name"],
            num_labels=3  # Positive, Negative, Neutral
        ).to(self.device)
        
        # Initialize emotion detection pipeline
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1
        )
        
        # Load crypto-specific terms
        self.crypto_terms = self.settings["crypto_specific_terms"]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for sentiment and emotions.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment and emotion analysis results
        """
        try:
            # Basic sentiment analysis
            sentiment = self._analyze_sentiment(text)
            
            # Emotion detection
            emotions = self._detect_emotions(text)
            
            # Crypto-specific term analysis
            crypto_analysis = self._analyze_crypto_terms(text)
            
            return {
                "sentiment": sentiment,
                "emotions": emotions,
                "crypto_analysis": crypto_analysis,
                "text": text
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                "sentiment": {"label": "neutral", "score": 0.0},
                "emotions": {},
                "crypto_analysis": {},
                "text": text
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze basic sentiment of text."""
        try:
            inputs = self.sentiment_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings["model"]["max_length"]
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)[0]
                
            sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
            sentiment_idx = scores.argmax().item()
            
            return {
                "label": sentiment_map[sentiment_idx],
                "score": scores[sentiment_idx].item()
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"label": "neutral", "score": 0.0}
    
    def _detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text."""
        try:
            emotions = self.emotion_analyzer(text)[0]
            return {
                "label": emotions["label"],
                "score": emotions["score"]
            }
        except Exception as e:
            logger.error(f"Error in emotion detection: {str(e)}")
            return {}
    
    def _analyze_crypto_terms(self, text: str) -> Dict[str, Any]:
        """Analyze crypto-specific terminology in text."""
        text_lower = text.lower()
        results = {
            "positive_terms": [],
            "negative_terms": [],
            "neutral_terms": []
        }
        
        # Check for positive terms
        for term in self.crypto_terms["positive"]:
            if term.lower() in text_lower:
                results["positive_terms"].append(term)
        
        # Check for negative terms
        for term in self.crypto_terms["negative"]:
            if term.lower() in text_lower:
                results["negative_terms"].append(term)
        
        # Check for neutral terms
        for term in self.crypto_terms["neutral"]:
            if term.lower() in text_lower:
                results["neutral_terms"].append(term)
        
        return results
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze a batch of texts for sentiment and emotions.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of dictionaries containing analysis results
        """
        return [self.analyze_text(text) for text in texts]
    
    def detect_sarcasm(self, text: str, context: str = "") -> Dict[str, Any]:
        """
        Detect sarcasm in text using contextual analysis.
        
        Args:
            text: Text to analyze
            context: Optional context for better sarcasm detection
            
        Returns:
            Dictionary containing sarcasm detection results
        """
        # This is a placeholder for a more sophisticated sarcasm detection model
        # In a real implementation, this would use a fine-tuned model for sarcasm detection
        return {
            "is_sarcastic": False,
            "confidence": 0.0
        }
    
    def get_sentiment_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of sentiment analyses.
        
        Args:
            analyses: List of sentiment analysis results
            
        Returns:
            Dictionary containing sentiment summary statistics
        """
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        emotion_counts = {}
        
        for analysis in analyses:
            # Count sentiments
            sentiment = analysis["sentiment"]["label"]
            sentiment_counts[sentiment] += 1
            
            # Count emotions
            emotion = analysis["emotions"].get("label")
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(analyses)
        return {
            "sentiment_distribution": {
                k: v/total for k, v in sentiment_counts.items()
            },
            "emotion_distribution": {
                k: v/total for k, v in emotion_counts.items()
            },
            "total_analyses": total
        } 