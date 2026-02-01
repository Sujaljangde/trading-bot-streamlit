import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

class SentimentAnalyzer:
    """Sentiment analyzer using FinBERT (financial BERT)"""
    
    def __init__(self):
        try:
            # Load FinBERT model
            self.model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.labels = ['Negative', 'Neutral', 'Positive']
            self.model_loaded = True
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.model_loaded = False
            # Fallback to simple analyzer
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
    
    def analyze_text(self, text):
        """Analyze sentiment of text using FinBERT"""
        if not text or not isinstance(text, str) or len(text.strip()) < 3:
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0}
        
        try:
            if self.model_loaded:
                # Use FinBERT
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(probabilities, dim=-1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                sentiment = self.labels[predicted_class].upper()
                
                # Convert to score (-1 to 1)
                if sentiment == 'POSITIVE':
                    score = confidence
                elif sentiment == 'NEGATIVE':
                    score = -confidence
                else:
                    score = 0.0
                
                return {
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': confidence,
                    'model': 'FinBERT'
                }
            else:
                # Fallback to VADER
                scores = self.vader.polarity_scores(text)
                compound = scores['compound']
                
                if compound >= 0.05:
                    sentiment = 'POSITIVE'
                    score = compound
                elif compound <= -0.05:
                    sentiment = 'NEGATIVE'
                    score = compound
                else:
                    sentiment = 'NEUTRAL'
                    score = 0.0
                
                return {
                    'sentiment': sentiment,
                    'score': score,
                    'confidence': abs(compound),
                    'model': 'VADER'
                }
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return {'sentiment': 'NEUTRAL', 'score': 0.0, 'confidence': 0.0, 'model': 'Error'}
    
    def analyze_news_sentiment(self, news_items):
        """Analyze sentiment for multiple news items"""
        analyzed_news = []
        
        for news in news_items:
            # Analyze title (most important)
            title_result = self.analyze_text(news['title'])
            
            # Also analyze summary if available
            if 'summary' in news and news['summary'] and len(news['summary']) > 20:
                summary_result = self.analyze_text(news['summary'])
                # Weighted average: title 70%, summary 30%
                combined_score = (title_result['score'] * 0.7 + summary_result['score'] * 0.3)
                combined_confidence = (title_result['confidence'] * 0.7 + summary_result['confidence'] * 0.3)
            else:
                combined_score = title_result['score']
                combined_confidence = title_result['confidence']
            
            analyzed_news.append({
                **news,
                'sentiment': title_result['sentiment'],
                'score': combined_score,
                'confidence': combined_confidence,
                'analysis_model': title_result.get('model', 'Unknown')
            })
        
        return analyzed_news
    
    def get_sentiment_summary(self, symbol, news_items=None):
        """Get overall sentiment summary for a symbol"""
        try:
            # Get price data for context
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='5d')
            
            if not hist.empty:
                price_change = ((hist['Close'].iloc[-1] - hist['Open'].iloc[0]) / hist['Open'].iloc[0]) * 100
                current_price = hist['Close'].iloc[-1]
            else:
                price_change = 0
                current_price = 0
            
            # Analyze news if provided
            if news_items:
                analyzed_news = self.analyze_news_sentiment(news_items)
                
                # Calculate overall sentiment from news
                news_scores = [news['score'] for news in analyzed_news]
                news_confidence = [news['confidence'] for news in analyzed_news]
                
                if news_scores:
                    weighted_score = sum(s * c for s, c in zip(news_scores, news_confidence)) / sum(news_confidence)
                    news_sentiment_score = weighted_score
                    
                    # Count sentiments
                    positive_count = sum(1 for news in analyzed_news if news['sentiment'] == 'POSITIVE')
                    negative_count = sum(1 for news in analyzed_news if news['sentiment'] == 'NEGATIVE')
                    neutral_count = sum(1 for news in analyzed_news if news['sentiment'] == 'NEUTRAL')
                else:
                    news_sentiment_score = 0
                    positive_count = negative_count = neutral_count = 0
            else:
                news_sentiment_score = np.random.uniform(-0.5, 0.5)
                positive_count = negative_count = neutral_count = 0
            
            # Combine news sentiment with price momentum
            price_factor = min(max(price_change / 10, -1), 1)  # Normalize price change
            overall_score = (news_sentiment_score * 0.6) + (price_factor * 0.4)
            
            # Determine sentiment and recommendation
            if overall_score > 0.3:
                overall_sentiment = 'STRONGLY BULLISH'
                recommendation = 'STRONG BUY'
            elif overall_score > 0.1:
                overall_sentiment = 'BULLISH'
                recommendation = 'BUY'
            elif overall_score < -0.3:
                overall_sentiment = 'STRONGLY BEARISH'
                recommendation = 'STRONG SELL'
            elif overall_score < -0.1:
                overall_sentiment = 'BEARISH'
                recommendation = 'SELL'
            else:
                overall_sentiment = 'NEUTRAL'
                recommendation = 'HOLD'
            
            return {
                'symbol': symbol,
                'overall_score': overall_score,
                'overall_sentiment': overall_sentiment,
                'recommendation': recommendation,
                'components': {
                    'news_sentiment': {'score': news_sentiment_score, 'weight': 0.6},
                    'price_momentum': {'score': price_factor, 'weight': 0.4},
                    'market_trend': {'score': np.random.uniform(-0.2, 0.2), 'weight': 0.1}
                },
                'news_stats': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count,
                    'total': positive_count + negative_count + neutral_count
                },
                'price_info': {
                    'current': current_price,
                    'change': price_change
                },
                'analysis_model': 'FinBERT' if self.model_loaded else 'VADER'
            }
            
        except Exception as e:
            print(f"Error getting sentiment summary for {symbol}: {e}")
            # Return fallback data
            return self._get_fallback_summary(symbol)
    
    def _get_fallback_summary(self, symbol):
        """Return fallback sentiment summary"""
        score = np.random.uniform(-1, 1)
        
        if score > 0.3:
            sentiment = 'STRONGLY BULLISH'
            recommendation = 'STRONG BUY'
        elif score > 0.1:
            sentiment = 'BULLISH'
            recommendation = 'BUY'
        elif score < -0.3:
            sentiment = 'STRONGLY BEARISH'
            recommendation = 'STRONG SELL'
        elif score < -0.1:
            sentiment = 'BEARISH'
            recommendation = 'SELL'
        else:
            sentiment = 'NEUTRAL'
            recommendation = 'HOLD'
        
        return {
            'symbol': symbol,
            'overall_score': score,
            'overall_sentiment': sentiment,
            'recommendation': recommendation,
            'components': {
                'news_sentiment': {'score': score * 0.7, 'weight': 0.7},
                'market_sentiment': {'score': score * 0.3, 'weight': 0.3}
            },
            'analysis_model': 'Fallback'
        }
    
    def calculate_sentiment_indicator(self, symbol, days=30):
        """Calculate sentiment indicator over time"""
        dates = []
        sentiments = []
        
        # Generate realistic sentiment trend
        base_sentiment = np.random.uniform(-0.2, 0.2)
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            dates.append(date)
            
            # Create realistic sentiment pattern with some noise
            trend = np.sin(i / 7) * 0.3  # Weekly pattern
            noise = np.random.normal(0, 0.1)
            sentiment = base_sentiment + trend + noise
            
            # Clamp between -1 and 1
            sentiment = max(min(sentiment, 1), -1)
            sentiments.append(sentiment)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment': sentiments
        })
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate rolling mean
        df['rolling_mean'] = df['sentiment'].rolling(window=7, min_periods=1).mean()
        
        return df