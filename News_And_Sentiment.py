import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

# Import modules
try:
    from scraper import NewsScraper
    from sentiment import SentimentAnalyzer
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    # Create fallback classes
    class NewsScraper:
        def get_news(self, symbol, max_news=10):
            return []
    class SentimentAnalyzer:
        def get_sentiment_summary(self, symbol, news_items=None):
            return {'overall_score': 0, 'overall_sentiment': 'NEUTRAL', 'recommendation': 'HOLD'}

st.set_page_config(
    page_title="FinBERT News & Sentiment",
    page_icon="📰",
    layout="wide"
)

# Initialize modules
@st.cache_resource
def load_modules():
    return {
        'scraper': NewsScraper(),
        'sentiment': SentimentAnalyzer()
    }

modules = load_modules()

# Page Header with FinBERT branding
st.title("🤖 FinBERT Financial News Sentiment Analysis")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 10px; border-radius: 10px; margin: 10px 0;">
    <p style="color: white; text-align: center; margin: 0;">
        <strong>Real-time sentiment analysis using FinBERT - specialized for financial news</strong>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Analysis Controls")
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Stock Symbol",
        ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'SPY', 'QQQ'],
        index=0,
        help="Select a stock symbol to analyze"
    )
    
    # Real-time updates
    auto_refresh = st.checkbox("🔄 Auto-refresh (30 seconds)", value=False)
    
    # News settings
    st.subheader("📰 News Settings")
    num_articles = st.slider(
        "Number of articles",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of recent news articles to analyze"
    )
    
    st.markdown("---")
    
    # Analysis options
    st.subheader("🔍 Analysis Options")
    show_detailed = st.checkbox("Show detailed analysis", value=True)
    show_timeline = st.checkbox("Show sentiment timeline", value=True)
    
    st.markdown("---")
    
    # Refresh button
    if st.button("🚀 Analyze Now", type="primary", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Real-time News Display - ONLY ONE SECTION
    st.subheader(f"📰 Real-time News for {symbol}")
    
    # Fetch news
    with st.spinner(f"Fetching latest news for {symbol}..."):
        news_items = modules['scraper'].get_news(symbol, max_news=num_articles)
    
    if news_items:
        st.success(f"✅ Found {len(news_items)} recent news articles")
        
        # Analyze sentiment with FinBERT
        with st.spinner("🔍 Analyzing sentiment with FinBERT..."):
            analyzed_news = modules['sentiment'].analyze_news_sentiment(news_items)
        
        # Display news using Streamlit components only
        for i, news in enumerate(analyzed_news):
            # Create a container for each news item
            with st.container():
                # Determine sentiment style
                if news['sentiment'] == 'POSITIVE':
                    sentiment_emoji = "📈"
                    badge_style = "success"
                elif news['sentiment'] == 'NEGATIVE':
                    sentiment_emoji = "📉"
                    badge_style = "error"
                else:
                    sentiment_emoji = "➖"
                    badge_style = "info"
                
                # Format time
                time_diff = datetime.now() - news['published']
                if time_diff.days > 0:
                    time_ago = f"{time_diff.days}d ago"
                elif time_diff.seconds > 3600:
                    time_ago = f"{time_diff.seconds // 3600}h ago"
                elif time_diff.seconds > 60:
                    time_ago = f"{time_diff.seconds // 60}m ago"
                else:
                    time_ago = "Just now"
                
                # Create columns for layout
                col_title, col_sentiment = st.columns([4, 1])
                
                with col_title:
                    # Display title with emoji
                    st.markdown(f"#### {sentiment_emoji} {news['title']}")
                
                with col_sentiment:
                    # Display sentiment badge
                    if badge_style == "success":
                        st.success(news['sentiment'])
                    elif badge_style == "error":
                        st.error(news['sentiment'])
                    else:
                        st.info(news['sentiment'])
                
                # Publisher, source, and time
                st.caption(f"📰 **{news.get('publisher', 'Unknown')}** • 🏷️ {news.get('source', 'Unknown')} • 🕐 {time_ago}")
                
                # Summary
                if news.get('summary') and len(news['summary']) > 10:
                    st.write(news['summary'])
                
                # Create columns for metrics and link
                col_metrics, col_link = st.columns([3, 1])
                
                with col_metrics:
                    # Create three columns for metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric("🎯 Score", f"{news['score']:.3f}")
                    
                    with metric_col2:
                        confidence = news.get('confidence', 0)
                        st.metric("⚡ Confidence", f"{confidence:.1%}")
                    
                    with metric_col3:
                        model = news.get('analysis_model', 'FinBERT')
                        st.metric("🤖 Model", model)
                
                with col_link:
                    # Display link button
                    if news.get('link') and news['link'].startswith('http'):
                        st.markdown(f"[🔗 Read Article]({news['link']})")
                    else:
                        st.write("🔗 Link unavailable")
                
                st.markdown("---")
        
        # News statistics
        st.subheader("📊 News Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            total_news = len(analyzed_news)
            st.metric("Total Articles", total_news)
        
        with col_stat2:
            positive_count = sum(1 for news in analyzed_news if news['sentiment'] == 'POSITIVE')
            st.metric("Positive", positive_count, f"{(positive_count/total_news*100):.0f}%")
        
        with col_stat3:
            negative_count = sum(1 for news in analyzed_news if news['sentiment'] == 'NEGATIVE')
            st.metric("Negative", negative_count, f"{(negative_count/total_news*100):.0f}%")
        
        with col_stat4:
            avg_score = np.mean([news['score'] for news in analyzed_news])
            st.metric("Avg Score", f"{avg_score:.3f}")
        
    else:
        st.error("❌ No news articles found!")
        st.info("""
        **Try these steps:**
        1. Check your internet connection
        2. Try a different stock symbol
        3. Click the refresh button
        4. The news API might be temporarily unavailable
        """)

with col2:
    # Overall Sentiment Analysis
    st.subheader("🎯 Overall Sentiment")
    
    with st.spinner("Calculating overall sentiment..."):
        sentiment_summary = modules['sentiment'].get_sentiment_summary(symbol, news_items if 'news_items' in locals() else None)
    
    # Sentiment gauge
    score = sentiment_summary['overall_score']
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{symbol} Sentiment"},
        gauge={
            'axis': {'range': [-100, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [-100, -50], 'color': "#ef4444"},
                {'range': [-50, -20], 'color': "#f97316"},
                {'range': [-20, 20], 'color': "#6b7280"},
                {'range': [20, 50], 'color': "#22c55e"},
                {'range': [50, 100], 'color': "#16a34a"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))
    
    fig.update_layout(height=250)
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation with styling
    sentiment = sentiment_summary['overall_sentiment']
    recommendation = sentiment_summary['recommendation']
    
    if 'STRONG BUY' in recommendation:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">🎯 {recommendation}</div>
            <div style="font-size: 14px; margin-top: 5px;">{sentiment}</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">Score: {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    elif 'BUY' in recommendation:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                    padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">🎯 {recommendation}</div>
            <div style="font-size: 14px; margin-top: 5px;">{sentiment}</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">Score: {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    elif 'STRONG SELL' in recommendation:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                    padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">🎯 {recommendation}</div>
            <div style="font-size: 14px; margin-top: 5px;">{sentiment}</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">Score: {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    elif 'SELL' in recommendation:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
                    padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">🎯 {recommendation}</div>
            <div style="font-size: 14px; margin-top: 5px;">{sentiment}</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">Score: {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
                    padding: 15px; border-radius: 10px; color: white; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;">🎯 {recommendation}</div>
            <div style="font-size: 14px; margin-top: 5px;">{sentiment}</div>
            <div style="font-size: 12px; margin-top: 5px; opacity: 0.9;">Score: {score:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sentiment Components
    st.subheader("📊 Sentiment Components")
    
    components = sentiment_summary.get('components', {})
    
    if components:
        for comp_name, comp_data in components.items():
            comp_score = comp_data.get('score', 0)
            weight = comp_data.get('weight', 0)
            
            # Create mini progress bar
            progress = min(max((comp_score + 1) / 2, 0), 1)  # Convert -1 to 1 range to 0 to 1
            
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-size: 13px; color: #d1d5db;">
                        {comp_name.replace('_', ' ').title()}
                    </span>
                    <span style="font-size: 13px; color: #d1d5db;">
                        {comp_score:.3f}
                    </span>
                </div>
                <div style="background-color: #374151; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="width: {progress*100}%; 
                                height: 100%; 
                                background: linear-gradient(90deg, #ef4444 0%, #f97316 50%, #22c55e 100%);">
                    </div>
                </div>
                <div style="font-size: 11px; color: #9ca3af; text-align: right; margin-top: 2px;">
                    Weight: {weight:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No component data available")

# Sentiment Timeline
if show_timeline and 'sentiment_summary' in locals():
    st.markdown("---")
    st.subheader("📈 Sentiment Timeline")
    
    with st.spinner("Generating sentiment timeline..."):
        sentiment_df = modules['sentiment'].calculate_sentiment_indicator(symbol, days=30)
    
    if not sentiment_df.empty:
        fig = go.Figure()
        
        # Sentiment line
        fig.add_trace(go.Scatter(
            x=sentiment_df['date'],
            y=sentiment_df['sentiment'],
            mode='lines',
            name='Daily Sentiment',
            line=dict(color='#3b82f6', width=2),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        # Rolling mean
        fig.add_trace(go.Scatter(
            x=sentiment_df['date'],
            y=sentiment_df['rolling_mean'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#f97316', width=1.5, dash='dash')
        ))
        
        # Current sentiment line
        fig.add_hline(y=score, line_dash="dot", line_color="white", opacity=0.7,
                     annotation_text=f"Current: {score:.3f}")
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="#6b7280", opacity=0.5)
        
        fig.update_layout(
            title=f"{symbol} Sentiment Trend (30 Days)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            template="plotly_dark",
            height=400,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Auto-refresh functionality
if auto_refresh:
    st.markdown("---")
    st.info("🔄 Auto-refresh enabled - Next update in 30 seconds")
    st.rerun()

# Footer with FinBERT info
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 12px;">
    <p>🤖 <strong>Powered by FinBERT</strong> - A BERT model fine-tuned on financial news for accurate sentiment analysis</p>
    <p>⚠️ <strong>Disclaimer:</strong> This analysis is for informational purposes only. 
    Always conduct your own research before making investment decisions.</p>
</div>
""", unsafe_allow_html=True)