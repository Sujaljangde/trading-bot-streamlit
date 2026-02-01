import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'selected_symbols' not in st.session_state:
    st.session_state.selected_symbols = ['AAPL', 'GOOGL', 'MSFT']
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 10000

# Simple News Scraper
def get_news(symbol):
    """Get news for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news[:5]
        return news if news else []
    except:
        return []

# Simple Sentiment Analysis
def analyze_sentiment(text):
    """Simple sentiment analysis"""
    positive_words = ['up', 'rise', 'gain', 'bullish', 'positive', 'strong', 'beat', 'growth']
    negative_words = ['down', 'fall', 'drop', 'bearish', 'negative', 'weak', 'miss', 'decline']
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count + negative_count == 0:
        return 0
    return (positive_count - negative_count) / (positive_count + negative_count)

# Simple Technical Indicators
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD"""
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Sidebar
with st.sidebar:
    st.title("🤖 Trading Bot")
    
    st.markdown("---")
    
    # Bot Control
    if st.button("▶️ Start Bot" if not st.session_state.bot_running else "⏸️ Pause Bot"):
        st.session_state.bot_running = not st.session_state.bot_running
        st.rerun()
    
    st.markdown("---")
    
    # Symbol Selection
    st.subheader("Select Symbols")
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']
    st.session_state.selected_symbols = st.multiselect(
        "Choose stocks to track",
        symbols,
        default=st.session_state.selected_symbols
    )
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    st.session_state.portfolio_value = st.number_input(
        "Initial Capital ($)",
        value=st.session_state.portfolio_value,
        min_value=1000,
        step=1000
    )
    
    risk_level = st.select_slider(
        "Risk Level",
        options=['Low', 'Medium', 'High'],
        value='Medium'
    )

# Main Dashboard
st.markdown('<h1 class="main-header">Trading Bot Dashboard</h1>', unsafe_allow_html=True)

# Top Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "🟢 Running" if st.session_state.bot_running else "🔴 Stopped"
        st.metric("Bot Status", status)
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Symbols Tracking", len(st.session_state.selected_symbols))
        st.markdown('</div>', unsafe_allow_html=True)

with col4:
    with st.container():
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Risk Level", risk_level)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["📊 Market Data", "📈 Charts", "📰 News"])

with tab1:
    # Market Data Table
    st.subheader("Market Overview")
    
    if st.session_state.selected_symbols:
        data = []
        for symbol in st.session_state.selected_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                    open_price = hist['Open'].iloc[-1]
                    change = price - open_price
                    change_pct = (change / open_price) * 100
                    
                    data.append({
                        'Symbol': symbol,
                        'Price': f"${price:.2f}",
                        'Change': f"{change:+.2f}",
                        'Change %': f"{change_pct:+.2f}%",
                        'Volume': f"{hist['Volume'].iloc[-1]:,}"
                    })
            except:
                continue
        
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("Could not fetch market data")
    else:
        st.info("Select symbols from the sidebar")

with tab2:
    # Price Charts
    st.subheader("Price Charts")
    
    if st.session_state.selected_symbols:
        selected_symbol = st.selectbox(
            "Select Symbol",
            st.session_state.selected_symbols
        )
        
        if selected_symbol:
            period = st.selectbox(
                "Time Period",
                ['1d', '5d', '1mo', '3mo', '6mo', '1y'],
                index=2
            )
            
            ticker = yf.Ticker(selected_symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                # Create chart
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close'],
                    name=selected_symbol
                ))
                
                # Add moving average
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['MA20'],
                    name='MA20',
                    line=dict(color='orange', width=1)
                ))
                
                fig.update_layout(
                    title=f"{selected_symbol} Price Chart",
                    yaxis_title="Price ($)",
                    xaxis_title="Date",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Indicators
                st.subheader("Technical Indicators")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    rsi = calculate_rsi(hist['Close'])
                    current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                    st.metric("RSI (14)", f"{current_rsi:.2f}")
                
                with col2:
                    macd, signal = calculate_macd(hist['Close'])
                    current_macd = macd.iloc[-1] if not macd.empty else 0
                    st.metric("MACD", f"{current_macd:.2f}")
                
                with col3:
                    sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                    current_price = hist['Close'].iloc[-1]
                    ma_position = (current_price - sma_20) / sma_20 * 100
                    st.metric("Price vs MA20", f"{ma_position:.1f}%")
            else:
                st.warning("No data available for this symbol")

with tab3:
    # News Section
    st.subheader("Market News")
    
    if st.session_state.selected_symbols:
        selected_symbol = st.selectbox(
            "Select Symbol for News",
            st.session_state.selected_symbols,
            key="news_symbol"
        )
        
        if st.button("Fetch News"):
            with st.spinner("Loading news..."):
                news = get_news(selected_symbol)
                
                if news:
                    for i, item in enumerate(news[:5]):
                        with st.expander(f"News {i+1}: {item.get('title', 'No title')[:100]}..."):
                            st.write(f"**Title:** {item.get('title', 'N/A')}")
                            st.write(f"**Publisher:** {item.get('publisher', 'N/A')}")
                            
                            # Simple sentiment analysis
                            title = item.get('title', '')
                            sentiment = analyze_sentiment(title)
                            
                            if sentiment > 0.1:
                                st.success(f"Sentiment: Positive ({sentiment:.2f})")
                            elif sentiment < -0.1:
                                st.error(f"Sentiment: Negative ({sentiment:.2f})")
                            else:
                                st.info(f"Sentiment: Neutral ({sentiment:.2f})")
                            
                            if item.get('link'):
                                st.markdown(f"[Read more]({item['link']})")
                else:
                    st.warning("No news found for this symbol")
    else:
        st.info("Select symbols from the sidebar")

# Generate Trading Signals
st.markdown("---")
st.subheader("Trading Signals")

if st.button("Generate Trading Signals"):
    signals = []
    
    for symbol in st.session_state.selected_symbols:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1mo')
            
            if len(hist) > 20:
                # Technical Analysis
                rsi = calculate_rsi(hist['Close'])
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                
                # News Sentiment
                news = get_news(symbol)
                sentiment_scores = [analyze_sentiment(item.get('title', '')) for item in news[:3]]
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                
                # Generate signal
                if current_rsi < 30 and avg_sentiment > 0:
                    signals.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'reason': f'RSI oversold ({current_rsi:.1f}), Positive sentiment',
                        'price': hist['Close'].iloc[-1]
                    })
                elif current_rsi > 70 and avg_sentiment < 0:
                    signals.append({
                        'symbol': symbol,
                        'action': 'SELL',
                        'reason': f'RSI overbought ({current_rsi:.1f}), Negative sentiment',
                        'price': hist['Close'].iloc[-1]
                    })
        except:
            continue
    
    if signals:
        st.success(f"Generated {len(signals)} trading signals!")
        
        for signal in signals:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                if signal['action'] == 'BUY':
                    st.success(f"**{signal['action']}**")
                else:
                    st.error(f"**{signal['action']}**")
            with col2:
                st.write(f"**{signal['symbol']}** - {signal['reason']}")
            with col3:
                st.write(f"${signal['price']:.2f}")
    else:
        st.info("No strong trading signals detected")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    Trading Bot Dashboard | For Educational Purposes Only
</div>
""", unsafe_allow_html=True)