import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from modules.scraper import NewsScraper
from modules.sentiment import SentimentAnalyzer
from modules.strategy import TradingStrategy

st.set_page_config(
    page_title="Live Trading Signals",
    page_icon="📈",
    layout="wide"
)

# Initialize modules
@st.cache_resource
def load_modules():
    return {
        'scraper': NewsScraper(),
        'sentiment': SentimentAnalyzer(),
        'strategy': TradingStrategy()
    }

modules = load_modules()

# Page Header
st.title("📈 Live Trading Signals")
st.markdown("Real-time trading signals combining FinBERT sentiment analysis and technical indicators")

st.markdown("---")

# Sidebar for signal generation
with st.sidebar:
    st.header("⚙️ Signal Generator")
    
    # Symbol selection
    symbols = st.multiselect(
        "Select Symbols to Analyze",
        ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX'],
        default=['AAPL', 'GOOGL', 'MSFT'],
        help="Select stocks to generate trading signals for"
    )
    
    # Analysis parameters
    st.subheader("📊 Analysis Parameters")
    
    analysis_type = st.radio(
        "Analysis Type",
        ['Sentiment Only', 'Technical Only', 'Combined Analysis'],
        index=2,
        help="Choose what factors to include in signal generation"
    )
    
    confidence_threshold = st.slider(
        "Minimum Confidence",
        0.0, 1.0, 0.6, 0.05,
        help="Only show signals with confidence above this threshold"
    )
    
    # Risk management
    st.subheader("🎯 Risk Management")
    max_position_size = st.number_input(
        "Max Position Size ($)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )
    
    stop_loss = st.slider(
        "Stop Loss (%)",
        1.0, 20.0, 5.0, 0.5,
        help="Automatic stop loss percentage"
    )
    
    st.markdown("---")
    
    # Generate button
    if st.button("🚀 Generate Trading Signals", type="primary", use_container_width=True):
        st.session_state.generate_signals = True
        st.session_state.analysis_type = analysis_type

# Helper function to calculate sentiment score
def calculate_overall_sentiment(news_items):
    """Calculate overall sentiment score from news items"""
    if not news_items:
        return 0.0, 0.0
    
    try:
        # Analyze news sentiment
        analyzed_news = modules['sentiment'].analyze_news_sentiment(news_items)
        
        if not analyzed_news:
            return 0.0, 0.0
        
        # Calculate weighted average sentiment
        total_score = 0
        total_confidence = 0
        
        for news in analyzed_news:
            score = news.get('score', 0)
            confidence = news.get('confidence', 0)
            total_score += score * confidence
            total_confidence += confidence
        
        if total_confidence > 0:
            overall_sentiment = total_score / total_confidence
            avg_confidence = total_confidence / len(analyzed_news)
        else:
            overall_sentiment = 0.0
            avg_confidence = 0.0
        
        return overall_sentiment, avg_confidence
    
    except Exception as e:
        st.error(f"Error calculating sentiment: {e}")
        return 0.0, 0.0

# Helper function to get technical indicators
def get_technical_indicators(prices):
    """Calculate basic technical indicators"""
    try:
        # Simple moving averages
        sma_20 = prices.rolling(window=20).mean().iloc[-1]
        sma_50 = prices.rolling(window=50).mean().iloc[-1]
        
        # RSI calculation
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # MACD
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_value = macd.iloc[-1]
        macd_signal_value = macd_signal.iloc[-1]
        
        # Bollinger Bands
        bb_middle = prices.rolling(window=20).mean().iloc[-1]
        bb_std = prices.rolling(window=20).std().iloc[-1]
        bb_upper = bb_middle + (2 * bb_std)
        bb_lower = bb_middle - (2 * bb_std)
        current_price = prices.iloc[-1]
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'rsi': rsi,
            'macd': macd_value,
            'macd_signal': macd_signal_value,
            'bb_position': bb_position,
            'current_price': current_price
        }
    
    except Exception as e:
        st.error(f"Error calculating technical indicators: {e}")
        return None

# Main content - Signal Generation
if 'generate_signals' in st.session_state and st.session_state.generate_signals:
    with st.spinner("🔍 Analyzing markets and generating signals..."):
        signals = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(symbols):
            try:
                # Update progress
                progress = (i + 1) / len(symbols)
                progress_bar.progress(progress)
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
                
                # Fetch price data
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='3mo')
                
                if hist.empty or len(hist) < 50:
                    st.warning(f"⚠️ Insufficient data for {symbol}")
                    continue
                
                current_price = hist['Close'].iloc[-1]
                
                # Get sentiment analysis
                sentiment_score = 0.0
                sentiment_confidence = 0.0
                news_items = []
                
                if st.session_state.analysis_type in ['Sentiment Only', 'Combined Analysis']:
                    news_items = modules['scraper'].get_news(symbol, max_news=5)
                    sentiment_score, sentiment_confidence = calculate_overall_sentiment(news_items)
                
                # Get technical analysis
                technical_data = None
                if st.session_state.analysis_type in ['Technical Only', 'Combined Analysis']:
                    technical_data = get_technical_indicators(hist['Close'])
                
                # Generate signal based on analysis type
                signal = {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'reason': '',
                    'risk_level': 'LOW'
                }
                
                if st.session_state.analysis_type == 'Sentiment Only':
                    # Sentiment-based signal
                    if sentiment_score > 0.3 and sentiment_confidence > 0.5:
                        signal['action'] = 'BUY'
                        signal['confidence'] = sentiment_confidence
                        signal['reason'] = f"Strong positive sentiment ({sentiment_score:.2f})"
                        signal['risk_level'] = 'MEDIUM' if sentiment_score < 0.6 else 'HIGH'
                    elif sentiment_score < -0.3 and sentiment_confidence > 0.5:
                        signal['action'] = 'SELL'
                        signal['confidence'] = sentiment_confidence
                        signal['reason'] = f"Strong negative sentiment ({sentiment_score:.2f})"
                        signal['risk_level'] = 'MEDIUM' if sentiment_score > -0.6 else 'HIGH'
                    else:
                        signal['action'] = 'HOLD'
                        signal['confidence'] = sentiment_confidence
                        signal['reason'] = f"Neutral sentiment ({sentiment_score:.2f})"
                        signal['risk_level'] = 'LOW'
                
                elif st.session_state.analysis_type == 'Technical Only':
                    # Technical-based signal
                    if technical_data:
                        # Generate signal based on technical indicators
                        reasons = []
                        confidence_factors = []
                        
                        # RSI analysis
                        if technical_data['rsi'] < 30:
                            reasons.append("RSI oversold")
                            confidence_factors.append(0.7)
                        elif technical_data['rsi'] > 70:
                            reasons.append("RSI overbought")
                            confidence_factors.append(0.7)
                        
                        # MACD analysis
                        if technical_data['macd'] > technical_data['macd_signal']:
                            reasons.append("MACD bullish")
                            confidence_factors.append(0.6)
                        else:
                            reasons.append("MACD bearish")
                            confidence_factors.append(0.6)
                        
                        # Moving average analysis
                        if current_price > technical_data['sma_20'] > technical_data['sma_50']:
                            reasons.append("Strong uptrend")
                            confidence_factors.append(0.8)
                        elif current_price < technical_data['sma_20'] < technical_data['sma_50']:
                            reasons.append("Strong downtrend")
                            confidence_factors.append(0.8)
                        
                        # Bollinger Bands analysis
                        if technical_data['bb_position'] < 0.2:
                            reasons.append("Near lower Bollinger Band")
                            confidence_factors.append(0.5)
                        elif technical_data['bb_position'] > 0.8:
                            reasons.append("Near upper Bollinger Band")
                            confidence_factors.append(0.5)
                        
                        # Determine action
                        buy_signals = sum(1 for r in reasons if 'bullish' in r.lower() or 'oversold' in r.lower() or 'uptrend' in r.lower())
                        sell_signals = sum(1 for r in reasons if 'bearish' in r.lower() or 'overbought' in r.lower() or 'downtrend' in r.lower())
                        
                        if buy_signals > sell_signals and len(confidence_factors) > 0:
                            signal['action'] = 'BUY'
                            signal['confidence'] = np.mean(confidence_factors)
                            signal['reason'] = ', '.join(reasons)
                            signal['risk_level'] = 'MEDIUM' if signal['confidence'] < 0.7 else 'HIGH'
                        elif sell_signals > buy_signals and len(confidence_factors) > 0:
                            signal['action'] = 'SELL'
                            signal['confidence'] = np.mean(confidence_factors)
                            signal['reason'] = ', '.join(reasons)
                            signal['risk_level'] = 'MEDIUM' if signal['confidence'] < 0.7 else 'HIGH'
                        else:
                            signal['action'] = 'HOLD'
                            signal['confidence'] = 0.3
                            signal['reason'] = "Mixed technical signals"
                            signal['risk_level'] = 'LOW'
                
                else:  # Combined Analysis
                    # Combine sentiment and technical analysis
                    if technical_data:
                        sentiment_weight = 0.4
                        technical_weight = 0.6
                        
                        # Sentiment contribution
                        sentiment_contribution = sentiment_score * sentiment_weight
                        
                        # Technical contribution (simplified)
                        technical_score = 0
                        if technical_data['rsi'] < 30:
                            technical_score += 0.3
                        elif technical_data['rsi'] > 70:
                            technical_score -= 0.3
                        
                        if technical_data['macd'] > technical_data['macd_signal']:
                            technical_score += 0.2
                        else:
                            technical_score -= 0.2
                        
                        if current_price > technical_data['sma_20'] > technical_data['sma_50']:
                            technical_score += 0.3
                        elif current_price < technical_data['sma_20'] < technical_data['sma_50']:
                            technical_score -= 0.3
                        
                        technical_contribution = technical_score * technical_weight
                        
                        # Combined score
                        combined_score = sentiment_contribution + technical_contribution
                        combined_confidence = (sentiment_confidence * sentiment_weight + 
                                             np.mean([0.7, 0.6, 0.8]) * technical_weight)  # Average technical confidence
                        
                        # Generate signal
                        if combined_score > 0.2 and combined_confidence > 0.5:
                            signal['action'] = 'BUY'
                            signal['confidence'] = combined_confidence
                            signal['reason'] = f"Positive combined analysis (Sentiment: {sentiment_score:.2f}, Technical score: {technical_score:.2f})"
                            signal['risk_level'] = 'MEDIUM' if combined_score < 0.4 else 'HIGH'
                        elif combined_score < -0.2 and combined_confidence > 0.5:
                            signal['action'] = 'SELL'
                            signal['confidence'] = combined_confidence
                            signal['reason'] = f"Negative combined analysis (Sentiment: {sentiment_score:.2f}, Technical score: {technical_score:.2f})"
                            signal['risk_level'] = 'MEDIUM' if combined_score > -0.4 else 'HIGH'
                        else:
                            signal['action'] = 'HOLD'
                            signal['confidence'] = combined_confidence
                            signal['reason'] = f"Neutral combined analysis (Score: {combined_score:.2f})"
                            signal['risk_level'] = 'LOW'
                
                # Add signal if confidence meets threshold
                if signal['confidence'] >= confidence_threshold:
                    signals.append({
                        'symbol': symbol,
                        'action': signal['action'],
                        'confidence': signal['confidence'],
                        'price': current_price,
                        'reason': signal['reason'],
                        'sentiment': sentiment_score,
                        'sentiment_confidence': sentiment_confidence,
                        'risk_level': signal['risk_level'],
                        'timestamp': datetime.now(),
                        'analysis_type': st.session_state.analysis_type,
                        'news_count': len(news_items)
                    })
                    
            except Exception as e:
                st.error(f"Error processing {symbol}: {str(e)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if signals:
            st.session_state.signals = signals
            st.success(f"✅ Generated {len(signals)} trading signals!")
        else:
            st.warning("⚠️ No signals generated above confidence threshold")
        
        st.session_state.generate_signals = False

# Display signals if available
if 'signals' in st.session_state and st.session_state.signals:
    signals_df = pd.DataFrame(st.session_state.signals)
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        buy_signals = len([s for s in st.session_state.signals if s['action'] == 'BUY'])
        st.metric("📈 Buy Signals", buy_signals)
    
    with col2:
        sell_signals = len([s for s in st.session_state.signals if s['action'] == 'SELL'])
        st.metric("📉 Sell Signals", sell_signals)
    
    with col3:
        hold_signals = len([s for s in st.session_state.signals if s['action'] == 'HOLD'])
        st.metric("⚖️ Hold Signals", hold_signals)
    
    with col4:
        strong_signals = len(signals_df[signals_df['confidence'] > 0.8])
        st.metric("💪 Strong Signals", strong_signals)
    
    st.markdown("---")
    
    # Display signals table
    st.subheader("📋 Generated Trading Signals")
    
    # Format DataFrame for display
    display_df = signals_df.copy()
    display_df = display_df[['symbol', 'action', 'confidence', 'price', 'risk_level', 'sentiment', 'reason']]
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['sentiment'] = display_df['sentiment'].apply(lambda x: f"{x:.3f}")
    display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
    
     # Color coding function - FIXED VERSION
    def color_cells(val, col):
        if col == 'action':
            if val == 'BUY':
                return 'background-color: rgba(0, 255, 0, 0.2); color: green; font-weight: bold'
            elif val == 'SELL':
                return 'background-color: rgba(255, 0, 0, 0.2); color: red; font-weight: bold'
            else:
                return 'background-color: rgba(128, 128, 128, 0.2); color: gray; font-weight: bold'
        elif col == 'risk_level':
            if val == 'HIGH':
                return 'background-color: rgba(255, 165, 0, 0.2); color: orange; font-weight: bold'
            elif val == 'MEDIUM':
                return 'background-color: rgba(255, 255, 0, 0.2); color: #ffcc00; font-weight: bold'
            else:
                return 'background-color: rgba(0, 128, 0, 0.2); color: lightgreen; font-weight: bold'
        return ''
    
    # Apply styling column by column
    styled_df = display_df.style
    
    # Style action column
    styled_df = styled_df.map(
        lambda x: color_cells(x, 'action'), 
        subset=['action']
    )
    
    # Style risk_level column  
    styled_df = styled_df.map(
        lambda x: color_cells(x, 'risk_level'),
        subset=['risk_level']
    )
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Detailed signal analysis
    st.markdown("---")
    st.subheader("🔍 Detailed Signal Analysis")
    
    if signals_df.empty:
        st.info("No signals to analyze")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Signal Distribution", "📈 Performance Metrics", "🎯 Trade Execution"])
        
        with tab1:
            # Signal distribution chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=['BUY', 'SELL', 'HOLD'],
                    values=[buy_signals, sell_signals, hold_signals],
                    hole=0.3,
                    marker_colors=['#00ff00', '#ff0000', '#808080']
                )
            ])
            
            fig.update_layout(
                title="Signal Distribution",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk level distribution
            risk_counts = signals_df['risk_level'].value_counts()
            fig2 = go.Figure(data=[
                go.Bar(
                    x=risk_counts.index,
                    y=risk_counts.values,
                    marker_color=['orange', 'yellow', 'lightgreen']
                )
            ])
            
            fig2.update_layout(
                title="Risk Level Distribution",
                height=300,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        
        with tab2:
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Confidence", f"{signals_df['confidence'].mean():.1%}")
                st.metric("Average Sentiment", f"{signals_df['sentiment'].mean():.3f}")
                st.metric("Total News Analyzed", signals_df['news_count'].sum())
            
            with col2:
                st.metric("High Risk Signals", len(signals_df[signals_df['risk_level'] == 'HIGH']))
                st.metric("Analysis Type", st.session_state.analysis_type)
                st.metric("Signal Generation Time", signals_df['timestamp'].iloc[0].strftime('%H:%M:%S'))
            
            # Confidence vs Sentiment scatter plot
            fig3 = go.Figure(data=[
                go.Scatter(
                    x=signals_df['sentiment'],
                    y=signals_df['confidence'],
                    mode='markers',
                    marker=dict(
                        size=signals_df['confidence'] * 20,
                        color=signals_df['confidence'],
                        colorscale='RdYlGn',
                        showscale=True
                    ),
                    text=signals_df['symbol'],
                    hovertemplate='<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Confidence: %{y:.1%}'
                )
            ])
            
            fig3.update_layout(
                title="Confidence vs Sentiment",
                xaxis_title="Sentiment Score",
                yaxis_title="Confidence",
                height=400,
                template="plotly_dark"
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with tab3:
            # Trade execution simulation
            st.info("💡 This is a simulation. Connect to your broker API for live trading.")
            
            selected_symbols = st.multiselect(
                "Select signals to execute",
                [f"{s['symbol']} - {s['action']}" for s in st.session_state.signals],
                default=[f"{s['symbol']} - {s['action']}" for s in st.session_state.signals if s['action'] != 'HOLD']
            )
            
            if selected_symbols:
                execution_df = pd.DataFrame([
                    {
                        'symbol': s.split(' - ')[0],
                        'action': s.split(' - ')[1],
                        'price': next(sig['price'] for sig in st.session_state.signals if sig['symbol'] == s.split(' - ')[0]),
                        'confidence': next(sig['confidence'] for sig in st.session_state.signals if sig['symbol'] == s.split(' - ')[0]),
                        'position_size': min(max_position_size, max_position_size * 0.2)  # 20% per position
                    }
                    for s in selected_symbols
                ])
                
                st.dataframe(execution_df, use_container_width=True)
                
                total_investment = execution_df['position_size'].sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("✅ Execute Selected Trades", type="primary", use_container_width=True):
                        st.success(f"Simulated execution of {len(selected_symbols)} trades!")
                        st.balloons()
                
                with col2:
                    st.metric("Total Investment", f"${total_investment:,.2f}")
                    st.metric("Estimated Stop Loss", f"${total_investment * (stop_loss/100):,.2f}")

else:
    # Initial state - instructions
    st.info("👈 Configure parameters in the sidebar and click 'Generate Signals' to start")
    
    # Show example dashboard
    st.subheader("🎯 Example Signal Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Signal Types
        - **BUY Signals**: Generated when sentiment is positive AND technical indicators are bullish
        - **SELL Signals**: Generated when sentiment is negative AND technical indicators are bearish
        - **HOLD Signals**: Generated when signals are mixed or confidence is low
        
        ### 🎯 Confidence Levels
        - **High (>80%)**: Strong, clear signals
        - **Medium (60-80%)**: Good signals with some uncertainty
        - **Low (<60%)**: Weak signals - consider avoiding
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Analysis Methods
        **1. Sentiment Analysis**
        - Uses FinBERT NLP model
        - Analyzes recent news headlines
        - Provides -1 to 1 sentiment scores
        
        **2. Technical Analysis**
        - RSI (Relative Strength Index)
        - MACD (Moving Average Convergence Divergence)
        - Moving Averages (20 & 50 day)
        - Bollinger Bands
        
        **3. Combined Analysis**
        - Weighted combination of sentiment and technical factors
        - Most reliable signal generation
        """)
    
    st.markdown("---")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Select symbols** you want to analyze
        2. **Choose analysis type** (Combined is recommended)
        3. **Set confidence threshold** (0.6 is a good starting point)
        4. **Configure risk management** (position size, stop loss)
        5. **Click "Generate Trading Signals"**
        6. **Review signals** and execute trades if confident
        
        ⚠️ **Remember**: Always do your own research before trading!
        """)

# Footer
st.markdown("---")
st.caption("🤖 Powered by FinBERT Sentiment Analysis + Technical Indicators | ⚠️ For educational purposes only")