import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Strategy Settings",
    page_icon="⚙️",
    layout="wide"
)

# Page Header
st.title("⚙️ Strategy Settings")
st.markdown("Configure your trading strategy parameters")

st.markdown("---")

# Initialize session state for settings
if 'strategy_settings' not in st.session_state:
    st.session_state.strategy_settings = {
        'technical': {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2.0,
            'sma_short': 20,
            'sma_long': 50,
            'atr_period': 14
        },
        'sentiment': {
            'weight': 0.3,
            'positive_threshold': 0.3,
            'negative_threshold': -0.3,
            'strong_bullish': 0.6,
            'strong_bearish': -0.6
        },
        'risk': {
            'max_position_size': 0.1,
            'risk_per_trade': 0.02,
            'stop_loss_pct': 0.05,
            'take_profit_ratio': 2.0,
            'max_daily_loss': 0.05,
            'max_portfolio_risk': 0.25
        },
        'execution': {
            'paper_trading': True,
            'use_stop_loss': True,
            'use_take_profit': True,
            'trailing_stop': False,
            'trail_percent': 0.03,
            'slippage': 0.001,
            'commission': 0.001
        }
    }

# Create tabs for different setting categories
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Technical Indicators",
    "😊 Sentiment Analysis",
    "🛡️ Risk Management",
    "⚡ Execution Settings"
])

with tab1:
    st.subheader("Technical Indicator Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### RSI Settings")
        rsi_overbought = st.slider(
            "RSI Overbought Level",
            min_value=50,
            max_value=90,
            value=int(st.session_state.strategy_settings['technical']['rsi_overbought']),
            step=1,
            help="RSI level considered overbought (sell signal)"
        )
        
        rsi_oversold = st.slider(
            "RSI Oversold Level",
            min_value=10,
            max_value=50,
            value=int(st.session_state.strategy_settings['technical']['rsi_oversold']),
            step=1,
            help="RSI level considered oversold (buy signal)"
        )
        
        st.markdown("#### MACD Settings")
        macd_fast = st.slider(
            "MACD Fast Period",
            min_value=5,
            max_value=20,
            value=int(st.session_state.strategy_settings['technical']['macd_fast']),
            step=1
        )
        
        macd_slow = st.slider(
            "MACD Slow Period",
            min_value=20,
            max_value=50,
            value=int(st.session_state.strategy_settings['technical']['macd_slow']),
            step=1
        )
        
        macd_signal = st.slider(
            "MACD Signal Period",
            min_value=5,
            max_value=20,
            value=int(st.session_state.strategy_settings['technical']['macd_signal']),
            step=1
        )
    
    with col2:
        st.markdown("#### Bollinger Bands")
        bb_period = st.slider(
            "BB Period",
            min_value=10,
            max_value=50,
            value=int(st.session_state.strategy_settings['technical']['bb_period']),
            step=1
        )
        
        bb_std = st.slider(
            "BB Standard Deviations",
            min_value=1.0,
            max_value=3.0,
            value=float(st.session_state.strategy_settings['technical']['bb_std']),
            step=0.1,
            format="%.1f"
        )
        
        st.markdown("#### Moving Averages")
        sma_short = st.slider(
            "Short SMA Period",
            min_value=5,
            max_value=50,
            value=int(st.session_state.strategy_settings['technical']['sma_short']),
            step=1
        )
        
        sma_long = st.slider(
            "Long SMA Period",
            min_value=50,
            max_value=200,
            value=int(st.session_state.strategy_settings['technical']['sma_long']),
            step=1
        )
        
        st.markdown("#### ATR Settings")
        atr_period = st.slider(
            "ATR Period",
            min_value=5,
            max_value=30,
            value=int(st.session_state.strategy_settings['technical']['atr_period']),
            step=1
        )
    
    # Update technical settings
    st.session_state.strategy_settings['technical'].update({
        'rsi_overbought': rsi_overbought,
        'rsi_oversold': rsi_oversold,
        'macd_fast': macd_fast,
        'macd_slow': macd_slow,
        'macd_signal': macd_signal,
        'bb_period': bb_period,
        'bb_std': bb_std,
        'sma_short': sma_short,
        'sma_long': sma_long,
        'atr_period': atr_period
    })
    
    # Visualize RSI levels
    st.markdown("---")
    st.subheader("RSI Level Visualization")
    
    fig = go.Figure()
    
    # Add RSI range
    fig.add_hrect(
        y0=0, y1=rsi_oversold,
        fillcolor="green", opacity=0.2,
        layer="below", line_width=0,
        annotation_text=f"Oversold (<{rsi_oversold})"
    )
    
    fig.add_hrect(
        y0=rsi_oversold, y1=rsi_overbought,
        fillcolor="yellow", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Neutral"
    )
    
    fig.add_hrect(
        y0=rsi_overbought, y1=100,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        annotation_text=f"Overbought (>{rsi_overbought})"
    )
    
    # Add reference lines
    fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.5)
    fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green")
    fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="RSI Signal Zones",
        xaxis_title="Time",
        yaxis_title="RSI Value",
        yaxis_range=[0, 100],
        height=300,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sentiment Analysis Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sentiment Weighting")
        sentiment_weight = st.slider(
            "Sentiment Weight in Strategy",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.strategy_settings['sentiment']['weight']),
            step=0.05,
            help="How much weight to give sentiment vs technical analysis"
        )
        
        st.markdown("#### Sentiment Thresholds")
        positive_threshold = st.slider(
            "Positive Sentiment Threshold",
            min_value=0.1,
            max_value=0.8,
            value=float(st.session_state.strategy_settings['sentiment']['positive_threshold']),
            step=0.05,
            help="Minimum sentiment score for buy signal"
        )
        
        negative_threshold = st.slider(
            "Negative Sentiment Threshold",
            min_value=-0.8,
            max_value=-0.1,
            value=float(st.session_state.strategy_settings['sentiment']['negative_threshold']),
            step=0.05,
            help="Maximum sentiment score for sell signal"
        )
    
    with col2:
        st.markdown("#### Strong Sentiment Levels")
        strong_bullish = st.slider(
            "Strong Bullish Threshold",
            min_value=0.3,
            max_value=1.0,
            value=float(st.session_state.strategy_settings['sentiment']['strong_bullish']),
            step=0.05,
            help="Sentiment score for strong buy signal"
        )
        
        strong_bearish = st.slider(
            "Strong Bearish Threshold",
            min_value=-1.0,
            max_value=-0.3,
            value=float(st.session_state.strategy_settings['sentiment']['strong_bearish']),
            step=0.05,
            help="Sentiment score for strong sell signal"
        )
        
        st.markdown("#### Data Sources")
        include_news = st.checkbox("Include News Sentiment", value=True)
        include_social = st.checkbox("Include Social Media", value=True)
        include_analyst = st.checkbox("Include Analyst Ratings", value=True)
        
        if include_news:
            news_weight = st.slider("News Weight", 0.0, 1.0, 0.4, 0.1)
        else:
            news_weight = 0.0
            
        if include_social:
            social_weight = st.slider("Social Weight", 0.0, 1.0, 0.3, 0.1)
        else:
            social_weight = 0.0
            
        if include_analyst:
            analyst_weight = st.slider("Analyst Weight", 0.0, 1.0, 0.3, 0.1)
        else:
            analyst_weight = 0.0
    
    # Update sentiment settings
    st.session_state.strategy_settings['sentiment'].update({
        'weight': float(sentiment_weight),
        'positive_threshold': float(positive_threshold),
        'negative_threshold': float(negative_threshold),
        'strong_bullish': float(strong_bullish),
        'strong_bearish': float(strong_bearish),
        'include_news': include_news,
        'include_social': include_social,
        'include_analyst': include_analyst,
        'news_weight': float(news_weight),
        'social_weight': float(social_weight),
        'analyst_weight': float(analyst_weight)
    })
    
    # Visualize sentiment thresholds
    st.markdown("---")
    st.subheader("Sentiment Signal Zones")
    
    fig = go.Figure()
    
    # Add sentiment zones
    fig.add_hrect(
        y0=-1, y1=negative_threshold,
        fillcolor="red", opacity=0.3,
        layer="below", line_width=0,
        annotation_text=f"Sell Zone (<{negative_threshold:.2f})"
    )
    
    fig.add_hrect(
        y0=negative_threshold, y1=positive_threshold,
        fillcolor="yellow", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Neutral Zone"
    )
    
    fig.add_hrect(
        y0=positive_threshold, y1=1,
        fillcolor="green", opacity=0.3,
        layer="below", line_width=0,
        annotation_text=f"Buy Zone (>{positive_threshold:.2f})"
    )
    
    # Add strong sentiment zones
    if strong_bearish < negative_threshold:
        fig.add_hrect(
            y0=-1, y1=strong_bearish,
            fillcolor="darkred", opacity=0.5,
            layer="below", line_width=0,
            annotation_text=f"Strong Sell (<{strong_bearish:.2f})"
        )
    
    if strong_bullish > positive_threshold:
        fig.add_hrect(
            y0=strong_bullish, y1=1,
            fillcolor="darkgreen", opacity=0.5,
            layer="below", line_width=0,
            annotation_text=f"Strong Buy (>{strong_bullish:.2f})"
        )
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
    fig.add_hline(y=negative_threshold, line_dash="dash", line_color="red")
    fig.add_hline(y=positive_threshold, line_dash="dash", line_color="green")
    
    if strong_bearish < negative_threshold:
        fig.add_hline(y=strong_bearish, line_dash="dash", line_color="darkred")
    
    if strong_bullish > positive_threshold:
        fig.add_hline(y=strong_bullish, line_dash="dash", line_color="darkgreen")
    
    fig.update_layout(
        title="Sentiment Score Signal Zones",
        xaxis_title="Time",
        yaxis_title="Sentiment Score",
        yaxis_range=[-1, 1],
        height=300,
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Risk Management Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Position Sizing")
        max_position_size = st.slider(
            "Maximum Position Size (% of portfolio)",
            min_value=1.0,
            max_value=50.0,
            value=float(st.session_state.strategy_settings['risk']['max_position_size']) * 100,
            step=1.0,
            help="Maximum percentage of portfolio to allocate to a single position"
        ) / 100
        
        risk_per_trade = st.slider(
            "Risk per Trade (% of portfolio)",
            min_value=0.5,
            max_value=10.0,
            value=float(st.session_state.strategy_settings['risk']['risk_per_trade']) * 100,
            step=0.5,
            help="Maximum percentage of portfolio to risk on a single trade"
        ) / 100
        
        st.markdown("#### Stop Loss Settings")
        stop_loss_pct = st.slider(
            "Default Stop Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.strategy_settings['risk']['stop_loss_pct']) * 100,
            step=0.5,
            help="Percentage stop loss from entry price"
        ) / 100
        
        take_profit_ratio = st.slider(
            "Risk/Reward Ratio",
            min_value=1.0,
            max_value=5.0,
            value=float(st.session_state.strategy_settings['risk']['take_profit_ratio']),
            step=0.5,
            help="Minimum profit target relative to stop loss"
        )
    
    with col2:
        st.markdown("#### Portfolio Limits")
        max_daily_loss = st.slider(
            "Maximum Daily Loss (%)",
            min_value=1.0,
            max_value=20.0,
            value=float(st.session_state.strategy_settings['risk']['max_daily_loss']) * 100,
            step=0.5,
            help="Stop trading for the day if this loss is reached"
        ) / 100
        
        max_portfolio_risk = st.slider(
            "Maximum Portfolio Risk (%)",
            min_value=5.0,
            max_value=50.0,
            value=float(st.session_state.strategy_settings['risk']['max_portfolio_risk']) * 100,
            step=1.0,
            help="Maximum total risk across all positions"
        ) / 100
        
        st.markdown("#### Risk Controls")
        enable_circuit_breaker = st.checkbox("Enable Circuit Breaker", value=True)
        
        if enable_circuit_breaker:
            circuit_breaker_pct = st.slider(
                "Circuit Breaker Threshold (%)",
                min_value=5.0,
                max_value=20.0,
                value=10.0,
                step=0.5
            ) / 100
        else:
            circuit_breaker_pct = 0.0
        
        use_position_scaling = st.checkbox("Use Position Scaling", value=True)
        
        if use_position_scaling:
            scaling_factor = st.slider(
                "Scaling Factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
        else:
            scaling_factor = 1.0
    
    # Update risk settings
    st.session_state.strategy_settings['risk'].update({
        'max_position_size': float(max_position_size),
        'risk_per_trade': float(risk_per_trade),
        'stop_loss_pct': float(stop_loss_pct),
        'take_profit_ratio': float(take_profit_ratio),
        'max_daily_loss': float(max_daily_loss),
        'max_portfolio_risk': float(max_portfolio_risk),
        'enable_circuit_breaker': enable_circuit_breaker,
        'circuit_breaker_pct': float(circuit_breaker_pct),
        'use_position_scaling': use_position_scaling,
        'scaling_factor': float(scaling_factor)
    })
    
    # Risk visualization
    st.markdown("---")
    st.subheader("Risk Management Visualization")
    
    # Create example position sizing calculation
    example_capital = 10000
    example_price = 100
    example_stop_loss = example_price * (1 - stop_loss_pct)
    
    shares_by_risk = (example_capital * risk_per_trade) / (example_price - example_stop_loss)
    shares_by_position = (example_capital * max_position_size) / example_price
    optimal_shares = min(shares_by_risk, shares_by_position)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Shares by Risk",
            f"{int(shares_by_risk)}",
            "Based on risk per trade"
        )
    
    with col2:
        st.metric(
            "Shares by Position Size",
            f"{int(shares_by_position)}",
            "Based on max position size"
        )
    
    with col3:
        st.metric(
            "Optimal Shares",
            f"{int(optimal_shares)}",
            "Limited by both constraints"
        )
    
    # Risk/Reward visualization
    entry_price = 100
    stop_loss = entry_price * (1 - stop_loss_pct)
    take_profit = entry_price + (entry_price - stop_loss) * take_profit_ratio
    
    fig = go.Figure()
    
    # Add price levels
    fig.add_hline(y=take_profit, line_dash="dash", line_color="green", 
                  annotation_text=f"Take Profit: ${take_profit:.2f}")
    fig.add_hline(y=entry_price, line_dash="solid", line_color="yellow", 
                  annotation_text=f"Entry: ${entry_price:.2f}")
    fig.add_hline(y=stop_loss, line_dash="dash", line_color="red", 
                  annotation_text=f"Stop Loss: ${stop_loss:.2f}")
    
    # Add risk/reward zones
    fig.add_hrect(
        y0=stop_loss, y1=entry_price,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Risk Zone"
    )
    
    fig.add_hrect(
        y0=entry_price, y1=take_profit,
        fillcolor="green", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Reward Zone"
    )
    
    fig.update_layout(
        title=f"Risk/Reward Profile (Ratio: {take_profit_ratio:.1f}:1)",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        yaxis_range=[stop_loss * 0.95, take_profit * 1.05],
        height=300,
        template="plotly_dark",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Execution Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Trading Mode")
        paper_trading = st.checkbox(
            "Paper Trading Mode",
            value=st.session_state.strategy_settings['execution']['paper_trading'],
            help="Simulate trades without real money"
        )
        
        if not paper_trading:
            st.warning("⚠️ Live trading mode selected. Trades will be executed with real money!")
            
            # Broker connection settings
            st.markdown("#### Broker Connection")
            broker = st.selectbox(
                "Select Broker",
                ["Alpaca", "Interactive Brokers", "TD Ameritrade", "Custom API"]
            )
            
            if broker == "Alpaca":
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
            elif broker == "Custom API":
                api_endpoint = st.text_input("API Endpoint")
                api_key = st.text_input("API Key", type="password")
        
        st.markdown("#### Order Types")
        use_stop_loss = st.checkbox(
            "Use Stop Loss Orders",
            value=st.session_state.strategy_settings['execution']['use_stop_loss']
        )
        
        use_take_profit = st.checkbox(
            "Use Take Profit Orders",
            value=st.session_state.strategy_settings['execution']['use_take_profit']
        )
        
        trailing_stop = st.checkbox(
            "Use Trailing Stop",
            value=st.session_state.strategy_settings['execution']['trailing_stop']
        )
        
        if trailing_stop:
            trail_percent = st.slider(
                "Trailing Stop Percentage",
                min_value=0.5,
                max_value=10.0,
                value=float(st.session_state.strategy_settings['execution']['trail_percent']) * 100,
                step=0.5
            ) / 100
    
    with col2:
        st.markdown("#### Transaction Costs")
        slippage = st.slider(
            "Expected Slippage (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.strategy_settings['execution']['slippage']) * 100,
            step=0.05
        ) / 100
        
        commission = st.slider(
            "Commission per Trade (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.strategy_settings['execution']['commission']) * 100,
            step=0.05
        ) / 100
        
        st.markdown("#### Execution Rules")
        min_volume = st.number_input(
            "Minimum Daily Volume",
            min_value=1000,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Skip symbols with volume below this threshold"
        )
        
        max_spread = st.slider(
            "Maximum Bid-Ask Spread (%)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        
        trade_time = st.selectbox(
            "Preferred Trading Time",
            ["Market Open (9:30-10:30)", "Mid-day", "Market Close (15:00-16:00)", "All Day"]
        )
    
    # Update execution settings
    st.session_state.strategy_settings['execution'].update({
        'paper_trading': paper_trading,
        'broker': broker if not paper_trading else None,
        'use_stop_loss': use_stop_loss,
        'use_take_profit': use_take_profit,
        'trailing_stop': trailing_stop,
        'trail_percent': float(trail_percent) if trailing_stop else 0,
        'slippage': float(slippage),
        'commission': float(commission),
        'min_volume': int(min_volume),
        'max_spread': float(max_spread),
        'trade_time': trade_time
    })
    
    # Execution summary
    st.markdown("---")
    st.subheader("Execution Summary")
    
    summary_data = {
        "Setting": [
            "Trading Mode",
            "Stop Loss Orders",
            "Take Profit Orders",
            "Trailing Stop",
            "Expected Slippage",
            "Commission",
            "Minimum Volume",
            "Preferred Time"
        ],
        "Value": [
            "Paper Trading" if paper_trading else "Live Trading",
            "Enabled" if use_stop_loss else "Disabled",
            "Enabled" if use_take_profit else "Disabled",
            f"{trail_percent*100:.1f}%" if trailing_stop else "Disabled",
            f"{slippage*100:.3f}%",
            f"{commission*100:.3f}%",
            f"{min_volume:,}",
            trade_time
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

# Save and Reset Buttons
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("💾 Save Settings", type="primary", use_container_width=True):
        # In a real app, you would save to database or file
        st.success("✅ Settings saved successfully!")
        
        # Show summary
        with st.expander("View Saved Settings"):
            st.json(st.session_state.strategy_settings)

with col2:
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        # Reset to default settings
        st.session_state.strategy_settings = {
            'technical': {
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'bb_period': 20,
                'bb_std': 2.0,
                'sma_short': 20,
                'sma_long': 50,
                'atr_period': 14
            },
            'sentiment': {
                'weight': 0.3,
                'positive_threshold': 0.3,
                'negative_threshold': -0.3,
                'strong_bullish': 0.6,
                'strong_bearish': -0.6
            },
            'risk': {
                'max_position_size': 0.1,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_portfolio_risk': 0.25
            },
            'execution': {
                'paper_trading': True,
                'use_stop_loss': True,
                'use_take_profit': True,
                'trailing_stop': False,
                'trail_percent': 0.03,
                'slippage': 0.001,
                'commission': 0.001
            }
        }
        st.rerun()

with col3:
    st.info("ℹ️ Settings are saved in session and will be used across all pages.")

# Export/Import Settings
st.markdown("---")
st.subheader("📤 Export / Import Settings")

col1, col2 = st.columns(2)

with col1:
    # Export settings
    import json
    settings_json = json.dumps(st.session_state.strategy_settings, indent=2)
    st.download_button(
        label="📥 Export Settings",
        data=settings_json,
        file_name=f"trading_bot_settings_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )

with col2:
    # Import settings
    uploaded_file = st.file_uploader("Import Settings File", type=['json'])
    if uploaded_file is not None:
        try:
            imported_settings = json.load(uploaded_file)
            st.session_state.strategy_settings = imported_settings
            st.success("✅ Settings imported successfully!")
            st.rerun()
        except:
            st.error("❌ Error importing settings file")

# Footer
st.markdown("---")
st.caption("⚙️ Configure your trading strategy carefully. These settings directly affect your trading performance and risk exposure.")