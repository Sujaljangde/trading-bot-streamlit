import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
from modules.backtester import Backtester
from modules.strategy import TradingStrategy
from modules.sentiment import SentimentAnalyzer

st.set_page_config(
    page_title="Strategy Backtesting",
    page_icon="🔍",
    layout="wide"
)

# Initialize modules
@st.cache_resource
def load_modules():
    return {
        'backtester': Backtester(),
        'strategy': TradingStrategy(),
        'sentiment': SentimentAnalyzer()
    }

modules = load_modules()

# Page Header
st.title("🔍 Strategy Backtesting")
st.markdown("Test your trading strategies on historical data")

st.markdown("---")

# Sidebar for backtest configuration
with st.sidebar:
    st.header("Backtest Configuration")
    
    # Symbol selection
    symbol = st.selectbox(
        "Select Symbol",
        ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX', 'SPY'],
        index=0
    )
    
    # Time period
    period = st.selectbox(
        "Time Period",
        ['1 Month', '3 Months', '6 Months', '1 Year', '2 Years', '5 Years'],
        index=2
    )
    
    # Map period to Yahoo Finance format
    period_map = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y'
    }
    
    # Initial capital
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
    
    # Risk parameters
    risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100
    max_position_size = st.slider("Max Position Size (%)", 5, 50, 10, 5) / 100
    commission = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.05) / 100
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        st.session_state.run_backtest = True

# Main content area
if 'run_backtest' in st.session_state and st.session_state.run_backtest:
    with st.spinner(f"Running backtest for {symbol}..."):
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period_map[period])
        
        if hist.empty:
            st.error(f"No historical data available for {symbol}")
        else:
            # Prepare sentiment data (mock for demonstration)
            sentiment_data = None
            if include_sentiment:
                # Generate mock sentiment time series
                dates = hist.index
                sentiment_values = np.random.uniform(-0.5, 0.5, len(dates))
                sentiment_data = pd.DataFrame({
                    'sentiment': sentiment_values
                }, index=dates)
            
            # Initialize backtester with initial capital
            backtester = Backtester(initial_capital=initial_capital)
            
            # Run backtest
            results = backtester.run_backtest(
                prices=hist,
                strategy=modules['strategy'],
                sentiment_data=sentiment_data,
                commission=commission
            )
            
            if 'error' in results:
                st.error(results['error'])
            else:
                st.session_state.backtest_results = results
                st.session_state.backtest_symbol = symbol
                st.session_state.backtest_period = period
                
                st.success("✅ Backtest completed successfully!")
    
    st.session_state.run_backtest = False

# Display results if available
if 'backtest_results' in st.session_state:
    results = st.session_state.backtest_results
    
    # Performance Summary
    st.subheader("📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"${results['total_return']:,.2f}",
            f"{results['total_return_pct']:.2f}%"
        )
    
    with col2:
        st.metric(
            "Final Value",
            f"${results['final_value']:,.2f}",
            f"from ${results['initial_capital']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Annualized Return",
            f"{results['annualized_return_pct']:.2f}%",
            f"Sharpe: {results['sharpe_ratio']:.2f}"
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown_pct']:.2f}%",
            f"Trades: {results['total_trades']}"
        )
    
    # Portfolio Value Chart
    st.markdown("---")
    st.subheader("📈 Portfolio Value Over Time")
    
    fig = modules['backtester'].plot_results()
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Metrics
    st.markdown("---")
    st.subheader("📋 Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Trading Metrics
        st.markdown("#### Trading Metrics")
        
        trading_metrics = {
            "Total Trades": results['total_trades'],
            "Win Rate": f"{results['win_rate_pct']:.1f}%",
            "Profit Factor": "N/A",  # Will be calculated in report
            "Avg Win": "N/A",
            "Avg Loss": "N/A",
            "Largest Win": "N/A",
            "Largest Loss": "N/A"
        }
        
        for metric, value in trading_metrics.items():
            st.metric(metric, value)
    
    with col2:
        # Risk Metrics
        st.markdown("#### Risk Metrics")
        
        risk_metrics = {
            "Volatility": f"{results['volatility_pct']:.2f}%",
            "Sharpe Ratio": f"{results['sharpe_ratio']:.2f}",
            "Max Drawdown": f"{results['max_drawdown_pct']:.2f}%",
            "Calmar Ratio": "N/A",
            "Sortino Ratio": "N/A",
            "VaR (95%)": "N/A"
        }
        
        for metric, value in risk_metrics.items():
            st.metric(metric, value)
    
    # Trade History
    st.markdown("---")
    st.subheader("📝 Trade History")
    
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        
        # Format columns
        if 'pl' in trades_df.columns:
            trades_df['pl'] = trades_df['pl'].apply(lambda x: f"${x:.2f}")
            trades_df['pl_percent'] = trades_df['pl_percent'].apply(lambda x: f"{x:.2f}%" if not pd.isna(x) else "N/A")
        
        trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
        trades_df['commission'] = trades_df['commission'].apply(lambda x: f"${x:.2f}")
        trades_df['total'] = trades_df['total'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A")
        
        st.dataframe(trades_df, use_container_width=True)
        
        # Trade Statistics
        if 'pl' in trades_df.columns:
            winning_trades = len([t for t in results['trades'] if 'pl' in t and t['pl'] > 0])
            losing_trades = len([t for t in results['trades'] if 'pl' in t and t['pl'] < 0])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Winning Trades", winning_trades)
            
            with col2:
                st.metric("Losing Trades", losing_trades)
            
            with col3:
                if losing_trades > 0:
                    win_loss_ratio = winning_trades / losing_trades
                    st.metric("Win/Loss Ratio", f"{win_loss_ratio:.2f}")
                else:
                    st.metric("Win/Loss Ratio", "∞")
    else:
        st.info("No trades were executed during this backtest period.")
    
    # Generate Performance Report
    st.markdown("---")
    st.subheader("📄 Performance Report")
    
    if st.button("Generate Detailed Report", type="primary"):
        with st.spinner("Generating report..."):
            report = modules['backtester'].generate_performance_report()
            
            # Display report
            st.markdown("### 📊 Performance Summary")
            summary_df = pd.DataFrame([report['summary']]).T
            summary_df.columns = ['Value']
            st.dataframe(summary_df, use_container_width=True)
            
            st.markdown("### 📈 Trading Metrics")
            trading_df = pd.DataFrame([report['trading_metrics']]).T
            trading_df.columns = ['Value']
            st.dataframe(trading_df, use_container_width=True)
            
            st.markdown("### 🛡️ Risk Metrics")
            risk_df = pd.DataFrame([report['risk_metrics']]).T
            risk_df.columns = ['Value']
            st.dataframe(risk_df, use_container_width=True)
            
            # Download button for report
            report_text = f"""
            Backtest Report for {st.session_state.backtest_symbol}
            Period: {st.session_state.backtest_period}
            Initial Capital: ${initial_capital:,.2f}
            
            Performance Summary:
            {report['summary']}
            
            Trading Metrics:
            {report['trading_metrics']}
            
            Risk Metrics:
            {report['risk_metrics']}
            """
            
            st.download_button(
                label="📥 Download Report",
                data=report_text,
                file_name=f"backtest_report_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Strategy Comparison
    st.markdown("---")
    st.subheader("⚖️ Strategy Comparison")
    
    if st.button("Compare with Buy & Hold"):
        with st.spinner("Running comparison..."):
            # Buy & Hold strategy
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period_map[period])
            
            if not hist.empty:
                # Calculate Buy & Hold returns
                initial_price = hist['Close'].iloc[0]
                final_price = hist['Close'].iloc[-1]
                shares = initial_capital / initial_price
                final_value = shares * final_price
                bh_return = ((final_value - initial_capital) / initial_capital) * 100
                
                # Strategy returns
                strategy_return = results['total_return_pct']
                
                # Create comparison chart
                comparison_data = pd.DataFrame({
                    'Strategy': ['Our Strategy', 'Buy & Hold'],
                    'Return %': [strategy_return, bh_return],
                    'Final Value': [results['final_value'], final_value]
                })
                
                fig = px.bar(
                    comparison_data,
                    x='Strategy',
                    y='Return %',
                    color='Return %',
                    color_continuous_scale='RdYlGn',
                    title="Strategy vs Buy & Hold"
                )
                
                fig.update_layout(
                    height=400,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Comparison metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Strategy Return",
                        f"{strategy_return:.2f}%",
                        f"vs {bh_return:.2f}%"
                    )
                
                with col2:
                    outperformance = strategy_return - bh_return
                    st.metric(
                        "Outperformance",
                        f"{outperformance:.2f}%",
                        "Strategy vs B&H"
                    )
                
                with col3:
                    st.metric(
                        "Strategy Final Value",
                        f"${results['final_value']:,.2f}",
                        f"B&H: ${final_value:,.2f}"
                    )

else:
    # Initial state - instructions
    st.info("👈 Configure backtest parameters in the sidebar and click 'Run Backtest'")
    
    # Show example backtest results
    st.subheader("📊 Example Backtest Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Typical Backtest Output:**
        
        - **Total Return**: 15.3%
        - **Annualized Return**: 18.7%
        - **Sharpe Ratio**: 1.2
        - **Max Drawdown**: -8.5%
        - **Win Rate**: 62.4%
        - **Total Trades**: 45
        """)
    
    with col2:
        st.markdown("""
        **What Backtesting Tells You:**
        
        1. **Strategy Viability**: Does the strategy work historically?
        2. **Risk Metrics**: Understand potential losses
        3. **Win Rate**: Probability of successful trades
        4. **Profit Factor**: Risk vs reward ratio
        5. **Drawdowns**: Maximum loss periods
        """)
    
    # Backtesting best practices
    st.markdown("---")
    st.subheader("📚 Backtesting Best Practices")
    
    practices = """
    1. **Use Sufficient Data**: Test across different market conditions
    2. **Include Transaction Costs**: Account for commissions and slippage
    3. **Avoid Look-Ahead Bias**: Only use data available at decision time
    4. **Test Out-of-Sample**: Validate on unseen data
    5. **Consider Market Regimes**: Bull, bear, and sideways markets
    6. **Monitor Drawdowns**: Ensure risk is within tolerance
    7. **Check Robustness**: Test with different parameters
    """
    
    st.markdown(practices)

# Footer
st.markdown("---")
st.caption("⚠️ Past performance does not guarantee future results. Backtesting is for educational purposes only.")