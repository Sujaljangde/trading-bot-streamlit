import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(self,
                    prices: pd.DataFrame,
                    strategy,
                    sentiment_data: pd.DataFrame = None,
                    commission: float = 0.001,
                    slippage: float = 0.001) -> Dict:
        """Run backtest on historical data"""
        
        capital = self.initial_capital
        positions = {}
        trades = []
        portfolio_values = []
        dates = []
        
        # Convert prices to Series if DataFrame
        if isinstance(prices, pd.DataFrame):
            price_series = prices['Close'] if 'Close' in prices.columns else prices.iloc[:, 0]
        else:
            price_series = prices
        
        # Ensure we have enough data
        if len(price_series) < 100:
            return {
                'error': 'Insufficient data for backtesting',
                'min_data_points': 100,
                'available_points': len(price_series)
            }
        
        # Run simulation
        for i in range(50, len(price_series)):
            current_date = price_series.index[i]
            current_price = price_series.iloc[i]
            
            # Get historical data up to current point
            historical_data = price_series.iloc[:i+1]
            
            # Get sentiment if available
            sentiment_score = 0.0
            if sentiment_data is not None and current_date in sentiment_data.index:
                sentiment_score = sentiment_data.loc[current_date, 'sentiment']
            
            # Generate signal
            signal = strategy.generate_signal(historical_data, sentiment_score)
            
            # Execute trades based on signal
            if signal['action'] != 'HOLD' and signal['confidence'] > 0.6:
                action = signal['action']
                
                if action == 'BUY' and capital > 0:
                    # Calculate position size
                    stop_loss = current_price * 0.95  # 5% stop loss
                    position_info = strategy.calculate_position_size(
                        capital, current_price, stop_loss
                    )
                    
                    if position_info['shares'] > 0:
                        # Execute buy
                        shares = position_info['shares']
                        cost = shares * current_price
                        commission_cost = cost * commission
                        total_cost = cost + commission_cost
                        
                        if total_cost <= capital:
                            capital -= total_cost
                            positions[current_date] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'stop_loss': stop_loss,
                                'type': 'LONG'
                            }
                            
                            trades.append({
                                'date': current_date,
                                'action': 'BUY',
                                'shares': shares,
                                'price': current_price,
                                'commission': commission_cost,
                                'total': total_cost
                            })
                
                elif action == 'SELL' and positions:
                    # Close all positions (simplified)
                    for pos_date, position in list(positions.items()):
                        shares = position['shares']
                        exit_value = shares * current_price
                        commission_cost = exit_value * commission
                        net_proceeds = exit_value - commission_cost
                        
                        capital += net_proceeds
                        
                        # Calculate P&L
                        entry_value = shares * position['entry_price']
                        pl = exit_value - entry_value - commission_cost * 2
                        
                        trades.append({
                            'date': current_date,
                            'action': 'SELL',
                            'shares': shares,
                            'price': current_price,
                            'commission': commission_cost,
                            'pl': pl,
                            'pl_percent': (pl / entry_value) * 100
                        })
                    
                    # Clear positions
                    positions = {}
            
            # Calculate portfolio value
            position_value = 0
            for position in positions.values():
                position_value += position['shares'] * current_price
            
            total_value = capital + position_value
            portfolio_values.append(total_value)
            dates.append(current_date)
        
        # Close any remaining positions at the end
        if positions:
            last_price = price_series.iloc[-1]
            for position in positions.values():
                capital += position['shares'] * last_price
        
        final_value = capital
        total_trades = len(trades)
        
        # Calculate performance metrics
        if portfolio_values:
            portfolio_series = pd.Series(portfolio_values, index=dates)
            returns = portfolio_series.pct_change().dropna()
            
            # Basic metrics
            total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100
            
            if len(returns) > 0:
                annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100
                volatility = returns.std() * np.sqrt(252) * 100
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(portfolio_series)
                
                # Win rate
                if trades:
                    winning_trades = [t for t in trades if 'pl' in t and t['pl'] > 0]
                    win_rate = (len(winning_trades) / len(trades)) * 100
                else:
                    win_rate = 0
            else:
                annualized_return = 0
                volatility = 0
                sharpe_ratio = 0
                max_drawdown = 0
                win_rate = 0
        else:
            total_return = 0
            annualized_return = 0
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            win_rate = 0
        
        self.results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return,
            'total_return': final_value - self.initial_capital,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': total_trades,
            'win_rate_pct': win_rate,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'dates': dates
        }
        
        return self.results
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) == 0:
            return 0
        
        cumulative = portfolio_values
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.results:
            return {}
        
        metrics = self.results
        
        # Calculate additional metrics
        trades = metrics.get('trades', [])
        if trades:
            winning_trades = [t for t in trades if 'pl' in t and t['pl'] > 0]
            losing_trades = [t for t in trades if 'pl' in t and t['pl'] < 0]
            
            avg_win = np.mean([t['pl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t['pl']) for t in losing_trades]) if losing_trades else 0
            profit_factor = (sum([t['pl'] for t in winning_trades]) / 
                           sum([abs(t['pl']) for t in losing_trades])) if losing_trades else float('inf')
        else:
            avg_win = avg_loss = profit_factor = 0
        
        report = {
            'summary': {
                'initial_capital': f"${metrics['initial_capital']:,.2f}",
                'final_value': f"${metrics['final_value']:,.2f}",
                'net_profit': f"${metrics['total_return']:,.2f}",
                'total_return': f"{metrics['total_return_pct']:.2f}%",
                'annualized_return': f"{metrics['annualized_return_pct']:.2f}%",
                'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
                'max_drawdown': f"{metrics['max_drawdown_pct']:.2f}%",
                'volatility': f"{metrics['volatility_pct']:.2f}%"
            },
            'trading_metrics': {
                'total_trades': metrics['total_trades'],
                'win_rate': f"{metrics['win_rate_pct']:.1f}%",
                'profit_factor': f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞",
                'avg_win': f"${avg_win:.2f}" if avg_win > 0 else "$0.00",
                'avg_loss': f"${avg_loss:.2f}" if avg_loss > 0 else "$0.00",
                'largest_win': self._get_largest_trade(trades, 'win'),
                'largest_loss': self._get_largest_trade(trades, 'loss')
            },
            'risk_metrics': {
                'var_95': self._calculate_var(metrics.get('portfolio_values', []), 0.95),
                'expected_shortfall': self._calculate_expected_shortfall(metrics.get('portfolio_values', []), 0.95),
                'calmar_ratio': self._calculate_calmar_ratio(metrics['annualized_return_pct'], metrics['max_drawdown_pct']),
                'sortino_ratio': self._calculate_sortino_ratio(metrics.get('portfolio_values', []))
            }
        }
        
        return report
    
    def _get_largest_trade(self, trades: List[Dict], trade_type: str) -> str:
        """Get largest winning or losing trade"""
        if not trades:
            return "$0.00"
        
        if trade_type == 'win':
            relevant_trades = [t for t in trades if 'pl' in t and t['pl'] > 0]
        else:  # loss
            relevant_trades = [t for t in trades if 'pl' in t and t['pl'] < 0]
        
        if not relevant_trades:
            return "$0.00"
        
        largest = max(relevant_trades, key=lambda x: abs(x['pl']))
        return f"${largest['pl']:.2f}"
    
    def _calculate_var(self, values: List[float], confidence: float = 0.95) -> str:
        """Calculate Value at Risk"""
        if len(values) < 2:
            return "N/A"
        
        returns = pd.Series(values).pct_change().dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        return f"{var*100:.2f}%"
    
    def _calculate_expected_shortfall(self, values: List[float], confidence: float = 0.95) -> str:
        """Calculate Expected Shortfall (CVaR)"""
        if len(values) < 2:
            return "N/A"
        
        returns = pd.Series(values).pct_change().dropna()
        var = np.percentile(returns, (1 - confidence) * 100)
        es = returns[returns <= var].mean()
        return f"{es*100:.2f}%"
    
    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> str:
        """Calculate Calmar Ratio"""
        if max_drawdown == 0:
            return "∞"
        return f"{annual_return / abs(max_drawdown):.2f}"
    
    def _calculate_sortino_ratio(self, values: List[float]) -> str:
        """Calculate Sortino Ratio"""
        if len(values) < 2:
            return "N/A"
        
        returns = pd.Series(values).pct_change().dropna()
        target_return = 0
        downside_returns = returns[returns < target_return]
        
        if len(downside_returns) == 0:
            return "∞"
        
        downside_std = downside_returns.std()
        if downside_std == 0:
            return "∞"
        
        excess_return = returns.mean() - target_return
        sortino = excess_return / downside_std * np.sqrt(252)
        return f"{sortino:.2f}"
    
    def plot_results(self):
        """Plot backtesting results"""
        if not self.results or not self.results.get('portfolio_values'):
            return go.Figure()
        
        dates = self.results['dates']
        portfolio_values = self.results['portfolio_values']
        
        # Create portfolio value chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ))
        
        # Add buy/sell markers if trades exist
        trades = self.results.get('trades', [])
        buy_dates = [t['date'] for t in trades if t['action'] == 'BUY']
        sell_dates = [t['date'] for t in trades if t['action'] == 'SELL']
        
        if buy_dates:
            buy_values = [portfolio_values[dates.index(d)] if d in dates else None for d in buy_dates]
            fig.add_trace(go.Scatter(
                x=buy_dates,
                y=buy_values,
                mode='markers',
                name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
        
        if sell_dates:
            sell_values = [portfolio_values[dates.index(d)] if d in dates else None for d in sell_dates]
            fig.add_trace(go.Scatter(
                x=sell_dates,
                y=sell_values,
                mode='markers',
                name='Sell Signals',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
        
        fig.update_layout(
            title='Backtest Results - Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig