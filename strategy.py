import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TradingStrategy:
    """Trading strategy engine combining technical and sentiment analysis"""
    
    def __init__(self):
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_signal_period = 9
        self.bb_std = 2
        self.atr_period = 14
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal_period, adjust=False).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * self.bb_std)
        lower_band = sma - (std * self.bb_std)
        return upper_band, lower_band
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.atr_period).mean()
        return atr
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict:
        """Calculate various moving averages"""
        return {
            'sma_20': prices.rolling(window=20).mean(),
            'sma_50': prices.rolling(window=50).mean(),
            'sma_200': prices.rolling(window=200).mean(),
            'ema_12': prices.ewm(span=12, adjust=False).mean(),
            'ema_26': prices.ewm(span=26, adjust=False).mean()
        }
    
    def generate_signal(self, 
                       prices: pd.Series, 
                       sentiment_score: float = 0.0,
                       high: pd.Series = None,
                       low: pd.Series = None) -> Dict:
        """Generate trading signal based on technical analysis and sentiment"""
        
        if len(prices) < 50:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': 'Insufficient data'
            }
        
        current_price = prices.iloc[-1]
        
        # Calculate indicators
        rsi = self.calculate_rsi(prices)
        macd, signal = self.calculate_macd(prices)
        
        if high is not None and low is not None:
            bb_upper, bb_lower = self.calculate_bollinger_bands(prices)
            atr = self.calculate_atr(high, low, prices)
        else:
            bb_upper, bb_lower = self.calculate_bollinger_bands(prices)
            atr = pd.Series([0] * len(prices))
        
        mas = self.calculate_moving_averages(prices)
        
        # Initialize signal components
        signals = []
        confidence = 0.0
        
        # RSI signals
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        if current_rsi < self.rsi_oversold:
            signals.append(('RSI Oversold', 'BUY', 0.7))
        elif current_rsi > self.rsi_overbought:
            signals.append(('RSI Overbought', 'SELL', 0.7))
        
        # MACD signals
        current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
        current_signal = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
        prev_macd = macd.iloc[-2] if len(macd) > 1 else 0
        prev_signal = signal.iloc[-2] if len(signal) > 1 else 0
        
        if current_macd > current_signal and prev_macd <= prev_signal:
            signals.append(('MACD Bullish Crossover', 'BUY', 0.6))
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signals.append(('MACD Bearish Crossover', 'SELL', 0.6))
        
        # Bollinger Bands signals
        current_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.1
        current_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.9
        
        if current_price < current_bb_lower:
            signals.append(('Below Lower Bollinger Band', 'BUY', 0.5))
        elif current_price > current_bb_upper:
            signals.append(('Above Upper Bollinger Band', 'SELL', 0.5))
        
        # Moving Average signals
        sma_20 = mas['sma_20'].iloc[-1] if not pd.isna(mas['sma_20'].iloc[-1]) else current_price
        sma_50 = mas['sma_50'].iloc[-1] if not pd.isna(mas['sma_50'].iloc[-1]) else current_price
        
        if current_price > sma_20 and sma_20 > sma_50:
            signals.append(('Golden Cross Formation', 'BUY', 0.8))
        elif current_price < sma_20 and sma_20 < sma_50:
            signals.append(('Death Cross Formation', 'SELL', 0.8))
        
        # Sentiment signals
        if sentiment_score > 0.3:
            signals.append(('Positive Sentiment', 'BUY', abs(sentiment_score)))
        elif sentiment_score < -0.3:
            signals.append(('Negative Sentiment', 'SELL', abs(sentiment_score)))
        
        # Aggregate signals
        if not signals:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reason': 'No clear signals detected'
            }
        
        # Calculate weighted signals
        buy_signals = [s for s in signals if s[1] == 'BUY']
        sell_signals = [s for s in signals if s[1] == 'SELL']
        
        buy_score = sum([s[2] for s in buy_signals]) / len(buy_signals) if buy_signals else 0
        sell_score = sum([s[2] for s in sell_signals]) / len(sell_signals) if sell_signals else 0
        
        # Determine final action
        if buy_score > sell_score and buy_score >= 0.5:
            action = 'BUY'
            confidence = buy_score
            reason = f"Multiple buy signals: {', '.join([s[0] for s in buy_signals])}"
        elif sell_score > buy_score and sell_score >= 0.5:
            action = 'SELL'
            confidence = sell_score
            reason = f"Multiple sell signals: {', '.join([s[0] for s in sell_signals])}"
        else:
            action = 'HOLD'
            confidence = max(buy_score, sell_score)
            reason = "Mixed or weak signals"
        
        return {
            'action': action,
            'confidence': min(confidence, 1.0),
            'price': current_price,
            'reason': reason,
            'signals': signals,
            'indicators': {
                'rsi': current_rsi,
                'macd': current_macd,
                'bb_position': (current_price - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
        }
    
    def calculate_position_size(self, 
                               capital: float, 
                               entry_price: float, 
                               stop_loss: float,
                               risk_per_trade: float = 0.02,
                               max_position_size: float = 0.1) -> Dict:
        """Calculate optimal position size based on risk parameters"""
        
        risk_amount = capital * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share <= 0:
            return {
                'shares': 0,
                'position_value': 0,
                'risk_amount': 0,
                'stop_loss': stop_loss
            }
        
        # Calculate shares based on risk
        shares_by_risk = risk_amount / risk_per_share
        
        # Limit by maximum position size
        max_shares_by_capital = (capital * max_position_size) / entry_price
        
        shares = min(shares_by_risk, max_shares_by_capital)
        shares = int(shares)  # Whole shares
        
        position_value = shares * entry_price
        actual_risk = shares * risk_per_share
        
        return {
            'shares': shares,
            'position_value': position_value,
            'risk_amount': actual_risk,
            'stop_loss': stop_loss,
            'risk_percentage': (actual_risk / capital) * 100
        }
    
    def calculate_stop_loss(self, 
                           entry_price: float, 
                           signal_type: str,
                           atr: float = None,
                           stop_loss_pct: float = 0.05) -> float:
        """Calculate stop loss price"""
        
        if atr is not None and atr > 0:
            # ATR-based stop loss
            if signal_type == 'BUY':
                stop_loss = entry_price - (atr * 2)
            else:  # SELL (short)
                stop_loss = entry_price + (atr * 2)
        else:
            # Percentage-based stop loss
            if signal_type == 'BUY':
                stop_loss = entry_price * (1 - stop_loss_pct)
            else:  # SELL (short)
                stop_loss = entry_price * (1 + stop_loss_pct)
        
        return stop_loss
    
    def calculate_take_profit(self,
                            entry_price: float,
                            signal_type: str,
                            stop_loss: float,
                            risk_reward_ratio: float = 2) -> float:
        """Calculate take profit price"""
        
        risk = abs(entry_price - stop_loss)
        
        if signal_type == 'BUY':
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:  # SELL (short)
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit