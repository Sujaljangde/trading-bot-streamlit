import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Collection of technical indicators for trading analysis"""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Moving Average Convergence Divergence"""
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: int = 2) -> Dict:
        """Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bandwidth = (upper_band - lower_band) / sma * 100
        bb_percent = (prices - lower_band) / (upper_band - lower_band) * 100
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bandwidth': bandwidth,
            'percent': bb_percent
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                             k_period: int = 14, d_period: int = 3) -> Dict:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_line = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d_line = k_line.rolling(window=d_period).mean()
        
        return {
            'k': k_line,
            'd': d_line
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = (typical_price - sma_tp).abs().rolling(window=period).mean()
        return (typical_price - sma_tp) / (0.015 * mean_deviation)
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index"""
        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = low.diff().abs()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth the DMs
        plus_dm_smooth = pd.Series(plus_dm, index=high.index).rolling(window=period).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=high.index).rolling(window=period).mean()
        
        # Calculate True Range
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / atr)
        minus_di = 100 * (minus_dm_smooth / atr)
        
        # Calculate DX and ADX
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def ichimoku_cloud(high: pd.Series, low: pd.Series, 
                       tenkan_period: int = 9, 
                       kijun_period: int = 26,
                       senkou_span_b_period: int = 52,
                       displacement: int = 26) -> Dict:
        """Ichimoku Cloud"""
        tenkan_sen = (high.rolling(window=tenkan_period).max() + 
                     low.rolling(window=tenkan_period).min()) / 2
        kijun_sen = (high.rolling(window=kijun_period).max() + 
                    low.rolling(window=kijun_period).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
        senkou_span_b = ((high.rolling(window=senkou_span_b_period).max() + 
                         low.rolling(window=senkou_span_b_period).min()) / 2).shift(displacement)
        chikou_span = high.shift(-displacement)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict:
        """Fibonacci Retracement Levels"""
        diff = high - low
        return {
            '0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100%': low
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict:
        """Pivot Points"""
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2 * (pivot - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }