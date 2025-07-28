"""
Advanced Trading Signals Generator
Provides buy/sell recommendations based on multiple technical indicators
"""

import pandas as pd
import numpy as np
import ta

class TradingSignalGenerator:
    def __init__(self):
        self.signal_weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bollinger': 0.2,
            'moving_average': 0.15,
            'volume': 0.1,
            'momentum': 0.1
        }
    
    def calculate_rsi_signal(self, data):
        """RSI-based signal: Buy when oversold (<30), Sell when overbought (>70)"""
        rsi = data['RSI'].iloc[-1]
        
        if rsi < 30:
            return 1, "Strong Buy - Oversold"
        elif rsi < 40:
            return 0.5, "Buy - Approaching oversold"
        elif rsi > 70:
            return -1, "Strong Sell - Overbought"
        elif rsi > 60:
            return -0.5, "Sell - Approaching overbought"
        else:
            return 0, "Hold - Neutral RSI"
    
    def calculate_macd_signal(self, data):
        """MACD-based signal"""
        macd = data['MACD'].iloc[-1]
        macd_signal = data['MACD_signal'].iloc[-1]
        macd_hist = data['MACD_histogram'].iloc[-1]
        
        if macd > macd_signal and macd_hist > 0:
            return 1, "Strong Buy - MACD bullish crossover"
        elif macd > macd_signal:
            return 0.5, "Buy - MACD above signal"
        elif macd < macd_signal and macd_hist < 0:
            return -1, "Strong Sell - MACD bearish crossover"
        elif macd < macd_signal:
            return -0.5, "Sell - MACD below signal"
        else:
            return 0, "Hold - MACD neutral"
    
    def calculate_bollinger_signal(self, data):
        """Bollinger Bands signal"""
        current_price = data['Close'].iloc[-1]
        bb_upper = data['BB_upper'].iloc[-1]
        bb_lower = data['BB_lower'].iloc[-1]
        bb_middle = data['BB_middle'].iloc[-1]
        
        if current_price <= bb_lower:
            return 1, "Strong Buy - Price at lower Bollinger Band"
        elif current_price < bb_middle:
            return 0.5, "Buy - Price below middle band"
        elif current_price >= bb_upper:
            return -1, "Strong Sell - Price at upper Bollinger Band"
        elif current_price > bb_middle:
            return -0.5, "Sell - Price above middle band"
        else:
            return 0, "Hold - Price at middle band"
    
    def calculate_moving_average_signal(self, data):
        """Moving average crossover signal"""
        current_price = data['Close'].iloc[-1]
        ma_5 = data['MA_5'].iloc[-1]
        ma_20 = data['MA_20'].iloc[-1]
        ma_50 = data['MA_50'].iloc[-1]
        
        # Golden cross and death cross patterns
        if ma_5 > ma_20 > ma_50 and current_price > ma_5:
            return 1, "Strong Buy - Golden cross pattern"
        elif ma_5 > ma_20 and current_price > ma_5:
            return 0.5, "Buy - Short-term uptrend"
        elif ma_5 < ma_20 < ma_50 and current_price < ma_5:
            return -1, "Strong Sell - Death cross pattern"
        elif ma_5 < ma_20 and current_price < ma_5:
            return -0.5, "Sell - Short-term downtrend"
        else:
            return 0, "Hold - Mixed MA signals"
    
    def calculate_volume_signal(self, data):
        """Volume-based signal"""
        current_volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume_MA'].iloc[-1]
        volume_ratio = data['Volume_ratio'].iloc[-1]
        price_change = data['Price_change'].iloc[-1]
        
        if volume_ratio > 1.5 and price_change > 0:
            return 1, "Strong Buy - High volume with price increase"
        elif volume_ratio > 1.2 and price_change > 0:
            return 0.5, "Buy - Above average volume with gains"
        elif volume_ratio > 1.5 and price_change < 0:
            return -1, "Strong Sell - High volume with price decrease"
        elif volume_ratio > 1.2 and price_change < 0:
            return -0.5, "Sell - Above average volume with losses"
        else:
            return 0, "Hold - Normal volume activity"
    
    def calculate_momentum_signal(self, data):
        """Price momentum signal"""
        price_changes = data['Price_change'].tail(5)
        recent_momentum = price_changes.mean()
        
        if recent_momentum > 0.02:  # 2% average gain
            return 1, "Strong Buy - Strong upward momentum"
        elif recent_momentum > 0.005:  # 0.5% average gain
            return 0.5, "Buy - Positive momentum"
        elif recent_momentum < -0.02:  # 2% average loss
            return -1, "Strong Sell - Strong downward momentum"
        elif recent_momentum < -0.005:  # 0.5% average loss
            return -0.5, "Sell - Negative momentum"
        else:
            return 0, "Hold - Sideways momentum"
    
    def generate_comprehensive_signal(self, data):
        """Generate comprehensive buy/sell signal with confidence"""
        signals = {}
        explanations = {}
        
        # Calculate individual signals
        signals['rsi'], explanations['rsi'] = self.calculate_rsi_signal(data)
        signals['macd'], explanations['macd'] = self.calculate_macd_signal(data)
        signals['bollinger'], explanations['bollinger'] = self.calculate_bollinger_signal(data)
        signals['moving_average'], explanations['moving_average'] = self.calculate_moving_average_signal(data)
        signals['volume'], explanations['volume'] = self.calculate_volume_signal(data)
        signals['momentum'], explanations['momentum'] = self.calculate_momentum_signal(data)
        
        # Calculate weighted score
        weighted_score = sum(signals[key] * self.signal_weights[key] for key in signals)
        
        # Determine final signal
        if weighted_score >= 0.6:
            final_signal = "STRONG BUY"
            confidence = min(95, int(80 + weighted_score * 15))
        elif weighted_score >= 0.3:
            final_signal = "BUY"
            confidence = min(85, int(70 + weighted_score * 15))
        elif weighted_score <= -0.6:
            final_signal = "STRONG SELL"
            confidence = min(95, int(80 + abs(weighted_score) * 15))
        elif weighted_score <= -0.3:
            final_signal = "SELL"
            confidence = min(85, int(70 + abs(weighted_score) * 15))
        else:
            final_signal = "HOLD"
            confidence = int(60 + abs(weighted_score) * 10)
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'score': weighted_score,
            'individual_signals': signals,
            'explanations': explanations
        }
    
    def get_price_targets(self, data, signal_result):
        """Calculate price targets based on technical analysis"""
        current_price = data['Close'].iloc[-1]
        
        # Support and resistance levels
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        # Bollinger bands for targets
        bb_upper = data['BB_upper'].iloc[-1]
        bb_lower = data['BB_lower'].iloc[-1]
        
        if signal_result['signal'] in ['BUY', 'STRONG BUY']:
            target_1 = current_price * 1.05  # 5% gain
            target_2 = min(recent_high, bb_upper)  # Resistance level
            stop_loss = max(recent_low, bb_lower, current_price * 0.95)  # 5% loss protection
        elif signal_result['signal'] in ['SELL', 'STRONG SELL']:
            target_1 = current_price * 0.95  # 5% decline
            target_2 = max(recent_low, bb_lower)  # Support level
            stop_loss = min(recent_high, bb_upper, current_price * 1.05)  # 5% gain limit
        else:
            target_1 = current_price
            target_2 = current_price
            stop_loss = current_price
        
        return {
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
            'current_price': current_price
        }
