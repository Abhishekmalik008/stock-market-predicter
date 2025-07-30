import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Tuple, Optional

class AdvancedIndicators:
    """Class for calculating advanced technical indicators"""
    
    @staticmethod
    def ichimoku_cloud(df: pd.DataFrame, 
                      conversion_period: int = 9, 
                      base_period: int = 26,
                      leading_span_b_period: int = 52,
                      displacement: int = 26) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators
        
        Args:
            df: DataFrame with OHLCV data
            conversion_period: Conversion line period (default: 9)
            base_period: Base line period (default: 26)
            leading_span_b_period: Leading Span B period (default: 52)
            displacement: Cloud displacement (default: 26)
            
        Returns:
            DataFrame with Ichimoku Cloud indicators
        """
        high = df['High']
        low = df['Low']
        
        # Conversion Line (Tenkan-sen)
        conversion_line = (high.rolling(window=conversion_period).max() + 
                         low.rolling(window=conversion_period).min()) / 2
        
        # Base Line (Kijun-sen)
        base_line = (high.rolling(window=base_period).max() + 
                    low.rolling(window=base_period).min()) / 2
        
        # Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(displacement)
        
        # Leading Span B (Senkou Span B)
        leading_span_b = ((high.rolling(window=leading_span_b_period).max() + 
                          low.rolling(window=leading_span_b_period).min()) / 2).shift(displacement)
        
        # Add to DataFrame
        df['ICH_Conversion'] = conversion_line
        df['ICH_Base'] = base_line
        df['ICH_Span_A'] = leading_span_a
        df['ICH_Span_B'] = leading_span_b
        
        return df
    
    @staticmethod
    def keltner_channels(df: pd.DataFrame, 
                        period: int = 20, 
                        atr_period: int = 10,
                        multiplier: float = 2.0) -> pd.DataFrame:
        """
        Calculate Keltner Channels
        
        Args:
            df: DataFrame with OHLCV data
            period: EMA period (default: 20)
            atr_period: ATR period (default: 10)
            multiplier: ATR multiplier (default: 2.0)
            
        Returns:
            DataFrame with Keltner Channels
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        ema = typical_price.ewm(span=period, adjust=False).mean()
        
        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean()
        
        # Calculate channels
        upper_channel = ema + (multiplier * atr)
        lower_channel = ema - (multiplier * atr)
        
        # Add to DataFrame
        df[f'KC_UPPER_{period}'] = upper_channel
        df[f'KC_MIDDLE_{period}'] = ema
        df[f'KC_LOWER_{period}'] = lower_channel
        df[f'KC_WIDTH_{period}'] = (upper_channel - lower_channel) / ema
        
        return df
    
    @staticmethod
    def volume_profile(df: pd.DataFrame, 
                      price_bins: int = 20,
                      lookback: int = 252) -> pd.DataFrame:
        """
        Calculate Volume Profile
        
        Args:
            df: DataFrame with OHLCV data
            price_bins: Number of price bins (default: 20)
            lookback: Lookback period in days (default: 252)
            
        Returns:
            DataFrame with Volume Profile
        """
        df = df.copy()
        
        # Calculate price range for lookback period
        rolling_high = df['High'].rolling(window=lookback).max()
        rolling_low = df['Low'].rolling(window=lookback).min()
        
        # Create price bins
        df['Price_Bin'] = pd.cut(
            df['Close'],
            bins=price_bins,
            labels=range(price_bins)
        )
        
        # Calculate volume profile
        volume_profile = df.groupby('Price_Bin')['Volume'].sum()
        
        # Normalize volume profile
        total_volume = volume_profile.sum()
        if total_volume > 0:
            volume_profile = volume_profile / total_volume
        
        # Add to DataFrame
        for i in range(price_bins):
            if i in volume_profile.index:
                df[f'VP_{i}'] = volume_profile[i]
            else:
                df[f'VP_{i}'] = 0
        
        return df.drop('Price_Bin', axis=1, errors='ignore')
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all advanced indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all indicators
        """
        df = df.copy()
        
        # Add basic technical indicators if not present
        if 'RSI' not in df.columns:
            df['RSI'] = ta.rsi(df['Close'])
        if 'MACD' not in df.columns:
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
        
        # Add advanced indicators
        df = AdvancedIndicators.ichimoku_cloud(df)
        df = AdvancedIndicators.keltner_channels(df)
        df = AdvancedIndicators.volume_profile(df)
        
        return df
