import ta

def add_all_signals(df):
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_signal'] = 0
    df.loc[df['rsi'] < 30, 'rsi_signal'] = 1
    df.loc[df['rsi'] > 70, 'rsi_signal'] = -1

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['macd_signal_flag'] = 0
    df.loc[df['macd'] > df['macd_signal'], 'macd_signal_flag'] = 1
    df.loc[df['macd'] < df['macd_signal'], 'macd_signal_flag'] = -1

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_signal'] = 0
    df.loc[df['close'] < df['bb_low'], 'bb_signal'] = 1
    df.loc[df['close'] > df['bb_high'], 'bb_signal'] = -1

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch'] = stoch.stoch()
    df['stoch_signal'] = 0
    df.loc[df['stoch'] < 20, 'stoch_signal'] = 1
    df.loc[df['stoch'] > 80, 'stoch_signal'] = -1

    return df

def ensemble_signals(df, min_agree=3):
    df = add_all_signals(df)
    # Ensemble voting: sum signals, require at least min_agree agreement
    signals = df[['rsi_signal', 'macd_signal_flag', 'bb_signal', 'stoch_signal']]
    df['buy_signal'] = (signals.sum(axis=1) >= min_agree)
    df['sell_signal'] = (signals.sum(axis=1) <= -min_agree)
    return df