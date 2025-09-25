def moving_average_signal(df, short_window=5, long_window=20):
    df = df.copy()
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[short_window:, 'signal'] = (df['SMA_short'][short_window:] > df['SMA_long'][short_window:]).astype(int)
    df['position'] = df['signal'].diff()
    return df[['datetime', 'close', 'SMA_short', 'SMA_long', 'signal', 'position']]
