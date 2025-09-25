from src.data.data_fetcher import fetch_intraday_nse
from src.strategy.ma_signal import moving_average_signal

if __name__ == "__main__":
    symbol = 'RELIANCE'
    df = fetch_intraday_nse(symbol)
    signals = moving_average_signal(df)
    print("Buy signals:")
    print(signals[signals['position'] == 1])
    print("Sell signals:")
    print(signals[signals['position'] == -1])
