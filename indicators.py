from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

def add_indicators(df):
    close = df['Close'].squeeze()

    df['SMA'] = SMAIndicator(close, 14).sma_indicator()
    df['EMA'] = EMAIndicator(close, 14).ema_indicator()
    df['RSI'] = RSIIndicator(close, 14).rsi()

    bb = BollingerBands(close)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()


    df.dropna(inplace=True)
    return df
