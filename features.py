import pandas as pd

def add_features(df):
    df = df.sort_values("Date").copy()

    # Moving average in 5/10/20 days
    df["ma_5"] = df["Close"].rolling(5).mean()
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()

    # Daily return
    df["daily_return"] = df["Close"].pct_change()

    # Volatility in 10 days
    df["volatility_10"] = df["daily_return"].rolling(10).std()

    # Range position in 5 days -> capture momentum
    high_5 = df["High"].rolling(5).max()
    low_5 = df["Low"].rolling(5).min()
    df["range_position"] = (df["Close"] - low_5) / (high_5 - low_5)

    # Volume change in 5 days (compare to the average in 5 days)
    df["volume_change"] = df["Volume"] / (df["Volume"].rolling(5).mean())

    # Momentum (Compares to 10 days before)
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    #RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Target - highest/lowest price over the next 5 trading days
    df["target_high"] = df["High"].rolling(5).max().shift(-4)
    df["target_low"] = df["Low"].rolling(5).min().shift(-4)

    feature_cols = ["ma_5", "ma_10", "ma_20", "daily_return", "volatility_10",
                    "range_position", "volume_change", "momentum_10", "rsi_14"]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)


    return df

if __name__ == "__main__":
    from data_loader import load_from_db
    df = load_from_db("AAPL")
    df = add_features(df)
    print(df[["Date", "Close", "ma_5", "ma_10", "ma_20", "daily_return", "volatility_10" , "range_position",
              "momentum_10", "rsi_14", "target_high", "target_low"]].tail())
    print("num len: ", len(df))