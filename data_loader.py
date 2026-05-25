import yfinance as yf
import pandas as pd
import sqlite3

def fetch_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5y")

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["Symbol"] = symbol.upper()

    return df

def save_to_db(df, db_path="stock_data.db"):
    conn = sqlite3.connect(db_path)
    symbol = df["Symbol"].iloc[0]

    # Delete data of the symbol if existed in the db to prevent duplicates
    try: 
        conn.execute("DELETE FROM stock_prices " \
                    "WHERE Symbol = ?", (symbol,))
        conn.commit()
    except sqlite3.OperationalError:
        pass

    df.to_sql("stock_prices", conn, if_exists="append", index=False)
    conn.close()

    print(f"Saved {len(df)} data onto {db_path}")

def load_from_db(symbol, db_path="stock_data.db"):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM stock_prices " \
            "WHERE Symbol = ? " \
            "ORDER BY Date"
    df = pd.read_sql(query, conn, params=(symbol.upper(),))

    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])

    return df

if __name__ == "__main__":
    data = fetch_data("AAPL")
    save_to_db(data)

    loaded_df = load_from_db("AAPL")
    print("Data loaded from stock_data.db")
    print(loaded_df.tail())