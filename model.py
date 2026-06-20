import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

FEATURE_COLUMNS = [
    "ma_5", "ma_10", "ma_20",
    "daily_return", "volatility_10",
    "range_position", "volume_change",
    "momentum_10", "rsi_14",
]

def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

def select_best_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    scores = {}

    for name, model in get_models().items():
        mae_list = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mae_list.append(mean_absolute_error(y_test, preds))

        scores[name] = sum(mae_list) / len(mae_list)

    best_name = min(scores, key=scores.get)
    best_model = get_models()[best_name]
    best_model.fit(X, y)

    return best_name, best_model, scores

if __name__ == "__main__":
    from data_loader import fetch_data, save_to_db, load_from_db
    from features import add_features

    data = fetch_data("MU")
    save_to_db(data)
    df = load_from_db("MU")

    df = add_features(df)

    train_df = df.dropna(subset=["target_high", "target_low"])
    X_train = train_df[FEATURE_COLUMNS]
    y_high = train_df["target_high"]
    y_low = train_df["target_low"]

    print("=== Highest price prediction ===")
    high_name, high_model, high_score = select_best_model(X_train, y_high)
    for name, score in high_score.items():
        print(f"{name}: MAE = {score:.3f}")
    print(f" --> Best model: {high_name}")

    print("=== Lowest price prediction ===")
    low_name, low_model, low_score = select_best_model(X_train, y_low)
    for name, score in low_score.items():
        print(f"{name}: MAE = {score:.3f}")
    print(f" --> Best model: {low_name}")

    latest_feature = df[FEATURE_COLUMNS].iloc[[-1]]
    pred_high = high_model.predict(latest_feature)[0]
    pred_low = low_model.predict(latest_feature)[0]
    last_close = df["Close"].iloc[-1]

    print("\n=== Next week prediction ===")
    print(f"  Latest close: {last_close:.2f}")
    print(f"  Predicted high: {pred_high:.2f}")
    print(f"  Predicted low:  {pred_low:.2f}")

    # === Graph ===
    recent = df.tail(60)
    plt.figure(figsize=(11, 6))
    plt.plot(recent["Date"], recent["Close"], label="Close price", linewidth=1.6)
    plt.axhline(pred_high, color="green", linestyle="--", label=f"Predicted high ({pred_high:.1f})")
    plt.axhline(pred_low, color="red", linestyle="--", label=f"Predicted low ({pred_low:.1f})")
    plt.title("Last 60 days & predicted next-week range")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("prediction_plot.png", dpi=120)
    print("\n  Chart saved to prediction_plot.png")
