import pandas as pd
import numpy as np

HORIZONS = [1, 3, 6, 12, 24]


def load_data(coin_file):
    df = pd.read_csv(f"data/{coin_file}")

    
    df.columns = df.columns.str.lower().str.strip()

    required_cols = ["timestamp", "open", "high", "low", "close"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")
    df = df.drop_duplicates()

    df = df[df["close"] > 0]

    df = df.dropna()

    return df


def create_targets(df):
    for h in HORIZONS:
        df[f"target_{h}h"] = (
            df["close"].shift(-h) - df["close"]
        ) / df["close"] * 100

    return df


def create_features(df):
    df["return_1"] = df["close"].pct_change()

    for lag in range(1, 25):
        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)

    df["rolling_mean_6"] = df["close"].rolling(6).mean()
    df["rolling_std_6"] = df["close"].rolling(6).std()

    df["rolling_mean_24"] = df["close"].rolling(24).mean()
    df["rolling_std_24"] = df["close"].rolling(24).std()

    return df


def clean_dataframe(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def get_feature_columns(df):
    return [
        col for col in df.columns
        if col.startswith("return_lag_") or col.startswith("rolling_")
    ]


def prepare_training_data(coin_file):
    df = load_data(coin_file)
    df = create_targets(df)
    df = create_features(df)
    df = clean_dataframe(df)

    feature_cols = get_feature_columns(df)
    target_cols = [f"target_{h}h" for h in HORIZONS]

    X = df[feature_cols]
    y = df[target_cols]

    return X, y


def prepare_latest_features(coin_file):
    df = load_data(coin_file)
    df = create_features(df)
    df = clean_dataframe(df)

    feature_cols = get_feature_columns(df)

    # 🔥 RETURN DATAFRAME, NOT NUMPY ARRAY
    X_latest = df[feature_cols].iloc[[-1]]   # double brackets → keeps DataFrame
    current_price = df["close"].iloc[-1]

    return X_latest, current_price
