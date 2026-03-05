import os
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from etl import prepare_training_data, HORIZONS


DATA_FOLDER = "data"
MODEL_FOLDER = "models"
MAX_ROWS = 3000  

os.makedirs(MODEL_FOLDER, exist_ok=True)


def time_series_split(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{name} → MAE: {mae:.4f} | R2: {r2:.4f}")


def train_coin(coin_file):
    print("\n==============================")
    print(f"Training for {coin_file}")
    print("==============================")

    X, y = prepare_training_data(coin_file)

    
    if len(X) > MAX_ROWS:
        X = X.iloc[-MAX_ROWS:]
        y = y.iloc[-MAX_ROWS:]
        print(f"Using last {MAX_ROWS} rows")

    if len(X) < 200:
        print("Not enough data. Skipping.")
        return

    X_train, X_test, y_train, y_test = time_series_split(X, y)

    
    linear_model = MultiOutputRegressor(LinearRegression())
    linear_model.fit(X_train, y_train)
    evaluate_model("Linear", linear_model, X_test, y_test)

  
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )
    )
    rf_model.fit(X_train, y_train)
    evaluate_model("Random Forest", rf_model, X_test, y_test)

    
    gbr_model = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=50,
            random_state=42
        )
    )
    gbr_model.fit(X_train, y_train)
    evaluate_model("Gradient Boosting", gbr_model, X_test, y_test)

    
    model_package = {
        "linear": linear_model,
        "rf": rf_model,
        "gbr": gbr_model,
        "horizons": HORIZONS
    }

    model_name = coin_file.replace(".csv", ".pkl")
    joblib.dump(model_package, f"{MODEL_FOLDER}/{model_name}")

    print(f"Saved model → models/{model_name}")


def main():
    coin_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

    if not coin_files:
        print("No CSV files found.")
        return

    for coin_file in coin_files:
        train_coin(coin_file)

    print("\nAll coins trained successfully.")


if __name__ == "__main__":
    main()