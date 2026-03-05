from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from etl import prepare_latest_features

app = Flask(__name__)

MODEL_FOLDER = "models"


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    predicted_price = None
    outcome = None
    highest_value = None
    lowest_value = None
    selected_coin = None
    selected_horizon = None
    plot_path = None

    coins = [f.replace(".pkl", "") for f in os.listdir(MODEL_FOLDER) if f.endswith(".pkl")]
    horizons = [1, 3, 6, 12, 24]

    if request.method == "POST":
        selected_coin = request.form["coin"]
        selected_horizon = int(request.form["horizon"])

        model_package = joblib.load(f"{MODEL_FOLDER}/{selected_coin}.pkl")

        linear_model = model_package["linear"]
        rf_model = model_package["rf"]
        gbr_model = model_package["gbr"]
        horizon_list = model_package["horizons"]

        X_latest, current_price = prepare_latest_features(f"{selected_coin}.csv")

        linear_pred = linear_model.predict(X_latest)[0]
        rf_pred = rf_model.predict(X_latest)[0]
        gbr_pred = gbr_model.predict(X_latest)[0]

        horizon_index = horizon_list.index(selected_horizon)

        prediction = np.mean([
            linear_pred[horizon_index],
            rf_pred[horizon_index],
            gbr_pred[horizon_index]
        ])

        prediction = round(float(prediction), 4)
        predicted_price = float(current_price * (1 + prediction / 100))

        outcome = "PROFIT" if prediction > 0 else "LOSS"

        steps = selected_horizon
        x = np.arange(0, steps + 1)

        baseline = np.linspace(current_price, predicted_price, steps + 1)

        volatility = current_price * 0.02
        noise = np.random.normal(0, volatility / 4, steps + 1)

        noise[0] = 0
        noise[-1] = 0

        price_path = baseline + noise

        highest_value = round(float(np.max(price_path)), 4)
        lowest_value = round(float(np.min(price_path)), 4)
        predicted_price = round(float(predicted_price), 4)

        plt.figure(figsize=(6, 4))
        plt.plot(x, price_path)

        plt.xlabel("Time (Hours)")
        plt.ylabel("Price")

        if prediction > 0:
            plt.title("Projected Price Path (Bullish Bias)")
        else:
            plt.title("Projected Price Path (Bearish Bias)")

        plt.tight_layout()

        plot_path = "static/plot.png"
        plt.savefig(plot_path)
        plt.close()

    return render_template(
        "index.html",
        coins=coins,
        horizons=horizons,
        prediction=prediction,
        predicted_price=predicted_price,
        outcome=outcome,
        highest_value=highest_value,
        lowest_value=lowest_value,
        selected_coin=selected_coin,
        selected_horizon=selected_horizon,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    app.run(debug=True)