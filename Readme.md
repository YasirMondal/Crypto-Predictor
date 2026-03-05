📈 Crypto Return Forecasting

A Flask-based machine learning web application that predicts short-term cryptocurrency price movements using an ensemble of regression models. The system performs complete ETL, feature engineering, multi-horizon forecasting (1h–24h), and visualizes projected price paths with profit/loss insights.

🚀 Features

Multi-horizon prediction (1h, 3h, 6h, 12h, 24h)

Ensemble model:

Linear Regression

Random Forest

Gradient Boosting

Time-series feature engineering

Profit / Loss classification

Projected price path visualization

Highest & Lowest projected values

Clean dark fintech UI

🧠 How It Works
1️⃣ Data Processing (ETL)

Reads hourly OHLC crypto CSV data

Cleans invalid values

Creates:

Lag returns (1–24 hours)

Rolling mean & volatility features

Generates multi-output targets for future returns

2️⃣ Model Training

Each coin trains one model package containing:

Linear Regression

Random Forest

Gradient Boosting

Prediction = average of all three models (ensemble).

Models are saved individually in /models.

📂 Project Structure
Crypto-Predictor/
│
├── data/               # CSV datasets (10 coins)
├── models/             # Trained model files (.pkl)
├── static/
│   ├── style.css
│   ├── script.js
│   └── plot.png
├── templates/
│   └── index.html
├── etl.py
├── train.py
├── app.py
└── README.md
⚙️ Installation
1. Clone the repository
git clone https://github.com/your-username/crypto-return-forecasting.git
cd crypto-return-forecasting
2. Create virtual environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install dependencies
pip install flask pandas numpy scikit-learn matplotlib joblib
🏋️ Train Models

Train all coins:

python train.py

This will:

Train all datasets

Print MAE & R²

Save models inside /models

🌐 Run the Application
python app.py

Open in browser:

http://127.0.0.1:5000/
📊 Output

For selected coin and time horizon:

Profit or Loss classification

Expected percentage change

Predicted future price

Highest projected value

Lowest projected value

Simulated projected price path

🔬 Technical Notes

Uses MultiOutputRegressor for simultaneous multi-horizon prediction

Time-series split prevents data leakage

Ensemble averaging stabilizes predictions

Visualization path is simulated (baseline + controlled stochastic noise)

⚠️ Limitations

No real-time API integration

No macroeconomic or sentiment features

Short-term crypto returns are highly noisy (low R² is realistic)

📌 Future Improvements

LSTM / GRU models

Hyperparameter tuning

Live data integration

Interactive Plotly/Chart.js visualization

Docker deployment

🎓 Academic Purpose

This project demonstrates:

End-to-end ML pipeline

Feature engineering for time series

Ensemble regression

Model deployment with Flask

Financial forecasting visualization