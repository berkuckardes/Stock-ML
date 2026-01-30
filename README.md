📈 Machine Learning Trading Strategy — AAPL

This project implements a simple machine learning–based trading strategy and compares its performance against a Buy & Hold benchmark using historical Apple (AAPL) stock data.

The goal is not to beat the market aggressively, but to demonstrate:

clean time-series modeling

leakage-free feature engineering

walk-forward validation

realistic transaction costs

interpretable performance evaluation

🧠 Overview

Asset: Apple Inc. (AAPL)

Period: From 2015 onward

Model: Logistic Regression

Strategy: Long / Flat (no short selling)

Benchmark: Buy & Hold

Validation: Walk-forward (TimeSeriesSplit)

The strategy predicts whether tomorrow’s return will be positive, and takes a position accordingly.

📊 Results

Below is the equity curve comparing the strategy against Buy & Hold:

Blue: ML Strategy

Orange: Buy & Hold

The plot shows how a simple ML model performs under realistic constraints, including transaction costs.

🏗️ Project Structure
.
├── run_ultralight.py        # Main experiment script
├── figures/
│   └── equity_curve.png    # Equity curve plot
├── results/
│   └── summary.csv         # Performance summary
└── README.md

🔧 Features & Target
Features (no future leakage)

Daily returns

Volume percentage change

SMA(5) / SMA(20) ratio

Rolling volatility (10-day)

Target
1 → Tomorrow’s return is positive
0 → Tomorrow’s return is negative or zero

🧪 Validation Methodology

TimeSeriesSplit (walk-forward)

No random shuffling

Model is always trained on past data only

Predictions are executed one day later to avoid look-ahead bias

💸 Backtesting Assumptions

Position: Long (1) or Flat (0)

Entry: Next trading day

Transaction cost: 5 bps per position switch

Compounding: Enabled

📦 Dependencies
pip install numpy pandas matplotlib yfinance scikit-learn

▶️ How to Run
python run_ultralight.py


Outputs:

figures/equity_curve.png

results/summary.csv

📄 Output Summary

The script reports:

Classification accuracy (walk-forward)

Final equity (strategy vs buy & hold)

Average prediction accuracy

Used price column (Adj Close / Close)

⚠️ Disclaimer

This project is for educational and research purposes only.
It does not constitute financial advice or an investment recommendation.

👤 Author

Berk Uçkardeş
GitHub: @berkuckardes
