import warnings; warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ==========
# Settings
# ==========
TICKER = "AAPL"         # e.g., "^GSPC" for S&P 500
START  = "2015-01-01"   # data start date (YYYY-MM-DD)
COST_PER_SWITCH = 0.0005  # 5 bps transaction cost when position flips

def main():
    print(f"Downloading {TICKER} from {START} ...")
    df = yf.download(TICKER, start=START)
    if df is None or df.empty:
        raise SystemExit("No data returned. Check TICKER/START or your internet connection.")

    # --- Normalize columns so we can reliably access price/volume ---
    # yfinance may return MultiIndex columns: (ticker, field) or (field, ticker).
    if isinstance(df.columns, pd.MultiIndex):
        # Try selecting the subframe for our ticker regardless of level order.
        try:
            df = df.xs(TICKER, axis=1, level=0, drop_level=True)
        except (KeyError, TypeError):
            try:
                df = df.xs(TICKER, axis=1, level=1, drop_level=True)
            except (KeyError, TypeError):
                # As a last resort, collapse to any level that contains OHLCV names
                levels = [set(map(str, lev)) for lev in df.columns.levels]
                names = {"Open","High","Low","Close","Adj Close","Volume"}
                chosen_level = None
                for i,lev in enumerate(levels):
                    if len(names & lev) > 0:
                        chosen_level = i
                        break
                if chosen_level is not None:
                    df = df.copy()
                    df.columns = df.columns.get_level_values(chosen_level)
                else:
                    raise SystemExit(f"Unexpected MultiIndex columns: {df.columns}")

    # Ensure simple string column names and strip whitespace
    df.columns = [str(c).strip() for c in df.columns]

    # Pick a price column that exists; prefer adjusted close
    price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
    if price_col is None:
        raise SystemExit(f"No 'Adj Close' or 'Close' column in downloaded data. Columns: {list(df.columns)}")

    # Ensure Volume exists (some symbols may not have it)
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna()

    # ---------
    # Features (no leakage) & target
    # ---------
    price = df[price_col]
    df["ret"] = price.pct_change()              # daily return
    df["vol_chg"] = df["Volume"].pct_change()   # volume % change
    df["sma5"] = price.rolling(5).mean()
    df["sma20"] = price.rolling(20).mean()
    df["sma_ratio"] = df["sma5"]/df["sma20"] - 1
    df["roll_vol_10"] = df["ret"].rolling(10).std()

    # Target: tomorrow up? (use future only for the label)
    df["y"] = (df["ret"].shift(-1) > 0).astype(int)

    # Drop warmup NaNs
    df = df.dropna().copy()

    FEATURES = ["ret","vol_chg","sma_ratio","roll_vol_10"]
    X = df[FEATURES].values
    y = df["y"].values
    dates = df.index

    # ---------
    # Time-aware validation (walk-forward)
    # ---------
    tscv = TimeSeriesSplit(n_splits=3)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    preds, trues, test_ix = [], [], []
    for tr, te in tscv.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        pipe.fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]      # prob of "Up"
        yhat = (p >= 0.5).astype(int)          # default threshold 0.5
        preds.append(yhat)
        trues.append(yte)
        test_ix.append(dates[te])

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    test_ix = np.concatenate(test_ix)

    print("\n=== Classification (walk-forward) ===")
    print(classification_report(trues, preds, digits=3))

    # ---------
    # Tiny backtest (long/flat, enter next day, cost per switch)
    # ---------
    test_df = df.loc[test_ix].copy()
    test_df["pred"] = preds
    test_df["position"] = test_df["pred"].shift(1).fillna(0)  # enter next day

    switches = test_df["position"].diff().abs().fillna(0)
    test_df["strategy_ret"] = test_df["position"]*test_df["ret"] - switches*COST_PER_SWITCH

    eq = (1 + test_df["strategy_ret"]).cumprod()
    bh = (1 + test_df["ret"]).cumprod()

    print(f"\nFinal equity (strategy): {eq.iloc[-1]:.3f}")
    print(f"Final equity (buy&hold): {bh.iloc[-1]:.3f}")

    # ---------
    # Save plot + summary
    # ---------
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(eq.index, eq.values, label="Strategy")
    plt.plot(bh.index, bh.values, label="Buy & Hold")
    plt.title(f"Equity Curve — {TICKER}")
    plt.xlabel("Date"); plt.ylabel("Equity"); plt.legend()
    plt.tight_layout()
    plt.savefig("figures/equity_curve.png", dpi=150)
    plt.close()

    avg_acc = float((trues == preds).mean())
    pd.DataFrame({
        "ticker":[TICKER],
        "start":[str(df.index.min().date())],
        "end":[str(df.index.max().date())],
        "avg_accuracy":[avg_acc],
        "final_equity_strategy":[float(eq.iloc[-1])],
        "final_equity_buyhold":[float(bh.iloc[-1])],
        "cost_per_switch":[COST_PER_SWITCH],
        "price_col":[price_col],
    }).to_csv("results/summary.csv", index=False)

    print("\nSaved:")
    print(" - figures/equity_curve.png")
    print(" - results/summary.csv")
    print("\nDone.")

if __name__ == "__main__":
    np.random.seed(42)
    main()

