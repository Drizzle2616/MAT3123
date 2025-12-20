import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    return {
        "train_rmse": rmse(y_train, pred_train),
        "test_rmse": rmse(y_test, pred_test),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
    }


def main():
    # 0) output dirs
    os.makedirs("results", exist_ok=True)

    # 1) load dataset (reproducible, no external file needed)
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # 2) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) define models
    results = []

    # (A) Linear Regression
    lr = LinearRegression()
    r = evaluate(lr, X_train, X_test, y_train, y_test)
    r.update({"model": "LinearRegression", "setting": "default"})
    results.append(r)

    # (B) Decision Tree with varying max_depth (controls complexity)
    for depth in [2, 4, 6, 10, None]:
        dt = DecisionTreeRegressor(random_state=42, max_depth=depth)
        r = evaluate(dt, X_train, X_test, y_train, y_test)
        r.update({"model": "DecisionTree", "setting": f"max_depth={depth}"})
        results.append(r)

    # (C) Random Forest (ensemble)
    # Keep it modest to run fast while still demonstrating effect
    rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )
    r = evaluate(rf, X_train, X_test, y_train, y_test)
    r.update({"model": "RandomForest", "setting": "n_estimators=200"})
    results.append(r)

    # 4) save metrics
    df = pd.DataFrame(results)[
        ["model", "setting", "train_rmse", "test_rmse", "train_r2", "test_r2"]
    ].sort_values(by=["model", "setting"])
    df.to_csv("results/metrics.csv", index=False)
    print(df.to_string(index=False))

    # 5) plot: train vs test RMSE to show overfitting gap
    labels = df["model"] + " (" + df["setting"] + ")"
    x = np.arange(len(df))

    plt.figure()
    plt.plot(x, df["train_rmse"], marker="o", label="Train RMSE")
    plt.plot(x, df["test_rmse"], marker="o", label="Test RMSE")
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("RMSE")
    plt.title("Model Complexity Comparison (Train vs Test RMSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/comparison_plot.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
