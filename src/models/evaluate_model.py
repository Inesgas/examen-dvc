import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
PREDICTIONS_PATH = Path("data/predictions.csv")


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()

    model = joblib.load(MODELS_DIR / "trained_model.pkl")

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    scores = {
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

    with open(METRICS_DIR / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    print("Model evaluation completed.")
    print(f"Scores: {scores}")
    print("Predictions saved to data/predictions.csv")
    print("Scores saved to metrics/scores.json")


if __name__ == "__main__":
    main()
