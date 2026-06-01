from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()

    best_params = joblib.load(MODELS_DIR / "best_params.pkl")

    model = RandomForestRegressor(
        random_state=42,
        **best_params,
    )

    model.fit(X_train, y_train)

    joblib.dump(model, MODELS_DIR / "trained_model.pkl")

    print("Model training completed.")
    print("Trained model saved to models/trained_model.pkl")


if __name__ == "__main__":
    main()
