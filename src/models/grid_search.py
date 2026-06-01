from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    joblib.dump(best_params, MODELS_DIR / "best_params.pkl")
    print("GridSearch completed.")
    print(f"Best parameters: {best_params}")
    print("Best parameters saved to models/best_params.pkl")
if __name__ == "__main__":
    main()
