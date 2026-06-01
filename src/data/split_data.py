from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/raw.csv")
PROCESSED_DATA_DIR = Path("data/processed")

TARGET_COLUMN = "silica_concentrate"
DROP_COLUMNS = ["date"]


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)

    columns_to_drop = [col for col in DROP_COLUMNS if col in df.columns]

    X = df.drop(columns=columns_to_drop + [TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("Data splitting completed.")
    print(f"Original data shape: {df.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")


if __name__ == "__main__":
    main()
