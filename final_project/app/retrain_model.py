from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


PART4_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PART4_ROOT
    / "model"
    / "brfss_lifestyle_risk_training.csv"
)
DEFAULT_MODEL_PATH = (
    PART4_ROOT
    / "model"
    / "chronic_disease_risk_model.joblib"
)
TARGET_COLUMN = "high_risk"
FEATURE_COLUMNS = [
    "age",
    "tobacco_user",
    "obese",
    "physical_inactivity",
    "binge_drinking",
    "heavy_drinking",
    "diabetes",
    "general_health",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrain the lifestyle chronic disease risk model from CSV data."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-depth", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data_path.resolve()
    model_output = args.model_output.resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Dataset must include target column: {TARGET_COLUMN}")

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing feature columns: {', '.join(missing_columns)}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    model = DecisionTreeClassifier(max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)

    print("CDC BRFSS Lifestyle Risk Model Retraining Complete")
    print("--------------------------------------------------")
    print(f"Training rows: {len(df)}")
    print(f"Feature columns: {', '.join(X.columns)}")
    print(f"Accuracy: {accuracy:.2f}")
    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print()
    print("Decision Rules:")
    print(export_text(model, feature_names=list(X.columns)))
    print()
    print(f"Saved retrained model: {model_output}")


if __name__ == "__main__":
    main()
