from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

FEATURES = [
    "age",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "health",
    "absences",
    "goout",
    "freetime",
    "traveltime",
    "higher",
]

TARGET = "G3"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Place student-mat.csv inside the data folder."
        )

    # Load the student mathematics dataset.
    df = pd.read_csv(DATA_PATH, sep=";")

    # Check whether all required columns exist.
    missing = [
        column for column in FEATURES + [TARGET]
        if column not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    # Inputs and target grade.
    X = df[FEATURES]
    y = df[TARGET]

    # Identify text and numerical columns.
    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical = [
        column for column in FEATURES
        if column not in categorical
    ]

    # Convert text categories into numerical features.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numerical", "passthrough", numerical),
        ]
    )

    # Create the machine-learning pipeline.
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                ),
            ),
        ]
    )

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Train the model and predict unseen test records.
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Measure the prediction errors.
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    # Compare the model with a mean-value baseline.
    baseline_predictions = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    print("\nSTUDENT PERFORMANCE PREDICTION")
    print("-" * 35)
    print(f"Total records: {len(df)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print(f"Baseline MAE: {baseline_mae:.2f} grade points")
    print(f"Model MAE:    {mae:.2f} grade points")
    print(f"Model RMSE:   {rmse:.2f} grade points")
    print(f"Model R-squared: {r2:.3f}")


if __name__ == "__main__":
    main()
