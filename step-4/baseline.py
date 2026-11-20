"""Majority-class baseline for the Nim dataset (AI Lab, Session 5).

Usage from the repository root:
    python step-4/baseline.py
    python step-4/baseline.py path/to/nim_games.csv --target first_wins

The script accepts either:
- Session 5's exhaustive position dataset with target column "loses"; or
- Step 3's simulated-games dataset with target column "first_wins".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a most-frequent-class baseline on a Nim CSV dataset."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="step-3/nim_games.csv",
        help="Path to the dataset CSV (default: step-3/nim_games.csv).",
    )
    parser.add_argument(
        "--target",
        default=None,
        help='Target column. Auto-detects "loses" or "first_wins" when omitted.',
    )
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Run the Step 3 notebook first "
            "to generate nim_games.csv, or pass the correct CSV path."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The dataset is empty.")

    target = args.target
    if target is None:
        target = next(
            (column for column in ("loses", "first_wins") if column in df.columns),
            None,
        )
    if target is None or target not in df.columns:
        raise ValueError(
            f"Target column not found. Available columns: {', '.join(df.columns)}. "
            'Pass --target with "loses" or "first_wins" (or another label column).'
        )

    data = df.loc[df[target].notna()].copy()
    if data.empty:
        raise ValueError(f"Target column '{target}' contains no non-null labels.")

    # A constant numeric placeholder is sufficient: DummyClassifier ignores X.
    # It also avoids depending on the types of unrelated dataset columns.
    X = pd.DataFrame({"_baseline_input": range(len(data))})
    y = data[target]

    class_counts = y.value_counts(dropna=False)
    if len(class_counts) < 2:
        raise ValueError(
            f"Target '{target}' has only one class; a classification baseline "
            "and per-class comparison require at least two classes."
        )
    if int(class_counts.min()) < 2:
        raise ValueError(
            "Each class needs at least two examples for a stratified train/test split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # Fit the baseline using training labels only; never inspect test labels
    # when selecting the majority class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)

    labels = sorted(y.unique(), key=lambda value: str(value))
    recalls = recall_score(
        y_test,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    print(f"Dataset: {csv_path}")
    print(f"Target: {target}")
    print(f"Rows used: {len(data)}")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print("\nClass distribution (full dataset):")
    for label, count in class_counts.items():
        print(f"  {label}: {count} ({count / len(y):.1%})")

    print(f"\nMajority class learned from training data: {baseline.classes_[baseline.class_prior_.argmax()]}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRecall by class (on test set):")
    for label, recall in zip(labels, recalls):
        support = int((y_test == label).sum())
        print(f"  {label}: recall={recall:.4f} (support={support})")


if __name__ == "__main__":
    main()
