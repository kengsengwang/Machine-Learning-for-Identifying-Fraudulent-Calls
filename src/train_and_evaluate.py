# src/train_and_evaluate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import get_logger, ensure_dir

# --------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------
DATA_CSV = Path("data") / "processed_calls.csv"
ARTIFACT_DIR = Path("artifacts")
TARGET_COL = "scam_label"


# --------------------------------------------------------------------
# Data loading & feature splitting
# --------------------------------------------------------------------
def load_processed(csv_path: Path = DATA_CSV) -> pd.DataFrame:
    """Load the processed CSV created in the previous step."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset {csv_path} not found.\n"
            "Run 'python -m src.build_features' first."
        )
    return pd.read_csv(csv_path)


def split_features(
    df: pd.DataFrame, target_col: str = TARGET_COL
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Split the dataset into X, y and identify numeric / categorical columns."""
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found in processed dataset."
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return X, y, numeric_features, categorical_features


# --------------------------------------------------------------------
# Preprocessing & models
# --------------------------------------------------------------------
def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline for numeric and categorical variables.

    - Numeric: impute missing values with median, then scale.
    - Categorical: impute missing values with most_frequent, then one-hot encode.
    """

    # Numeric pipeline: impute -> scale
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute -> one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,  # modern param; returns dense matrix
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_models(random_state: int = 42) -> Dict[str, object]:
    """
    Define at least three models to evaluate.
    All from scikit-learn so there is no heavy dependency.
    """
    models: Dict[str, object] = {
        "log_reg": LogisticRegression(
            max_iter=1000,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "grad_boost": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }
    return models


# --------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------
def train_and_evaluate(
    csv_path: Path = DATA_CSV, out_dir: Path = ARTIFACT_DIR
) -> None:
    logger = get_logger("train_and_evaluate")

    df = load_processed(csv_path)
    X, y, num_cols, cat_cols = split_features(df)

    logger.info(
        f"Dataset: {X.shape[0]} rows, {X.shape[1]} features "
        f"({len(num_cols)} numeric, {len(cat_cols)} categorical)."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    models = get_models()

    out_dir = ensure_dir(out_dir)
    summary_lines: List[str] = []

    for name, clf in models.items():
        logger.info(f"Training model: {name}")

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", clf),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # try to compute ROC-AUC if probabilities are available
        try:
            y_prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            y_prob = None
            auc = float("nan")

        report = classification_report(
            y_test,
            y_pred,
            digits=4,
        )

        logger.info(f"Results for {name}:\n{report}")
        if y_prob is not None:
            logger.info(f"ROC AUC: {auc:.4f}")

        # save per-model report
        report_path = out_dir / f"{name}_report.txt"
        with report_path.open("w", encoding="utf-8") as f:
            f.write(f"Model: {name}\n\n")
            f.write(report)
            if y_prob is not None:
                f.write(f"\nROC AUC: {auc:.4f}\n")

        summary_lines.append(f"{name}\tROC_AUC={auc:.4f}")

    # save a small summary file
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Model\tMetric\n")
        for line in summary_lines:
            f.write(line + "\n")

    logger.info(f"Evaluation complete. Reports saved under {out_dir}")


def main() -> None:
    train_and_evaluate()


if __name__ == "__main__":
    main()
