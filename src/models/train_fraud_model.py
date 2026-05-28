"""Entrainement et comparaison de modeles pour la fraude."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_fraud_data
from src.data.preprocessing import preprocess_fraud
from src.features.build_features import build_fraud_features
from src.models.evaluate import classification_metrics

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def _best_threshold_from_validation(y_true, y_proba, min_recall: float = 0.75) -> float:
    """Selectionne un seuil sur validation en max F1 sous contrainte de rappel minimal."""
    best_threshold = 0.5
    best_f1 = -1.0

    for i in range(5, 100):
        threshold = i / 100
        y_pred = (y_proba >= threshold).astype(int)
        recall = ((y_pred & y_true).sum() / max(y_true.sum(), 1))
        if recall < min_recall:
            continue
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold


def _prepare_fraud_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Charge, nettoie et prepare X/y pour la fraude."""
    raw_df = load_fraud_data()
    df = build_fraud_features(preprocess_fraud(raw_df))

    target_col = "isFraud"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Colonnes a exclure pour une baseline stable et exploitable.
    cols_to_exclude = [
        col
        for col in ("nameOrig", "nameDest", "isFlaggedFraud")
        if col in X.columns
    ]
    if cols_to_exclude:
        X = X.drop(columns=cols_to_exclude)

    return X, y


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit un preprocessor numerique + categoriel reutilisable."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _stratified_sample(
    X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Sous-echantillonne de facon stratifiee pour limiter le cout de calcul."""
    if len(X) <= max_rows:
        return X, y

    _, X_small, _, y_small = train_test_split(
        X,
        y,
        test_size=max_rows,
        random_state=random_state,
        stratify=y,
    )
    return X_small, y_small


@dataclass
class FraudRunConfig:
    name: str
    estimator_builder: Callable[[pd.Series], object]
    max_train_rows: int | None = None


def train_and_compare_fraud_models(
    models_dir: Path = Path("models/fraud"),
    summary_json_path: Path = Path("models/fraud/fraud_model_comparison.json"),
    summary_csv_path: Path = Path("models/fraud/fraud_model_comparison.csv"),
    min_recall: float = 0.75,
) -> Dict[str, Dict[str, float]]:
    """Entraine plusieurs modeles, calibre leur seuil et compare les metriques."""
    X, y = _prepare_fraud_dataset()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1765,  # ~15% du total en validation
        random_state=42,
        stratify=y_train_full,
    )

    pos = max(int((y_train == 1).sum()), 1)
    neg = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = neg / pos

    configs = [
        FraudRunConfig(
            name="logistic_regression",
            estimator_builder=lambda _: LogisticRegression(max_iter=1000, class_weight="balanced"),
            max_train_rows=None,
        ),
        FraudRunConfig(
            name="random_forest",
            estimator_builder=lambda _: RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            max_train_rows=350_000,
        ),
    ]

    if HAS_XGBOOST:
        configs.append(
            FraudRunConfig(
                name="xgboost",
                estimator_builder=lambda y_local: XGBClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=max(
                        int((y_local == 0).sum()) / max(int((y_local == 1).sum()), 1), 1.0
                    ),
                ),
                max_train_rows=450_000,
            )
        )

    all_metrics: Dict[str, Dict[str, float]] = {}
    best_model_name = None
    best_model_score = -1.0
    best_model_pipeline = None

    for config in configs:
        X_train_model, y_train_model = X_train, y_train
        if config.max_train_rows is not None:
            X_train_model, y_train_model = _stratified_sample(
                X_train, y_train, max_rows=config.max_train_rows, random_state=42
            )

        preprocessor = _build_preprocessor(X_train_model)
        estimator = config.estimator_builder(y_train_model)
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        model.fit(X_train_model, y_train_model)
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = _best_threshold_from_validation(y_val.to_numpy(), val_proba, min_recall=min_recall)

        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        metrics = classification_metrics(y_test, test_pred, test_proba)
        metrics["threshold"] = float(threshold)
        metrics["train_size"] = int(len(X_train_model))
        metrics["val_size"] = int(len(X_val))
        metrics["test_size"] = int(len(X_test))
        metrics["positive_rate_test"] = float(y_test.mean())
        metrics["min_recall_constraint"] = float(min_recall)
        all_metrics[config.name] = metrics

        # Critere de selection : meilleur PR-AUC
        if metrics["pr_auc"] > best_model_score:
            best_model_score = metrics["pr_auc"]
            best_model_name = config.name
            best_model_pipeline = model

    models_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    pd.DataFrame(all_metrics).T.to_csv(summary_csv_path, index=True)

    if best_model_pipeline is not None and best_model_name is not None:
        joblib.dump(best_model_pipeline, models_dir / "fraud_model.joblib")
        joblib.dump(best_model_pipeline, models_dir / f"fraud_model_{best_model_name}.joblib")
        best_info = {
            "best_model": best_model_name,
            "selection_metric": "pr_auc",
            "best_pr_auc": best_model_score,
        }
        (models_dir / "fraud_best_model.json").write_text(
            json.dumps(best_info, indent=2), encoding="utf-8"
        )

    return all_metrics


if __name__ == "__main__":
    results = train_and_compare_fraud_models()
    print("Fraud model comparison:", results)
