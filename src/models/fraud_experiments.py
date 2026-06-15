"""Experiences complementaires pour la fraude : split temporel et analyse d'erreurs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.models.evaluate import classification_metrics
from src.models.train_fraud_model import (
    HAS_XGBOOST,
    _best_threshold_from_validation,
    _build_preprocessor,
    _prepare_fraud_dataset,
    _stratified_sample,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # type: ignore


DEFAULT_MAX_TRAIN_ROWS = None  # None = utilise tout le jeu d'entraînement (~734k lignes)
MODELS_SCOPE = [
    {
        "model": "SMOTE (imbalanced-learn)",
        "status": "non_utilise",
        "raison": "Déséquilibre traité via class_weight / scale_pos_weight ; SMOTE non retenu (surdimensionnement artificiel).",
    },
]


def _temporal_masks(step: pd.Series, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Decoupe train/val/test selon la variable temporelle step."""
    train_cutoff = step.quantile(train_ratio)
    val_cutoff = step.quantile(train_ratio + val_ratio)

    train_mask = step <= train_cutoff
    val_mask = (step > train_cutoff) & (step <= val_cutoff)
    test_mask = step > val_cutoff
    return train_mask, val_mask, test_mask


def _build_xgboost_estimator(y_train: pd.Series) -> object:
    if not HAS_XGBOOST or XGBClassifier is None:
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    scale = max(int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1), 1.0)
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale,
    )


def evaluate_temporal_split(
    models_dir: Path = Path("models/fraud"),
    min_recall: float = 0.75,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
) -> Dict[str, float]:
    """Entraine le modele retenu (XGBoost ou RF) avec split temporel sur step."""
    X, y = _prepare_fraud_dataset()
    train_mask, val_mask, test_mask = _temporal_masks(X["step"])

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    X_train, y_train = _stratified_sample(X_train, y_train, max_rows=max_train_rows, random_state=42)

    preprocessor = _build_preprocessor(X_train)
    estimator = _build_xgboost_estimator(y_train)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = _best_threshold_from_validation(y_val.to_numpy(), val_proba, min_recall=min_recall)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)
    metrics = classification_metrics(y_test, test_pred, test_proba)
    metrics.update(
        {
            "threshold": float(threshold),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "positive_rate_test": float(y_test.mean()),
            "split_strategy": "temporal_step",
            "train_step_max": float(X_train["step"].max()),
            "test_step_min": float(X_test["step"].min()),
            "min_recall_constraint": float(min_recall),
        }
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fraud_temporal_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def export_error_analysis(
    models_dir: Path = Path("models/fraud"),
    min_recall: float = 0.75,
    max_train_rows: int | None = DEFAULT_MAX_TRAIN_ROWS,
    low_threshold: float = 0.50,
    max_cases: int = 10,
) -> Dict[str, object]:
    """Exporte les faux negatifs et quasi-faux positifs sur le jeu de test aleatoire."""
    X, y = _prepare_fraud_dataset()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1765,
        random_state=42,
        stratify=y_train_full,
    )
    X_train, y_train = _stratified_sample(X_train, y_train, max_rows=max_train_rows, random_state=42)

    preprocessor = _build_preprocessor(X_train)
    estimator = _build_xgboost_estimator(y_train)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = _best_threshold_from_validation(y_val.to_numpy(), val_proba, min_recall=min_recall)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    analysis_df = X_test.copy()
    analysis_df["isFraud"] = y_test.to_numpy()
    analysis_df["proba"] = test_proba
    analysis_df["prediction"] = test_pred
    analysis_df["error_type"] = "correct"
    analysis_df.loc[(analysis_df["isFraud"] == 1) & (analysis_df["prediction"] == 0), "error_type"] = "false_negative"
    analysis_df.loc[(analysis_df["isFraud"] == 0) & (analysis_df["prediction"] == 1), "error_type"] = "false_positive"

    fn_cases = analysis_df[analysis_df["error_type"] == "false_negative"].sort_values("proba").head(max_cases)
    fp_cases = analysis_df[analysis_df["error_type"] == "false_positive"].sort_values("proba", ascending=False)

    if fp_cases.empty:
        fp_cases = (
            analysis_df[(analysis_df["isFraud"] == 0) & (analysis_df["proba"] >= low_threshold)]
            .sort_values("proba", ascending=False)
            .head(max_cases)
        )
        fp_label = f"quasi_false_positive_threshold_{low_threshold}"
    else:
        fp_label = "false_positive"

    display_cols = [
        c
        for c in [
            "step",
            "type",
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest",
            "origin_error",
            "dest_error",
            "isFraud",
            "proba",
            "prediction",
            "error_type",
        ]
        if c in analysis_df.columns
    ]

    summary = {
        "threshold": float(threshold),
        "test_size": int(len(X_test)),
        "false_negatives": int((analysis_df["error_type"] == "false_negative").sum()),
        "false_positives": int((analysis_df["error_type"] == "false_positive").sum()),
        "recall_at_threshold": float(
            analysis_df.loc[analysis_df["isFraud"] == 1, "prediction"].mean()
        ),
        "precision_at_threshold": float(
            analysis_df.loc[analysis_df["prediction"] == 1, "isFraud"].mean()
            if analysis_df["prediction"].sum() > 0
            else 0.0
        ),
        "low_threshold_for_quasi_fp": float(low_threshold),
        "fp_case_label": fp_label,
        "false_negative_cases": fn_cases[display_cols].to_dict(orient="records"),
        "false_positive_cases": fp_cases[display_cols].to_dict(orient="records"),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fraud_error_analysis.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(fn_cases[display_cols]).to_csv(models_dir / "fraud_false_negatives.csv", index=False)
    pd.DataFrame(fp_cases[display_cols]).to_csv(models_dir / "fraud_false_positives.csv", index=False)
    return summary


def export_cost_analysis(
    models_dir: Path = Path("models/fraud"),
    cost_per_fp_review: float = 25.0,
    thresholds: list[float] | None = None,
) -> Dict[str, object]:
    """Chiffre le compromis economique FP/FN pour plusieurs seuils (hypotheses documentees)."""
    import joblib

    if thresholds is None:
        thresholds = [0.50, 0.67, 0.84]

    X, y = _prepare_fraud_dataset()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    model_path = models_dir / "fraud_model.joblib"
    if not model_path.exists():
        return {}

    model = joblib.load(model_path)
    test_proba = model.predict_proba(X_test)[:, 1]
    amounts = X_test["amount"].astype(float) if "amount" in X_test.columns else pd.Series(0.0, index=X_test.index)

    def _load_json(path: Path) -> dict:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    comparison = _load_json(models_dir / "fraud_model_comparison.json")
    best_name = _load_json(models_dir / "fraud_best_model.json").get("best_model", "xgboost")
    best_threshold = float(comparison.get(best_name, {}).get("threshold", 0.67))

    scenarios = []
    for threshold in thresholds:
        pred = (test_proba >= threshold).astype(int)
        y_np = y_test.to_numpy()
        fn_mask = (y_np == 1) & (pred == 0)
        fp_mask = (y_np == 0) & (pred == 1)
        fn_count = int(fn_mask.sum())
        fp_count = int(fp_mask.sum())
        tp = int(((y_np == 1) & (pred == 1)).sum())
        fn_loss = float(amounts.to_numpy()[fn_mask].sum())
        fp_cost = float(fp_count * cost_per_fp_review)
        scenarios.append(
            {
                "threshold": float(threshold),
                "false_negatives": fn_count,
                "false_positives": fp_count,
                "recall": float(tp / max((y_test == 1).sum(), 1)),
                "precision": float(tp / max(tp + fp_count, 1)),
                "fn_financial_loss_units": fn_loss,
                "fp_operational_cost_eur": fp_cost,
                "total_cost_eur_equivalent": fn_loss + fp_cost,
                "is_selected_threshold": abs(threshold - best_threshold) < 0.001,
            }
        )

    summary = {
        "assumptions": {
            "cost_per_fp_review_eur": cost_per_fp_review,
            "fn_cost_model": "Somme des montants des transactions frauduleuses manquees (unites du dataset).",
            "note": "Les montants sont en unites du dataset ; 1 unite = 1 EUR en lecture illustrative.",
        },
        "selected_threshold": best_threshold,
        "test_frauds_total": int((y_test == 1).sum()),
        "scenarios": scenarios,
        "recommendation": (
            f"Seuil {best_threshold:.2f} retenu : meilleur equilibre observe entre pertes FN "
            f"et cout de revue FP pour la capacite operationnelle supposee."
        ),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fraud_cost_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def export_models_scope(models_dir: Path = Path("models/fraud")) -> list:
    """Documente les modeles demandes par l'enonce mais non encore implementes."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "fraud_models_scope.json").write_text(
        json.dumps(MODELS_SCOPE, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return MODELS_SCOPE


if __name__ == "__main__":
    temporal = evaluate_temporal_split()
    errors = export_error_analysis()
    scope = export_models_scope()
    print("Temporal metrics:", temporal)
    print("Error summary:", {k: v for k, v in errors.items() if k.endswith("_cases") is False})
    print("Models scope:", scope)
