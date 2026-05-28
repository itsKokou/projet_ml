"""Metriques communes pour classification et clustering."""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Retourne les metriques principales pour la fraude."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def clustering_metrics(X, labels) -> Dict[str, float]:
    """Retourne des metriques de base pour le clustering."""
    if len(set(labels)) <= 1:
        return {"silhouette": -1.0}
    return {"silhouette": float(silhouette_score(X, labels))}
