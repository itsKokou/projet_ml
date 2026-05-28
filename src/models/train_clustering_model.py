"""Entrainement et comparaison de modeles de clustering."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_cluster_data
from src.data.preprocessing import preprocess_cluster
from src.features.build_features import build_cluster_features
from src.models.evaluate import clustering_metrics


@dataclass
class ClusterConfig:
    name: str
    estimator: object
    n_clusters: int


def _prepare_cluster_dataset() -> pd.DataFrame:
    """Charge, nettoie et prepare les donnees de segmentation."""
    raw_df = load_cluster_data()
    df = build_cluster_features(preprocess_cluster(raw_df))

    # Variables non pertinentes pour segmenter les comportements d'achat.
    drop_cols = [col for col in ["ID", "Response", "Complain", "Dt_Customer"] if col in df.columns]
    return df.drop(columns=drop_cols)


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le pipeline d'encodage et de mise a l'echelle."""
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
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def _compute_labels(estimator, X_transformed: np.ndarray) -> np.ndarray:
    """Retourne les labels de cluster pour un estimateur donne."""
    fitted = clone(estimator)
    if isinstance(fitted, GaussianMixture):
        return fitted.fit_predict(X_transformed)
    return fitted.fit_predict(X_transformed)


def _supports_predict(estimator) -> bool:
    """Indique si l'estimateur expose predict pour l'inference."""
    return callable(getattr(estimator, "predict", None))


def _score_clustering(X_transformed: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calcule un ensemble de metriques pour comparer les clusters."""
    metrics = clustering_metrics(X_transformed, labels)
    if len(np.unique(labels)) > 1:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X_transformed, labels))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X_transformed, labels))
    else:
        metrics["davies_bouldin"] = float("inf")
        metrics["calinski_harabasz"] = 0.0
    return metrics


def _profile_clusters(df_raw: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Produit un profil metier de chaque cluster."""
    prof = df_raw.copy()
    prof["cluster"] = labels
    selected_cols = [
        c
        for c in [
            "Income",
            "Age",
            "Recency",
            "Total_Spending",
            "Total_Purchases",
            "NumWebPurchases",
            "NumStorePurchases",
            "NumCatalogPurchases",
            "NumDealsPurchases",
            "Campaign_Acceptance_Total",
        ]
        if c in prof.columns
    ]
    out = prof.groupby("cluster")[selected_cols].mean().round(2)
    out["cluster_size"] = prof.groupby("cluster").size()
    return out.reset_index()


def train_and_compare_clustering_models(
    models_dir: Path = Path("models/clustering"),
) -> dict:
    """Compare KMeans, Agglomerative et GMM, puis sauvegarde le meilleur."""
    X = _prepare_cluster_dataset()
    preprocessor = _build_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    configs = []
    for k in [3, 4, 5, 6]:
        configs.extend(
            [
                ClusterConfig(
                    name="kmeans",
                    estimator=KMeans(n_clusters=k, random_state=42, n_init=20),
                    n_clusters=k,
                ),
                ClusterConfig(
                    name="agglomerative",
                    estimator=AgglomerativeClustering(n_clusters=k, linkage="ward"),
                    n_clusters=k,
                ),
                ClusterConfig(
                    name="gmm",
                    estimator=GaussianMixture(n_components=k, covariance_type="full", random_state=42),
                    n_clusters=k,
                ),
            ]
        )

    results: Dict[str, Dict[str, float]] = {}
    best_key = None
    best_score = -np.inf
    best_labels = None
    best_estimator = None
    best_predictable_key = None
    best_predictable_score = -np.inf

    for config in configs:
        labels = _compute_labels(config.estimator, X_transformed)
        metrics = _score_clustering(X_transformed, labels)
        metrics["n_clusters"] = int(config.n_clusters)
        result_key = f"{config.name}_k{config.n_clusters}"
        results[result_key] = metrics

        # Selection prioritaire sur silhouette, puis Davies-Bouldin.
        score = metrics["silhouette"] - 0.05 * metrics["davies_bouldin"]
        if score > best_score:
            best_score = score
            best_key = result_key
            best_labels = labels
            best_estimator = clone(config.estimator).fit(X_transformed)
        if _supports_predict(config.estimator) and score > best_predictable_score:
            best_predictable_score = score
            best_predictable_key = result_key

    if best_key is None or best_labels is None or best_estimator is None:
        raise RuntimeError("Aucun modele de clustering valide n'a ete trouve.")

    # Pour l'industrialisation (API/dashboard), on force un modele avec predict.
    if best_predictable_key is not None and best_predictable_key != best_key:
        best_key = best_predictable_key
        best_estimator_name, best_k = best_key.split("_k")
        best_k = int(best_k)
        if best_estimator_name == "kmeans":
            best_estimator = KMeans(n_clusters=best_k, random_state=42, n_init=20).fit(X_transformed)
            best_labels = best_estimator.labels_
        elif best_estimator_name == "gmm":
            best_estimator = GaussianMixture(
                n_components=best_k, covariance_type="full", random_state=42
            ).fit(X_transformed)
            best_labels = best_estimator.predict(X_transformed)

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("clustering", best_estimator),
        ]
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = models_dir / "clustering_model_comparison.json"
    comparison_csv = models_dir / "clustering_model_comparison.csv"
    profile_csv = models_dir / "cluster_profiles.csv"
    best_info_json = models_dir / "clustering_best_model.json"

    comparison_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results).T.to_csv(comparison_csv, index=True)
    _profile_clusters(X, best_labels).to_csv(profile_csv, index=False)

    best_info = {
        "best_model": best_key,
        "selection_score": float(best_score),
        "metrics": results[best_key],
    }
    best_info_json.write_text(json.dumps(best_info, indent=2), encoding="utf-8")

    joblib.dump(best_pipeline, models_dir / "cluster_model.joblib")
    joblib.dump(best_pipeline, models_dir / f"cluster_model_{best_key}.joblib")

    return {"best_model": best_key, "metrics": results[best_key], "all_results": results}


if __name__ == "__main__":
    results = train_and_compare_clustering_models()
    print("Clustering model comparison:", results)
