"""API FastAPI pour la prediction fraude et segmentation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.preprocessing import preprocess_cluster, preprocess_fraud
from src.features.build_features import build_cluster_features, build_fraud_features


app = FastAPI(title="Projet ML M2 CDSD", version="0.1.0")

FRAUD_DIR = Path("models/fraud")
CLUSTER_DIR = Path("models/clustering")


class FraudRequest(BaseModel):
    payload: Dict[str, Any]


class SegmentRequest(BaseModel):
    payload: Dict[str, Any]


def _safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _load_fraud_artifacts() -> Dict[str, Any]:
    best_info = _safe_json_load(FRAUD_DIR / "fraud_best_model.json")
    best_model_name = best_info.get("best_model")

    model_candidates = []
    if best_model_name:
        model_candidates.append(FRAUD_DIR / f"fraud_model_{best_model_name}.joblib")
    model_candidates.append(FRAUD_DIR / "fraud_model.joblib")

    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        return {}

    comparison = _safe_json_load(FRAUD_DIR / "fraud_model_comparison.json")
    threshold = 0.5
    if best_model_name and best_model_name in comparison:
        threshold = float(comparison[best_model_name].get("threshold", 0.5))

    return {
        "model": joblib.load(model_path),
        "model_name": best_model_name or model_path.stem,
        "threshold": threshold,
    }


@lru_cache(maxsize=1)
def _load_cluster_artifacts() -> Dict[str, Any]:
    best_info = _safe_json_load(CLUSTER_DIR / "clustering_best_model.json")
    best_model_name = best_info.get("best_model")

    model_candidates = []
    if best_model_name:
        model_candidates.append(CLUSTER_DIR / f"cluster_model_{best_model_name}.joblib")
    model_candidates.append(CLUSTER_DIR / "cluster_model.joblib")

    model_path = next((p for p in model_candidates if p.exists()), None)
    if model_path is None:
        return {}

    return {
        "model": joblib.load(model_path),
        "model_name": best_model_name or model_path.stem,
    }


def _align_to_expected_columns(df: pd.DataFrame, model_pipeline) -> pd.DataFrame:
    expected = list(model_pipeline.named_steps["preprocessor"].feature_names_in_)
    aligned = df.reindex(columns=expected)
    return aligned


@app.get("/health")
def health() -> Dict[str, str]:
    """Endpoint de sante de l'API."""
    return {"status": "ok"}


@app.get("/model/info")
def model_info() -> Dict[str, Any]:
    """Expose les modeles charges par l'API."""
    fraud = _load_fraud_artifacts()
    cluster = _load_cluster_artifacts()
    return {
        "fraud_model": fraud.get("model_name", "non_charge"),
        "fraud_threshold": fraud.get("threshold"),
        "cluster_model": cluster.get("model_name", "non_charge"),
    }


@app.post("/predict/fraud")
def predict_fraud(request: FraudRequest) -> Dict[str, Any]:
    """Predire la fraude a partir d'une transaction."""
    artifacts = _load_fraud_artifacts()
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modele fraude indisponible.")

    model = artifacts["model"]
    threshold = float(artifacts["threshold"])

    row = pd.DataFrame([request.payload])
    row = build_fraud_features(preprocess_fraud(row))
    row = row.drop(columns=[c for c in ("nameOrig", "nameDest", "isFlaggedFraud", "isFraud") if c in row.columns])
    row = _align_to_expected_columns(row, model)

    proba = float(model.predict_proba(row)[0, 1])
    pred = int(proba >= threshold)
    return {
        "prediction": pred,
        "probability": proba,
        "threshold": threshold,
        "model": artifacts["model_name"],
    }


@app.post("/predict/segment")
def predict_segment(request: SegmentRequest) -> Dict[str, Any]:
    """Attribuer un segment client a partir du profil fourni."""
    artifacts = _load_cluster_artifacts()
    if not artifacts:
        raise HTTPException(status_code=503, detail="Modele clustering indisponible.")

    model = artifacts["model"]
    row = pd.DataFrame([request.payload])
    row = build_cluster_features(preprocess_cluster(row))
    row = row.drop(columns=[c for c in ("ID", "Response", "Complain", "Dt_Customer") if c in row.columns])
    row = _align_to_expected_columns(row, model)

    cluster_step = model.named_steps["clustering"]
    transformed = model.named_steps["preprocessor"].transform(row)
    if hasattr(cluster_step, "predict"):
        cluster = int(cluster_step.predict(transformed)[0])
    else:
        # Fallback rare si modele sans predict.
        cluster = int(cluster_step.fit_predict(transformed)[0])

    return {
        "segment": cluster,
        "model": artifacts["model_name"],
        "n_input_features": int(np.asarray(transformed).shape[1]),
    }
