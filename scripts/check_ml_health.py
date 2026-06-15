"""Verification simple de la sante des artefacts ML (monitoring MVP)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    ROOT / "models/fraud/fraud_model.joblib",
    ROOT / "models/fraud/fraud_best_model.json",
    ROOT / "models/fraud/fraud_model_comparison.json",
    ROOT / "models/fraud/fraud_temporal_metrics.json",
    ROOT / "models/fraud/fraud_error_analysis.json",
    ROOT / "models/fraud/fraud_cost_analysis.json",
    ROOT / "models/clustering/cluster_model.joblib",
    ROOT / "models/clustering/clustering_best_model.json",
]

THRESHOLDS = {
    "fraud_pr_auc_min": 0.90,
    "cluster_silhouette_min": 0.10,
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []

    for path in REQUIRED_FILES:
        if not path.exists():
            errors.append(f"Fichier manquant : {path.relative_to(ROOT)}")

    comparison_path = ROOT / "models/fraud/fraud_model_comparison.json"
    best_path = ROOT / "models/fraud/fraud_best_model.json"
    if comparison_path.exists():
        comparison = _load_json(comparison_path)
        best_name = _load_json(best_path).get("best_model") if best_path.exists() else None
        if best_name and best_name in comparison:
            pr_auc = float(comparison[best_name].get("pr_auc", 0))
            if pr_auc < THRESHOLDS["fraud_pr_auc_min"]:
                errors.append(
                    f"PR-AUC fraude trop basse ({pr_auc:.4f} < {THRESHOLDS['fraud_pr_auc_min']})"
                )

    cluster_best_path = ROOT / "models/clustering/clustering_best_model.json"
    if cluster_best_path.exists():
        cluster_info = _load_json(cluster_best_path)
        silhouette = float(cluster_info.get("metrics", {}).get("silhouette", -1))
        if silhouette < THRESHOLDS["cluster_silhouette_min"]:
            errors.append(
                f"Silhouette clustering trop basse ({silhouette:.4f} < {THRESHOLDS['cluster_silhouette_min']})"
            )

    if comparison_path.exists() and best_path.exists():
        best_name = _load_json(best_path).get("best_model")
        comparison = _load_json(comparison_path)
        if best_name and best_name in comparison:
            model_threshold = float(comparison[best_name].get("threshold", -1))
            error_path = ROOT / "models/fraud/fraud_error_analysis.json"
            cost_path = ROOT / "models/fraud/fraud_cost_analysis.json"
            if error_path.exists():
                error_threshold = float(_load_json(error_path).get("threshold", -2))
                if abs(model_threshold - error_threshold) > 0.001:
                    errors.append(
                        f"Incoherence seuil : comparison={model_threshold} vs error_analysis={error_threshold}"
                    )
            if cost_path.exists():
                cost_threshold = float(_load_json(cost_path).get("selected_threshold", -2))
                if abs(model_threshold - cost_threshold) > 0.001:
                    errors.append(
                        f"Incoherence seuil : comparison={model_threshold} vs cost_analysis={cost_threshold}"
                    )

    if errors:
        print("ML health check : ECHEC")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("ML health check : OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
