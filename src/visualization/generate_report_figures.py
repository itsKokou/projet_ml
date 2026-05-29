"""Generation des figures statiques du rapport technique."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import preprocess_cluster  # noqa: E402
from src.features.build_features import build_cluster_features  # noqa: E402


TEAL = "#157f7f"
ROSE = "#c2415c"
AMBER = "#b7791f"
BLUE = "#3b6ea8"
INK = "#101828"


def _savefig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def _clean_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def fraud_type_summary() -> pd.DataFrame:
    """Agrege le gros fichier fraude par chunks."""
    rows = []
    path = PROJECT_ROOT / "data" / "raw" / "detection_fraude.csv"
    for chunk in pd.read_csv(path, sep=";", chunksize=200_000):
        rows.append(
            chunk.groupby("type").agg(
                transactions=("isFraud", "size"),
                frauds=("isFraud", "sum"),
                amount_sum=("amount", "sum"),
            )
        )
    out = pd.concat(rows).groupby(level=0).sum()
    out["amount_mean"] = out["amount_sum"] / out["transactions"]
    out["fraud_rate"] = out["frauds"] / out["transactions"]
    return out.reset_index().sort_values("fraud_rate", ascending=False)


def plot_fraud_rate_by_type() -> None:
    summary = fraud_type_summary()
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(
        summary["type"],
        summary["fraud_rate"] * 100,
        color=[ROSE if rate > 0 else TEAL for rate in summary["fraud_rate"]],
    )
    ax.set_title("Taux de fraude par type de transaction", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylabel("Taux de fraude (%)")
    ax.set_xlabel("Type de transaction")
    for bar, rate, frauds in zip(bars, summary["fraud_rate"], summary["frauds"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{rate * 100:.3f}%\n{int(frauds)} fraudes",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _clean_axes(ax)
    _savefig("01_fraud_rate_by_type.png")


def plot_fraud_model_comparison() -> None:
    comparison = pd.read_csv(PROJECT_ROOT / "models" / "fraud" / "fraud_model_comparison.csv", index_col=0)
    metrics = ["pr_auc", "recall", "precision", "f1"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(comparison.index))
    width = 0.18
    for i, (metric, color) in enumerate(zip(metrics, [TEAL, ROSE, AMBER, BLUE])):
        ax.bar(x + (i - 1.5) * width, comparison[metric], width=width, label=metric.upper(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index)
    ax.set_ylim(0, 1.08)
    ax.set_title("Comparaison des modeles de detection de fraude", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylabel("Score")
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.25))
    _clean_axes(ax)
    _savefig("02_fraud_model_comparison.png")


def plot_fraud_confusion_matrix() -> None:
    comparison = pd.read_csv(PROJECT_ROOT / "models" / "fraud" / "fraud_model_comparison.csv", index_col=0)
    best = json.loads((PROJECT_ROOT / "models" / "fraud" / "fraud_best_model.json").read_text())["best_model"]
    row = comparison.loc[best]
    positives = row["positive_rate_test"] * row["test_size"]
    tp = round(positives * row["recall"])
    fn = round(positives - tp)
    fp = round(tp / row["precision"] - tp) if row["precision"] else 0
    tn = round(row["test_size"] - tp - fn - fp)
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.set_title(f"Matrice de confusion estimee - {best}", fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks([0, 1], labels=["Pred normal", "Pred fraude"])
    ax.set_yticks([0, 1], labels=["Reel normal", "Reel fraude"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}".replace(",", " "),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color=INK,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig("03_fraud_confusion_matrix_estimated.png")


def plot_fraud_feature_importance() -> None:
    model = joblib.load(PROJECT_ROOT / "models" / "fraud" / "fraud_model.joblib")
    classifier = model.named_steps["classifier"]
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(classifier.feature_importances_))]
    importance = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    importance["feature"] = (
        importance["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("type_", "type=", regex=False)
    )
    top = importance.sort_values("importance", ascending=False).head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.barh(top["feature"], top["importance"], color=TEAL)
    ax.set_title("Variables les plus influentes du modele fraude", fontsize=14, fontweight="bold", color=INK)
    ax.set_xlabel("Importance")
    _clean_axes(ax)
    _savefig("04_fraud_feature_importance.png")


def _cluster_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    raw = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "data_cluster.csv", sep=";")
    model = joblib.load(PROJECT_ROOT / "models" / "clustering" / "cluster_model.joblib")
    prepared = build_cluster_features(preprocess_cluster(raw))
    model_input = prepared.drop(
        columns=[col for col in ("ID", "Response", "Complain", "Dt_Customer") if col in prepared.columns]
    )
    expected = list(model.named_steps["preprocessor"].feature_names_in_)
    model_input = model_input.reindex(columns=expected)
    transformed = model.named_steps["preprocessor"].transform(model_input)
    labels = model.named_steps["clustering"].predict(transformed)
    return transformed, labels, prepared


def plot_cluster_pca() -> None:
    transformed, labels, _ = _cluster_data()
    coords = PCA(n_components=2, random_state=42).fit_transform(transformed)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = [TEAL, ROSE, AMBER, BLUE, "#6b7280", "#7c3aed"]
    for label, color in zip(sorted(np.unique(labels)), colors):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=22, alpha=0.65, label=f"Cluster {label}", color=color)
    ax.set_title("Projection PCA des clients par cluster", fontsize=14, fontweight="bold", color=INK)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=True)
    _clean_axes(ax)
    _savefig("05_cluster_pca_projection.png")


def plot_cluster_heatmap() -> None:
    profiles = pd.read_csv(PROJECT_ROOT / "models" / "clustering" / "cluster_profiles.csv")
    heat_cols = [
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
    heat_cols = [col for col in heat_cols if col in profiles.columns]
    norm = profiles[heat_cols].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(norm.T, aspect="auto", cmap="viridis")
    ax.set_title("Profil normalise des clusters clients", fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks(np.arange(len(profiles["cluster"])), labels=[f"C{int(c)}" for c in profiles["cluster"]])
    ax.set_yticks(np.arange(len(heat_cols)), labels=heat_cols)
    for i in range(len(heat_cols)):
        for j in range(len(profiles)):
            value = norm.iloc[j, i]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Niveau normalise")
    _savefig("06_cluster_profile_heatmap.png")


def main() -> None:
    plot_fraud_rate_by_type()
    plot_fraud_model_comparison()
    plot_fraud_confusion_matrix()
    plot_fraud_feature_importance()
    plot_cluster_pca()
    plot_cluster_heatmap()


if __name__ == "__main__":
    main()
