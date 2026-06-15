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


def plot_fraud_shap_summary(sample_size: int = 4000) -> None:
    """Resume SHAP global sur un echantillon stratifie (modeles a arbres)."""
    try:
        import shap
    except Exception:
        return

    from sklearn.model_selection import train_test_split

    from src.models.train_fraud_model import _prepare_fraud_dataset

    model = joblib.load(PROJECT_ROOT / "models" / "fraud" / "fraud_model.joblib")
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return

    X, y = _prepare_fraud_dataset()
    if len(X) <= sample_size:
        X_sample = X
    else:
        X_sample, _, _, _ = train_test_split(
            X, y, train_size=sample_size, random_state=42, stratify=y
        )

    X_t = model.named_steps["preprocessor"].transform(X_sample)
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(X_t.shape[1])]

    clean_names = [
        str(n)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("type_", "type=")
        for n in names
    ]

    try:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_t)
    except Exception:
        return

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
    else:
        sv = np.asarray(shap_values)

    summary = pd.DataFrame(
        {"feature": clean_names, "mean_abs_shap": np.abs(sv).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False).head(12).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.barh(summary["feature"], summary["mean_abs_shap"], color=ROSE)
    ax.set_title("Importance SHAP moyenne (classe fraude)", fontsize=14, fontweight="bold", color=INK)
    ax.set_xlabel("|SHAP| moyen")
    _clean_axes(ax)
    _savefig("09_fraud_shap_summary.png")


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


def plot_fraud_temporal_comparison() -> None:
    random_metrics = pd.read_csv(PROJECT_ROOT / "models" / "fraud" / "fraud_model_comparison.csv", index_col=0)
    best = json.loads((PROJECT_ROOT / "models" / "fraud" / "fraud_best_model.json").read_text())["best_model"]
    temporal_path = PROJECT_ROOT / "models" / "fraud" / "fraud_temporal_metrics.json"
    if not temporal_path.exists():
        return

    temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
    random_row = random_metrics.loc[best]
    metrics = ["pr_auc", "recall", "precision", "f1"]
    random_vals = [random_row[m] for m in metrics]
    temporal_vals = [temporal[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width / 2, random_vals, width=width, label="Split aleatoire", color=TEAL)
    ax.bar(x + width / 2, temporal_vals, width=width, label="Split temporel (step)", color=ROSE)
    ax.set_xticks(x, labels=[m.upper() for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_title("Fraude : split aleatoire vs validation temporelle", fontsize=14, fontweight="bold", color=INK)
    ax.legend()
    _clean_axes(ax)
    _savefig("07_fraud_temporal_comparison.png")


def plot_fraud_error_summary() -> None:
    error_path = PROJECT_ROOT / "models" / "fraud" / "fraud_error_analysis.json"
    if not error_path.exists():
        return

    summary = json.loads(error_path.read_text(encoding="utf-8"))
    labels = ["Faux negatifs", "Faux positifs"]
    values = [summary["false_negatives"], summary["false_positives"]]
    colors = [ROSE, AMBER]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Erreurs sur le jeu de test (seuil retenu)", fontsize=14, fontweight="bold", color=INK)
    ax.set_ylabel("Nombre de transactions")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")
    _clean_axes(ax)
    _savefig("08_fraud_error_summary.png")


def plot_fraud_cost_scenarios() -> None:
    cost_path = PROJECT_ROOT / "models" / "fraud" / "fraud_cost_analysis.json"
    if not cost_path.exists():
        return

    data = json.loads(cost_path.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return

    labels = [f"{s['threshold']:.2f}" for s in scenarios]
    fn_loss = [s["fn_financial_loss_units"] / 1e6 for s in scenarios]
    fp_cost = [s["fp_operational_cost_eur"] / 1e3 for s in scenarios]
    selected = [s.get("is_selected_threshold", False) for s in scenarios]
    colors_fn = [TEAL if sel else "#94a3b8" for sel in selected]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, fn_loss, color=colors_fn, label="Perte FN (M unités)")
    ax.bar(x, fp_cost, bottom=fn_loss, color=AMBER, alpha=0.85, label="Coût FP (k €)")
    ax.set_xticks(x, labels=[f"Seuil {label}" for label in labels])
    ax.set_ylabel("Coût (échelle mixte)")
    ax.set_title("Compromis economique FP/FN par seuil", fontsize=14, fontweight="bold", color=INK)
    for i, (fn, fp, sel) in enumerate(zip(fn_loss, fp_cost, selected)):
        if sel:
            ax.text(i, fn + fp + 0.05, "retenu", ha="center", fontsize=9, color=ROSE, fontweight="bold")
    ax.legend()
    _clean_axes(ax)
    _savefig("10_fraud_cost_scenarios.png")


def plot_clustering_elbow() -> None:
    elbow_path = PROJECT_ROOT / "models" / "clustering" / "clustering_elbow.csv"
    if not elbow_path.exists():
        return

    elbow = pd.read_csv(elbow_path)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(elbow["k"], elbow["inertia"], marker="o", color=TEAL, linewidth=2)
    ax.axvline(4, color=ROSE, linestyle="--", linewidth=1.2, label="k retenu = 4")
    ax.set_title("Elbow Method - K-Means (inertie vs k)", fontsize=14, fontweight="bold", color=INK)
    ax.set_xlabel("Nombre de clusters (k)")
    ax.set_ylabel("Inertie")
    ax.legend()
    _clean_axes(ax)
    _savefig("07_clustering_elbow.png")


def plot_dbscan_comparison() -> None:
    dbscan_path = PROJECT_ROOT / "models" / "clustering" / "clustering_dbscan_comparison.csv"
    main_path = PROJECT_ROOT / "models" / "clustering" / "clustering_model_comparison.csv"
    if not dbscan_path.exists() or not main_path.exists():
        return

    dbscan = pd.read_csv(dbscan_path, index_col=0)
    if "status" in dbscan.columns:
        dbscan = dbscan[dbscan["status"] == "evaluated"]
    if dbscan.empty:
        return

    best_dbscan = dbscan.sort_values("silhouette", ascending=False).head(1).iloc[0]
    main = pd.read_csv(main_path, index_col=0)
    gmm = main.loc["gmm_k4"] if "gmm_k4" in main.index else main.sort_values("silhouette", ascending=False).iloc[0]

    labels = ["DBSCAN (meilleur)", "GMM k=4 (retenu)"]
    silhouettes = [best_dbscan["silhouette"], gmm["silhouette"]]
    noise = [best_dbscan.get("noise_ratio", 0) * 100, 0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(labels, silhouettes, color=[AMBER, TEAL])
    axes[0].set_ylim(0, max(silhouettes) * 1.2 + 0.05)
    axes[0].set_title("Silhouette : DBSCAN vs modele retenu", fontweight="bold", color=INK)
    axes[0].set_ylabel("Silhouette")

    axes[1].bar(labels, noise, color=[AMBER, TEAL])
    axes[1].set_title("Part de bruit (%)", fontweight="bold", color=INK)
    axes[1].set_ylabel("Bruit (%)")
    _savefig("08_clustering_dbscan_comparison.png")


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
    plot_fraud_shap_summary()
    plot_fraud_temporal_comparison()
    plot_fraud_error_summary()
    plot_fraud_cost_scenarios()
    plot_cluster_pca()
    plot_clustering_elbow()
    plot_dbscan_comparison()
    plot_cluster_heatmap()


if __name__ == "__main__":
    main()
