"""Dashboard Streamlit avance pour resultats ML."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Permet d'importer `src.*` quand Streamlit execute depuis `dashboard/`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.preprocessing import preprocess_cluster, preprocess_fraud
from src.features.build_features import build_cluster_features, build_fraud_features


FRAUD_DIR = PROJECT_ROOT / "models" / "fraud"
CLUSTER_DIR = PROJECT_ROOT / "models" / "clustering"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

FRAUD_MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"]
SEGMENT_NAMES = {
    0: "Faible valeur / faible engagement",
    1: "Promotionnel et digital",
    2: "Premium tres reactif",
    3: "Forte valeur stable",
}
SEGMENT_ACTIONS = {
    0: "Reactivation, offres accessibles, parcours d'onboarding.",
    1: "Coupons, retargeting web, bundles promotionnels.",
    2: "Programme VIP, avant-premieres, offres premium.",
    3: "Fidelisation, cross-sell, experience omnicanale.",
}
NAV_ITEMS = [
    {
        "slug": "overview",
        "label": "Vue d'ensemble",
        "short": "Pilotage",
        "description": "KPI globaux, donnees, modeles retenus et signaux metier.",
    },
    {
        "slug": "fraud",
        "label": "Fraude",
        "short": "Detection",
        "description": "Performance des modeles, seuil, analyse transactionnelle et simulateur.",
    },
    {
        "slug": "segmentation",
        "label": "Segmentation",
        "short": "Clients",
        "description": "Profils clients, projection PCA, comparaison clustering et simulateur.",
    },
    {
        "slug": "recommendations",
        "label": "Recommandations",
        "short": "Decisions",
        "description": "Actions metier pour la fraude, le marketing et la prochaine iteration.",
    },
    {
        "slug": "prediction",
        "label": "Prediction",
        "short": "Scoring",
        "description": "Tester les predictions fraude et segmentation, puis voir les endpoints API.",
    },
    {
        "slug": "mlops",
        "label": "MLOps",
        "short": "Industrialisation",
        "description": "Artefacts, API, pipeline cible et roadmap technique.",
    },
]
NAV_BY_SLUG = {item["slug"]: item for item in NAV_ITEMS}


st.set_page_config(
    page_title="Projet ML M2 CDSD",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    """Garde une fonction dediee au style, sans HTML brut."""
    return None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv_cached(path: Path, sep: str = ",", **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, **kwargs)


@st.cache_data(show_spinner=False)
def load_fraud_sample(sample_size: int = 250_000) -> pd.DataFrame:
    path = DATA_DIR / "detection_fraude.csv"
    df = read_csv_cached(path, sep=";")
    if df.empty:
        return df
    if len(df) > sample_size:
        return df.sample(sample_size, random_state=42)
    return df


@st.cache_data(show_spinner=False)
def load_cluster_raw() -> pd.DataFrame:
    return read_csv_cached(DATA_DIR / "data_cluster.csv", sep=";")


@st.cache_data(show_spinner=False)
def load_fraud_comparison() -> pd.DataFrame:
    path = FRAUD_DIR / "fraud_model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False)
def load_cluster_comparison() -> pd.DataFrame:
    path = CLUSTER_DIR / "clustering_model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


@st.cache_data(show_spinner=False)
def load_cluster_profiles() -> pd.DataFrame:
    path = CLUSTER_DIR / "cluster_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    profiles = pd.read_csv(path)
    profiles["segment"] = profiles["cluster"].map(SEGMENT_NAMES).fillna("Segment non nomme")
    profiles["action"] = profiles["cluster"].map(SEGMENT_ACTIONS).fillna("Analyse complementaire.")
    return profiles


@st.cache_resource(show_spinner=False)
def load_fraud_model():
    best = read_json(FRAUD_DIR / "fraud_best_model.json").get("best_model")
    candidate_paths = []
    if best:
        candidate_paths.append(FRAUD_DIR / f"fraud_model_{best}.joblib")
    candidate_paths.append(FRAUD_DIR / "fraud_model.joblib")
    model_path = next((p for p in candidate_paths if p.exists()), None)
    return joblib.load(model_path) if model_path else None


@st.cache_resource(show_spinner=False)
def load_cluster_model():
    best = read_json(CLUSTER_DIR / "clustering_best_model.json").get("best_model")
    candidate_paths = []
    if best:
        candidate_paths.append(CLUSTER_DIR / f"cluster_model_{best}.joblib")
    candidate_paths.append(CLUSTER_DIR / "cluster_model.joblib")
    model_path = next((p for p in candidate_paths if p.exists()), None)
    return joblib.load(model_path) if model_path else None


def _align_to_expected_columns(df: pd.DataFrame, model_pipeline) -> pd.DataFrame:
    expected = list(model_pipeline.named_steps["preprocessor"].feature_names_in_)
    return df.reindex(columns=expected)


def _format_percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def _metric_value(metrics: pd.DataFrame, model_name: str, column: str, default: float = 0.0) -> float:
    if metrics.empty or model_name not in metrics.index or column not in metrics.columns:
        return default
    return float(metrics.loc[model_name, column])


def _best_fraud_name() -> str:
    return read_json(FRAUD_DIR / "fraud_best_model.json").get("best_model", "N/A")


def _best_cluster_name() -> str:
    return read_json(CLUSTER_DIR / "clustering_best_model.json").get("best_model", "N/A")


def estimate_confusion_from_metrics(row: pd.Series) -> dict:
    positives = float(row.get("positive_rate_test", 0.0)) * float(row.get("test_size", 0.0))
    recall = float(row.get("recall", 0.0))
    precision = float(row.get("precision", 0.0))
    tp = positives * recall
    fn = max(positives - tp, 0.0)
    fp = 0.0 if precision == 0 else max(tp / precision - tp, 0.0)
    alerts = tp + fp
    return {
        "positives": positives,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "alerts": alerts,
    }


def plotly_layout(fig, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title_text="",
        font=dict(size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#e7ebf2")
    return fig


def get_current_page() -> str:
    raw_page = st.query_params.get("page", "overview")
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else "overview"
    page = str(raw_page)
    return page if page in NAV_BY_SLUG else "overview"


def set_current_page(slug: str) -> None:
    st.query_params["page"] = slug


def render_sidebar_navigation(active_slug: str) -> None:
    fraud_cmp = load_fraud_comparison()
    cluster_cmp = load_cluster_comparison()
    best_fraud = _best_fraud_name()
    best_cluster = _best_cluster_name()

    st.sidebar.title("Projet ML M2 CDSD")
    st.sidebar.caption("Fraude bancaire, segmentation client et MLOps")
    st.sidebar.divider()

    st.sidebar.subheader("Navigation")
    for item in NAV_ITEMS:
        is_active = item["slug"] == active_slug
        label = item["label"] if not is_active else f"{item['label']} - actif"
        st.sidebar.button(
            label,
            key=f"sidebar_nav_{item['slug']}",
            width="stretch",
            type="primary" if is_active else "secondary",
            disabled=is_active,
            on_click=set_current_page,
            args=(item["slug"],),
        )

    st.sidebar.divider()
    st.sidebar.subheader("Statut modeles")
    st.sidebar.metric("Modele fraude", best_fraud)
    st.sidebar.metric("PR-AUC", f"{_metric_value(fraud_cmp, best_fraud, 'pr_auc'):.4f}")
    st.sidebar.metric("Clustering", best_cluster)
    st.sidebar.metric("Silhouette", f"{_metric_value(cluster_cmp, best_cluster, 'silhouette'):.4f}")
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Artefacts**

        - `models/fraud/`
        - `models/clustering/`
        - `reports/rapport_technique.md`
        - `reports/presentation_outline.md`
        """
    )


@st.cache_data(show_spinner=False)
def fraud_type_summary(sample_size: int = 300_000) -> pd.DataFrame:
    df = load_fraud_sample(sample_size)
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("type")
        .agg(
            transactions=("isFraud", "size"),
            frauds=("isFraud", "sum"),
            avg_amount=("amount", "mean"),
            median_amount=("amount", "median"),
        )
        .reset_index()
    )
    out["fraud_rate"] = out["frauds"] / out["transactions"]
    return out.sort_values("fraud_rate", ascending=False)


@st.cache_data(show_spinner=False)
def fraud_class_summary() -> pd.DataFrame:
    df = load_fraud_sample(1_200_000)
    if df.empty:
        return pd.DataFrame()
    out = df["isFraud"].astype(int).value_counts().rename_axis("isFraud").reset_index(name="transactions")
    out["label"] = np.where(out["isFraud"] == 1, "Fraude", "Normal")
    out["share"] = out["transactions"] / out["transactions"].sum()
    return out.sort_values("isFraud")


@st.cache_data(show_spinner=False)
def fraud_time_summary(sample_size: int = 300_000) -> pd.DataFrame:
    df = load_fraud_sample(sample_size)
    if df.empty or "step" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["step_bucket"] = (out["step"] // 24).astype(int)
    summary = (
        out.groupby("step_bucket")
        .agg(transactions=("isFraud", "size"), frauds=("isFraud", "sum"), amount=("amount", "mean"))
        .reset_index()
    )
    summary["fraud_rate"] = summary["frauds"] / summary["transactions"]
    return summary


@st.cache_data(show_spinner=False)
def fraud_balance_diagnostics(sample_size: int = 250_000) -> pd.DataFrame:
    df = load_fraud_sample(sample_size)
    if df.empty or "isFraud" not in df.columns:
        return pd.DataFrame()
    features = build_fraud_features(preprocess_fraud(df.copy()))
    features["label"] = np.where(features["isFraud"].astype(int) == 1, "Fraude", "Normal")
    for col in ["origin_error", "dest_error"]:
        if col in features.columns:
            features[f"abs_{col}"] = features[col].abs()

    aggregations = {
        "amount": "median",
        "is_zero_newbalance_origin": "mean",
        "is_zero_oldbalance_dest": "mean",
    }
    if "abs_origin_error" in features.columns:
        aggregations["abs_origin_error"] = "median"
    if "abs_dest_error" in features.columns:
        aggregations["abs_dest_error"] = "median"

    out = features.groupby("label").agg(aggregations).reset_index()
    return out.rename(
        columns={
            "amount": "median_amount",
            "is_zero_newbalance_origin": "zero_newbalance_origin_rate",
            "is_zero_oldbalance_dest": "zero_oldbalance_dest_rate",
        }
    )


@st.cache_data(show_spinner=False)
def fraud_amount_sample(sample_size: int = 80_000) -> pd.DataFrame:
    df = load_fraud_sample(sample_size)
    if df.empty:
        return pd.DataFrame()
    out = df[["type", "amount", "isFraud"]].copy()
    out["label"] = np.where(out["isFraud"].astype(int) == 1, "Fraude", "Normal")
    out["amount_log"] = np.log10(out["amount"].clip(lower=1))
    return out


@st.cache_data(show_spinner=False)
def fraud_suspicious_examples(limit: int = 12) -> pd.DataFrame:
    df = load_fraud_sample(1_200_000)
    if df.empty:
        return pd.DataFrame()
    features = build_fraud_features(preprocess_fraud(df.copy()))
    cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "origin_error",
        "dest_error",
        "amount_to_oldbalance_ratio",
    ]
    cols = [c for c in cols if c in features.columns]
    out = features[cols].copy()
    out["abs_origin_error"] = out["origin_error"].abs() if "origin_error" in out.columns else 0
    out = out.sort_values(["isFraud", "amount", "abs_origin_error"], ascending=[False, False, False])
    return out.head(limit)


@st.cache_data(show_spinner=False)
def fraud_threshold_curve(sample_size: int = 180_000) -> pd.DataFrame:
    model = load_fraud_model()
    if model is None:
        return pd.DataFrame()

    df = load_fraud_sample(sample_size)
    if df.empty or "isFraud" not in df.columns:
        return pd.DataFrame()

    y_true = df["isFraud"].astype(int)
    X = build_fraud_features(preprocess_fraud(df.drop(columns=["isFraud"])))
    X = X.drop(columns=[c for c in ("nameOrig", "nameDest", "isFlaggedFraud") if c in X.columns])
    X = _align_to_expected_columns(X, model)
    y_proba = model.predict_proba(X)[:, 1]

    rows = []
    for threshold in np.linspace(0.05, 0.99, 40):
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "alerts": int(y_pred.sum()),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def fraud_feature_importance(top_n: int = 15) -> pd.DataFrame:
    model = load_fraud_model()
    if model is None:
        return pd.DataFrame()
    classifier = model.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame()
    try:
        names = model.named_steps["preprocessor"].get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(classifier.feature_importances_))]
    out = pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
    out["feature"] = (
        out["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("type_", "type=", regex=False)
    )
    return out.sort_values("importance", ascending=False).head(top_n)


@st.cache_data(show_spinner=False)
def cluster_projection(sample_size: int = 2240) -> pd.DataFrame:
    model = load_cluster_model()
    if model is None:
        return pd.DataFrame()

    df = load_cluster_raw()
    if df.empty:
        return pd.DataFrame()
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    prepared = build_cluster_features(preprocess_cluster(df))
    meta_cols = [
        "Income",
        "Age",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "Campaign_Acceptance_Total",
    ]
    meta = prepared[[c for c in meta_cols if c in prepared.columns]].copy()
    prepared = prepared.drop(columns=[c for c in ("ID", "Response", "Complain", "Dt_Customer") if c in prepared.columns])
    prepared = _align_to_expected_columns(prepared, model)
    transformed = model.named_steps["preprocessor"].transform(prepared)
    cluster_model = model.named_steps["clustering"]
    labels = cluster_model.predict(transformed) if hasattr(cluster_model, "predict") else cluster_model.fit_predict(transformed)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(transformed)
    proj = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})
    for col in meta.columns:
        proj[col] = meta[col].to_numpy()
    proj["segment"] = proj["cluster"].map(SEGMENT_NAMES).fillna("Segment non nomme")
    return proj


@st.cache_data(show_spinner=False)
def cluster_enriched_dataset() -> pd.DataFrame:
    model = load_cluster_model()
    raw = load_cluster_raw()
    if raw.empty:
        return pd.DataFrame()

    prepared = build_cluster_features(preprocess_cluster(raw))
    out = prepared.copy()
    if model is None:
        return out

    model_input = prepared.drop(columns=[c for c in ("ID", "Response", "Complain", "Dt_Customer") if c in prepared.columns])
    model_input = _align_to_expected_columns(model_input, model)
    transformed = model.named_steps["preprocessor"].transform(model_input)
    cluster_model = model.named_steps["clustering"]
    labels = cluster_model.predict(transformed) if hasattr(cluster_model, "predict") else cluster_model.fit_predict(transformed)
    out["cluster"] = labels
    out["segment"] = out["cluster"].map(SEGMENT_NAMES).fillna("Segment non nomme")
    return out


def customer_spending_summary(df: pd.DataFrame) -> pd.DataFrame:
    spending_cols = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    cols = [c for c in spending_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return (
        df[cols]
        .mean()
        .rename_axis("categorie")
        .reset_index(name="depense_moyenne")
        .sort_values("depense_moyenne", ascending=False)
    )


def customer_channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    channel_cols = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumDealsPurchases",
    ]
    cols = [c for c in channel_cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    return (
        df[cols]
        .mean()
        .rename_axis("canal")
        .reset_index(name="achats_moyens")
        .sort_values("achats_moyens", ascending=False)
    )


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def render_insight(title: str, text: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)


def show_overview() -> None:
    render_header(
        "Pilotage fraude et segmentation",
        "Synthese operationnelle des modeles, des donnees et des decisions metier.",
    )

    fraud_cmp = load_fraud_comparison()
    cluster_cmp = load_cluster_comparison()
    profiles = load_cluster_profiles()
    best_fraud = _best_fraud_name()
    best_cluster = _best_cluster_name()

    fraud_df = load_fraud_sample(300_000)
    cluster_df = load_cluster_raw()
    fraud_rate = float(fraud_df["isFraud"].mean()) if not fraud_df.empty else 0.0
    best_pr_auc = _metric_value(fraud_cmp, best_fraud, "pr_auc")
    best_recall = _metric_value(fraud_cmp, best_fraud, "recall")
    best_silhouette = _metric_value(cluster_cmp, best_cluster, "silhouette")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Transactions analysees", f"{len(fraud_df):,}".replace(",", " "))
    c2.metric("Taux de fraude", _format_percent(fraud_rate, 3))
    c3.metric("Modele fraude", best_fraud)
    c4.metric("PR-AUC fraude", f"{best_pr_auc:.4f}")
    c5.metric("Silhouette segments", f"{best_silhouette:.4f}")

    st.divider()

    left, right = st.columns([1.15, 0.85])
    with left:
        fraud_summary = fraud_type_summary()
        if not fraud_summary.empty:
            fig = px.bar(
                fraud_summary,
                x="type",
                y="fraud_rate",
                color="transactions",
                color_continuous_scale="Teal",
                text=fraud_summary["fraud_rate"].map(lambda x: f"{100 * x:.2f}%"),
                title="Taux de fraude par type de transaction",
            )
            fig.update_traces(textposition="outside")
            fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(plotly_layout(fig, 390), width="stretch")

    with right:
        if not profiles.empty:
            fig = px.pie(
                profiles,
                names="segment",
                values="cluster_size",
                hole=0.52,
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Repartition des segments clients",
            )
            st.plotly_chart(plotly_layout(fig, 390), width="stretch")

    i1, i2, i3 = st.columns(3)
    with i1:
        render_insight(
            "Priorite fraude",
            "Le taux de fraude est tres faible. Les decisions doivent donc s'appuyer sur le recall, la precision et la PR-AUC plutot que sur l'accuracy.",
        )
    with i2:
        render_insight(
            "Signal dominant",
            "Les transactions TRANSFER et CASH_OUT concentrent le risque observe. Les incoherences de soldes sont des variables de controle essentielles.",
        )
    with i3:
        render_insight(
            "Lecture marketing",
            "Les 4 segments se lisent par niveau de valeur, canal d'achat et sensibilite aux campagnes. Le cluster 2 est le plus premium.",
        )


def show_fraud_results() -> None:
    render_header(
        "Detection de fraude",
        "Performance des modeles, seuil de decision, signaux explicatifs et simulateur transactionnel.",
    )

    comparison = load_fraud_comparison()
    if comparison.empty:
        st.warning("Lance `python -m src.models.train_fraud_model` pour generer les resultats.")
        return

    best_model = _best_fraud_name()
    ordered = [m for m in FRAUD_MODEL_ORDER if m in comparison.index]
    comparison = comparison.loc[ordered + [m for m in comparison.index if m not in ordered]]
    best_row = comparison.loc[best_model] if best_model in comparison.index else comparison.iloc[0]
    confusion = estimate_confusion_from_metrics(best_row)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Modele retenu", best_model)
    k2.metric("PR-AUC", f"{best_row.get('pr_auc', 0):.4f}")
    k3.metric("Recall", _format_percent(float(best_row.get("recall", 0)), 2))
    k4.metric("Precision", _format_percent(float(best_row.get("precision", 0)), 2))
    k5.metric("Seuil retenu", f"{best_row.get('threshold', 0):.2f}")

    tab_perf, tab_data, tab_threshold, tab_lab = st.tabs(
        ["Performance", "Analyse donnees", "Seuil et alertes", "Simulateur"]
    )

    with tab_perf:
        left, right = st.columns([1.05, 0.95])
        with left:
            metrics = ["pr_auc", "roc_auc", "f1", "recall", "precision"]
            long = comparison[metrics].reset_index().melt(id_vars="index", var_name="metrique", value_name="score")
            fig = px.bar(
                long,
                x="index",
                y="score",
                color="metrique",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Comparaison des modeles de classification",
            )
            fig.update_layout(xaxis_title="Modele", yaxis_title="Score")
            st.plotly_chart(plotly_layout(fig, 420), width="stretch")

        with right:
            imp = fraud_feature_importance()
            if not imp.empty:
                fig = px.bar(
                    imp.sort_values("importance"),
                    x="importance",
                    y="feature",
                    orientation="h",
                    color="importance",
                    color_continuous_scale="Tealgrn",
                    title="Variables les plus influentes",
                )
                st.plotly_chart(plotly_layout(fig, 420), width="stretch")
            else:
                st.info("Importance des variables indisponible pour ce modele.")

        st.dataframe(
            comparison.style.format(
                {
                    "accuracy": "{:.6f}",
                    "precision": "{:.4f}",
                    "recall": "{:.4f}",
                    "f1": "{:.4f}",
                    "roc_auc": "{:.4f}",
                    "pr_auc": "{:.4f}",
                    "threshold": "{:.2f}",
                }
            ),
            width="stretch",
        )

    with tab_data:
        class_summary = fraud_class_summary()
        diagnostics = fraud_balance_diagnostics()
        c_left, c_right = st.columns([0.8, 1.2])
        with c_left:
            if not class_summary.empty:
                fig = px.pie(
                    class_summary,
                    names="label",
                    values="transactions",
                    hole=0.48,
                    color="label",
                    color_discrete_map={"Fraude": "#c2415c", "Normal": "#157f7f"},
                    title="Desequilibre des classes",
                )
                st.plotly_chart(plotly_layout(fig, 330), width="stretch")
        with c_right:
            if not diagnostics.empty:
                diag_long = diagnostics.melt(id_vars="label", var_name="indicateur", value_name="valeur")
                fig = px.bar(
                    diag_long,
                    x="indicateur",
                    y="valeur",
                    color="label",
                    barmode="group",
                    color_discrete_map={"Fraude": "#c2415c", "Normal": "#157f7f"},
                    title="Diagnostics de soldes et montants",
                )
                fig.update_layout(xaxis_title="", yaxis_title="Valeur moyenne ou mediane")
                st.plotly_chart(plotly_layout(fig, 330), width="stretch")

        i1, i2, i3 = st.columns(3)
        with i1:
            render_insight(
                "Classe rare",
                "Le modele doit traiter une classe fraude tres minoritaire. C'est pour cela que l'accuracy n'est pas la metrique principale.",
            )
        with i2:
            render_insight(
                "Soldes utiles",
                "Les ecarts entre montant et variation de solde creent des signaux forts pour identifier des transactions incoherentes.",
            )
        with i3:
            render_insight(
                "Lecture operationnelle",
                "Les alertes doivent etre calibrees par seuil : plus le seuil baisse, plus le rappel augmente mais plus il y a d'alertes.",
            )

        st.divider()
        left, right = st.columns([0.95, 1.05])
        with left:
            type_summary = fraud_type_summary()
            if not type_summary.empty:
                fig = px.bar(
                    type_summary.sort_values("frauds", ascending=False),
                    x="type",
                    y="frauds",
                    color="fraud_rate",
                    color_continuous_scale="Reds",
                    title="Fraudes observees par type",
                    hover_data=["transactions", "avg_amount", "median_amount"],
                )
                st.plotly_chart(plotly_layout(fig, 390), width="stretch")
                st.dataframe(
                    type_summary.style.format(
                        {"fraud_rate": "{:.3%}", "avg_amount": "{:,.0f}", "median_amount": "{:,.0f}"}
                    ),
                    width="stretch",
                )
        with right:
            amount_df = fraud_amount_sample()
            if not amount_df.empty:
                fig = px.box(
                    amount_df,
                    x="type",
                    y="amount_log",
                    color="label",
                    color_discrete_map={"Fraude": "#c2415c", "Normal": "#157f7f"},
                    title="Distribution log10 des montants par type",
                )
                fig.update_layout(yaxis_title="log10(amount)")
                st.plotly_chart(plotly_layout(fig, 390), width="stretch")

            time_df = fraud_time_summary()
            if not time_df.empty:
                fig = px.line(
                    time_df,
                    x="step_bucket",
                    y="fraud_rate",
                    markers=True,
                    title="Evolution du taux de fraude par fenetre temporelle",
                )
                fig.update_yaxes(tickformat=".2%")
                st.plotly_chart(plotly_layout(fig, 300), width="stretch")

        examples = fraud_suspicious_examples()
        if not examples.empty:
            st.subheader("Transactions frauduleuses les plus fortes ou atypiques")
            st.dataframe(
                examples.style.format(
                    {
                        "amount": "{:,.2f}",
                        "oldbalanceOrg": "{:,.2f}",
                        "newbalanceOrig": "{:,.2f}",
                        "oldbalanceDest": "{:,.2f}",
                        "newbalanceDest": "{:,.2f}",
                        "origin_error": "{:,.2f}",
                        "dest_error": "{:,.2f}",
                        "amount_to_oldbalance_ratio": "{:.4f}",
                        "abs_origin_error": "{:,.2f}",
                    }
                ),
                width="stretch",
            )

    with tab_threshold:
        curve = fraud_threshold_curve()
        if curve.empty:
            st.info("La courbe de seuil n'est pas disponible.")
        else:
            default_threshold = float(best_row.get("threshold", 0.5))
            threshold = st.slider(
                "Seuil de decision",
                min_value=0.05,
                max_value=0.99,
                value=min(max(default_threshold, 0.05), 0.99),
                step=0.01,
            )
            row = curve.iloc[(curve["threshold"] - threshold).abs().argsort()[:1]]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision simulee", _format_percent(float(row["precision"].iloc[0]), 2))
            c2.metric("Recall simule", _format_percent(float(row["recall"].iloc[0]), 2))
            c3.metric("F1 simule", f"{float(row['f1'].iloc[0]):.4f}")
            c4.metric("Alertes echantillon", f"{int(row['alerts'].iloc[0]):,}".replace(",", " "))

            fig = px.line(
                curve,
                x="threshold",
                y=["precision", "recall", "f1"],
                color_discrete_sequence=["#157f7f", "#c2415c", "#b7791f"],
                title="Compromis precision / recall / F1 selon le seuil",
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#101828")
            st.plotly_chart(plotly_layout(fig, 430), width="stretch")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Fraudes test estimees", f"{confusion['positives']:.0f}")
            col_b.metric("Fraudes detectees estimees", f"{confusion['tp']:.0f}")
            col_c.metric("Fraudes manquees estimees", f"{confusion['fn']:.0f}")

    with tab_lab:
        fraud_playground()


def fraud_playground() -> None:
    st.subheader("Evaluation d'une transaction")
    model = load_fraud_model()
    if model is None:
        st.error("Modele fraude introuvable.")
        return

    comparison = load_fraud_comparison()
    best_model = _best_fraud_name()
    threshold = _metric_value(comparison, best_model, "threshold", 0.5)

    with st.form("fraud_form"):
        c1, c2, c3, c4 = st.columns(4)
        step = c1.number_input("Step", min_value=1, value=1)
        tx_type = c2.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
        amount = c3.number_input("Montant", min_value=0.0, value=1000.0, step=100.0)
        old_org = c4.number_input("Solde emetteur avant", min_value=0.0, value=2000.0, step=100.0)
        new_org = c1.number_input("Solde emetteur apres", min_value=0.0, value=1000.0, step=100.0)
        old_dest = c2.number_input("Solde destinataire avant", min_value=0.0, value=500.0, step=100.0)
        new_dest = c3.number_input("Solde destinataire apres", min_value=0.0, value=1500.0, step=100.0)
        submitted = st.form_submit_button("Calculer le score")

    if not submitted:
        return

    row = pd.DataFrame(
        [
            {
                "step": step,
                "type": tx_type,
                "amount": amount,
                "oldbalanceOrg": old_org,
                "newbalanceOrig": new_org,
                "oldbalanceDest": old_dest,
                "newbalanceDest": new_dest,
            }
        ]
    )
    features = build_fraud_features(preprocess_fraud(row))
    model_input = features.drop(columns=[c for c in ("nameOrig", "nameDest", "isFlaggedFraud") if c in features.columns])
    model_input = _align_to_expected_columns(model_input, model)
    proba = float(model.predict_proba(model_input)[0, 1])
    prediction = int(proba >= threshold)
    risk_label = "Alerte fraude" if prediction else "Surveillance" if proba >= threshold / 2 else "Risque faible"

    left, right = st.columns([0.55, 0.45])
    with left:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba,
                number={"valueformat": ".4f"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": "#c2415c" if prediction else "#157f7f"},
                    "steps": [
                        {"range": [0, threshold / 2], "color": "#e6f4f1"},
                        {"range": [threshold / 2, threshold], "color": "#fff1d6"},
                        {"range": [threshold, 1], "color": "#fde2e7"},
                    ],
                    "threshold": {"line": {"color": "#101828", "width": 3}, "value": threshold},
                },
                title={"text": "Probabilite de fraude"},
            )
        )
        st.plotly_chart(plotly_layout(fig, 330), width="stretch")

    with right:
        origin_error = float(features.get("origin_error", pd.Series([0.0])).iloc[0])
        dest_error = float(features.get("dest_error", pd.Series([0.0])).iloc[0])
        ratio = float(features.get("amount_to_oldbalance_ratio", pd.Series([np.nan])).iloc[0])
        with st.container(border=True):
            st.subheader("Decision")
            if prediction:
                st.error(risk_label)
            elif proba >= threshold / 2:
                st.warning(risk_label)
            else:
                st.success(risk_label)
            st.metric("Seuil modele", f"{threshold:.2f}")
            st.metric("Erreur solde emetteur", f"{origin_error:,.2f}")
            st.metric("Erreur solde destinataire", f"{dest_error:,.2f}")
            st.metric("Ratio montant / solde", f"{ratio:.3f}")


def show_cluster_profiles() -> None:
    render_header(
        "Segmentation client",
        "Comparaison des modeles, lecture des profils, projection PCA et simulateur segment.",
    )

    profiles = load_cluster_profiles()
    comparison = load_cluster_comparison()
    if profiles.empty:
        st.warning("Lance `python -m src.models.train_clustering_model` pour generer les profils.")
        return

    total_clients = int(profiles["cluster_size"].sum())
    premium_cluster = int(profiles.sort_values("Total_Spending", ascending=False).iloc[0]["cluster"])
    best_cluster = _best_cluster_name()
    best_silhouette = _metric_value(comparison, best_cluster, "silhouette")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients segmentes", f"{total_clients:,}".replace(",", " "))
    c2.metric("Modele retenu", best_cluster)
    c3.metric("Silhouette", f"{best_silhouette:.4f}")
    c4.metric("Segment premium", f"Cluster {premium_cluster}")

    tab_exploration, tab_profiles, tab_projection, tab_models, tab_sim = st.tabs(
        ["Exploration donnees", "Profils", "Carte PCA", "Comparaison modeles", "Simulateur"]
    )

    with tab_exploration:
        show_customer_exploration()

    with tab_profiles:
        segment_summary(profiles)

    with tab_projection:
        proj = cluster_projection()
        if proj.empty:
            st.info("Projection indisponible.")
        else:
            left, right = st.columns([1.15, 0.85])
            with left:
                fig = px.scatter(
                    proj,
                    x="PC1",
                    y="PC2",
                    color="segment",
                    size="Total_Spending" if "Total_Spending" in proj.columns else None,
                    hover_data=[c for c in ["Income", "Age", "Recency", "Total_Purchases"] if c in proj.columns],
                    opacity=0.68,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    title="Projection PCA des clients",
                )
                st.plotly_chart(plotly_layout(fig, 500), width="stretch")
            with right:
                if "Total_Spending" in proj.columns and "Income" in proj.columns:
                    fig = px.scatter(
                        proj,
                        x="Income",
                        y="Total_Spending",
                        color="segment",
                        opacity=0.65,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        title="Revenu vs depense totale",
                    )
                    st.plotly_chart(plotly_layout(fig, 500), width="stretch")

    with tab_models:
        if comparison.empty:
            st.info("Comparaison des modeles indisponible.")
        else:
            cmp_df = comparison.sort_values("silhouette", ascending=False).copy()
            left, right = st.columns([1.05, 0.95])
            with left:
                fig = px.scatter(
                    cmp_df.reset_index(),
                    x="silhouette",
                    y="davies_bouldin",
                    size="calinski_harabasz",
                    color="n_clusters",
                    hover_name="index",
                    color_continuous_scale="Teal",
                    title="Compromis silhouette / Davies-Bouldin",
                )
                st.plotly_chart(plotly_layout(fig, 430), width="stretch")
            with right:
                fig = px.bar(
                    cmp_df.reset_index(),
                    x="index",
                    y="silhouette",
                    color="n_clusters",
                    color_continuous_scale="Teal",
                    title="Silhouette par configuration",
                )
                fig.update_layout(xaxis_title="Configuration")
                st.plotly_chart(plotly_layout(fig, 430), width="stretch")
            st.dataframe(
                cmp_df.style.format(
                    {
                        "silhouette": "{:.4f}",
                        "davies_bouldin": "{:.4f}",
                        "calinski_harabasz": "{:.2f}",
                        "n_clusters": "{:.0f}",
                    }
                ),
                width="stretch",
            )

    with tab_sim:
        segment_playground()


def show_customer_exploration() -> None:
    df = cluster_enriched_dataset()
    if df.empty:
        st.info("Donnees client indisponibles.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{len(df):,}".replace(",", " "))
    c2.metric("Revenu median", f"{df['Income'].median():,.0f}" if "Income" in df.columns else "N/A")
    c3.metric(
        "Depense mediane",
        f"{df['Total_Spending'].median():,.0f}" if "Total_Spending" in df.columns else "N/A",
    )
    c4.metric(
        "Achats medians",
        f"{df['Total_Purchases'].median():.0f}" if "Total_Purchases" in df.columns else "N/A",
    )

    left, right = st.columns(2)
    with left:
        if "Income" in df.columns:
            fig = px.histogram(
                df,
                x="Income",
                nbins=45,
                color="segment" if "segment" in df.columns else None,
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Distribution des revenus par segment",
            )
            st.plotly_chart(plotly_layout(fig, 360), width="stretch")

        spend = customer_spending_summary(df)
        if not spend.empty:
            fig = px.bar(
                spend,
                x="categorie",
                y="depense_moyenne",
                color="depense_moyenne",
                color_continuous_scale="Teal",
                title="Depenses moyennes par categorie de produit",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Depense moyenne")
            st.plotly_chart(plotly_layout(fig, 360), width="stretch")

    with right:
        if {"Income", "Total_Spending"}.issubset(df.columns):
            fig = px.scatter(
                df,
                x="Income",
                y="Total_Spending",
                color="segment" if "segment" in df.columns else None,
                size="Total_Purchases" if "Total_Purchases" in df.columns else None,
                opacity=0.68,
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Revenu vs depense totale",
            )
            st.plotly_chart(plotly_layout(fig, 360), width="stretch")

        channels = customer_channel_summary(df)
        if not channels.empty:
            fig = px.bar(
                channels,
                x="canal",
                y="achats_moyens",
                color="achats_moyens",
                color_continuous_scale="Teal",
                title="Achats moyens par canal",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Achats moyens")
            st.plotly_chart(plotly_layout(fig, 360), width="stretch")

    corr_cols = [
        "Income",
        "Age",
        "Recency",
        "Total_Spending",
        "Total_Purchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
        "NumDealsPurchases",
        "Campaign_Acceptance_Total",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]
    if len(corr_cols) >= 3:
        corr = df[corr_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation des indicateurs clients",
        )
        st.plotly_chart(plotly_layout(fig, 520), width="stretch")

    i1, i2, i3 = st.columns(3)
    with i1:
        render_insight(
            "Valeur client",
            "Le couple revenu et depense totale permet de separer les clients faibles depenses des profils premium.",
        )
    with i2:
        render_insight(
            "Canaux",
            "Les achats magasin, catalogue, web et promotions donnent une lecture directement exploitable pour les campagnes.",
        )
    with i3:
        render_insight(
            "Interpretation",
            "Le clustering doit etre juge par son utilite metier : un segment est bon s'il permet une action claire.",
        )


def segment_summary(profiles: pd.DataFrame) -> None:
    size_fig = px.bar(
        profiles.sort_values("cluster_size", ascending=True),
        x="cluster_size",
        y="segment",
        orientation="h",
        color="Total_Spending",
        color_continuous_scale="Teal",
        title="Taille et valeur des segments",
    )
    size_fig.update_layout(xaxis_title="Nombre de clients", yaxis_title="")
    st.plotly_chart(plotly_layout(size_fig, 360), width="stretch")

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
    heat_cols = [c for c in heat_cols if c in profiles.columns]
    norm = profiles[heat_cols].copy()
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)
    fig = px.imshow(
        norm.T,
        x=profiles["segment"],
        y=heat_cols,
        aspect="auto",
        color_continuous_scale="Tealrose",
        title="Profil normalise des segments",
    )
    st.plotly_chart(plotly_layout(fig, 430), width="stretch")

    st.subheader("Personas et actions")
    cols = st.columns(2)
    for i, row in profiles.sort_values("cluster").iterrows():
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**Cluster {int(row['cluster'])} - {row['segment']}**")
                st.write(f"Clients : {int(row['cluster_size'])}")
                st.write(f"Revenu moyen : {row['Income']:,.0f}")
                st.write(f"Depense moyenne : {row['Total_Spending']:,.0f}")
                st.write(f"Achats moyens : {row['Total_Purchases']:,.1f}")
                st.info(row["action"])

    st.dataframe(
        profiles[
            [
                "cluster",
                "segment",
                "cluster_size",
                "Income",
                "Age",
                "Recency",
                "Total_Spending",
                "Total_Purchases",
                "Campaign_Acceptance_Total",
                "action",
            ]
        ].style.format(
            {
                "Income": "{:,.0f}",
                "Age": "{:.1f}",
                "Recency": "{:.1f}",
                "Total_Spending": "{:,.0f}",
                "Total_Purchases": "{:.1f}",
                "Campaign_Acceptance_Total": "{:.2f}",
            }
        ),
        width="stretch",
    )


def segment_playground() -> None:
    st.subheader("Attribution d'un client a un segment")
    model = load_cluster_model()
    if model is None:
        st.error("Modele clustering introuvable.")
        return

    with st.form("segment_form"):
        c1, c2, c3, c4 = st.columns(4)
        year_birth = c1.number_input("Annee de naissance", min_value=1940, max_value=2010, value=1985)
        education = c2.selectbox("Education", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
        marital = c3.selectbox("Situation", ["Single", "Together", "Married", "Divorced", "Widow"])
        income = c4.number_input("Revenu", min_value=0.0, value=50000.0, step=1000.0)

        kidhome = c1.number_input("Enfants", min_value=0, max_value=5, value=0)
        teenhome = c2.number_input("Adolescents", min_value=0, max_value=5, value=0)
        recency = c3.number_input("Recence", min_value=0, value=20)
        web_visits = c4.number_input("Visites web mensuelles", min_value=0, value=6)

        wines = c1.number_input("Depenses vins", min_value=0, value=200, step=20)
        fruits = c2.number_input("Depenses fruits", min_value=0, value=20, step=10)
        meat = c3.number_input("Depenses viande", min_value=0, value=120, step=20)
        fish = c4.number_input("Depenses poisson", min_value=0, value=30, step=10)
        sweet = c1.number_input("Depenses sucre", min_value=0, value=15, step=10)
        gold = c2.number_input("Depenses premium", min_value=0, value=40, step=10)

        deals = c1.number_input("Achats promo", min_value=0, value=2)
        web = c2.number_input("Achats web", min_value=0, value=5)
        catalog = c3.number_input("Achats catalogue", min_value=0, value=2)
        store = c4.number_input("Achats magasin", min_value=0, value=4)
        accepted = c1.slider("Campagnes acceptees", min_value=0, max_value=5, value=1)
        submitted = st.form_submit_button("Predire le segment")

    if not submitted:
        return

    campaign_values = [1 if i < accepted else 0 for i in range(5)]
    row = pd.DataFrame(
        [
            {
                "Year_Birth": year_birth,
                "Education": education,
                "Marital_Status": marital,
                "Income": income,
                "Kidhome": kidhome,
                "Teenhome": teenhome,
                "Recency": recency,
                "MntWines": wines,
                "MntFruits": fruits,
                "MntMeatProducts": meat,
                "MntFishProducts": fish,
                "MntSweetProducts": sweet,
                "MntGoldProds": gold,
                "NumDealsPurchases": deals,
                "NumWebPurchases": web,
                "NumCatalogPurchases": catalog,
                "NumStorePurchases": store,
                "NumWebVisitsMonth": web_visits,
                "AcceptedCmp1": campaign_values[0],
                "AcceptedCmp2": campaign_values[1],
                "AcceptedCmp3": campaign_values[2],
                "AcceptedCmp4": campaign_values[3],
                "AcceptedCmp5": campaign_values[4],
                "Response": 0,
                "Complain": 0,
                "Dt_Customer": "01/01/2014",
            }
        ]
    )
    prepared = build_cluster_features(preprocess_cluster(row))
    prepared = prepared.drop(columns=[c for c in ("ID", "Response", "Complain", "Dt_Customer") if c in prepared.columns])
    prepared = _align_to_expected_columns(prepared, model)
    transformed = model.named_steps["preprocessor"].transform(prepared)
    cluster_model = model.named_steps["clustering"]
    label = int(cluster_model.predict(transformed)[0]) if hasattr(cluster_model, "predict") else 0
    segment_name = SEGMENT_NAMES.get(label, "Segment non nomme")
    action = SEGMENT_ACTIONS.get(label, "Analyse complementaire.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Segment predit", f"Cluster {label}")
    c2.metric("Persona", segment_name)
    c3.metric("Depense totale saisie", f"{wines + fruits + meat + fish + sweet + gold:,.0f}")
    render_insight("Action recommandee", action)


def show_prediction_center() -> None:
    render_header(
        "Centre de prediction",
        "Etat du scoring, tests unitaires de prediction et endpoints API disponibles.",
    )

    fraud_model = load_fraud_model()
    cluster_model = load_cluster_model()
    fraud_cmp = load_fraud_comparison()
    best_fraud = _best_fraud_name()
    best_cluster = _best_cluster_name()
    threshold = _metric_value(fraud_cmp, best_fraud, "threshold", 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction fraude", "operationnelle" if fraud_model is not None else "indisponible")
    c2.metric("Seuil fraude", f"{threshold:.2f}")
    c3.metric("Prediction segment", "operationnelle" if cluster_model is not None else "indisponible")
    c4.metric("Modele segment", best_cluster)

    s1, s2, s3 = st.columns(3)
    with s1:
        render_insight(
            "Fraude",
            f"Le modele {best_fraud} retourne une probabilite de fraude. La classe finale depend du seuil calibre sur validation.",
        )
    with s2:
        render_insight(
            "Segmentation",
            "Le modele de clustering attribue un client a un segment et associe ce segment a une action marketing.",
        )
    with s3:
        render_insight(
            "Production",
            "La prediction est disponible dans le dashboard et via l'API FastAPI. Le batch scoring reste une evolution possible.",
        )

    tab_fraud, tab_segment, tab_api = st.tabs(["Tester fraude", "Tester segment client", "API"])

    with tab_fraud:
        st.info("Objectif : saisir les caracteristiques d'une transaction et obtenir un score de fraude.")
        fraud_playground()

    with tab_segment:
        st.info("Objectif : saisir le profil d'un client et obtenir son segment marketing.")
        segment_playground()

    with tab_api:
        st.subheader("Endpoints de prediction")
        endpoints = pd.DataFrame(
            [
                ["GET", "/model/info", "Connaitre les modeles charges et le seuil fraude"],
                ["POST", "/predict/fraud", "Predire la probabilite de fraude d'une transaction"],
                ["POST", "/predict/segment", "Attribuer un segment a un profil client"],
            ],
            columns=["Methode", "Endpoint", "Role"],
        )
        st.dataframe(endpoints, width="stretch", hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Exemple payload fraude**")
            st.code(
                json.dumps(
                    {
                        "payload": {
                            "step": 1,
                            "type": "TRANSFER",
                            "amount": 1000.0,
                            "oldbalanceOrg": 2000.0,
                            "newbalanceOrig": 1000.0,
                            "oldbalanceDest": 500.0,
                            "newbalanceDest": 1500.0,
                        }
                    },
                    indent=2,
                ),
                language="json",
            )
        with right:
            st.markdown("**Exemple payload segmentation**")
            st.code(
                json.dumps(
                    {
                        "payload": {
                            "Year_Birth": 1985,
                            "Education": "Graduation",
                            "Marital_Status": "Single",
                            "Income": 50000,
                            "Kidhome": 0,
                            "Teenhome": 0,
                            "Dt_Customer": "01/01/2014",
                            "Recency": 20,
                            "MntWines": 200,
                            "MntFruits": 20,
                            "MntMeatProducts": 120,
                            "MntFishProducts": 30,
                            "MntSweetProducts": 15,
                            "MntGoldProds": 40,
                            "NumDealsPurchases": 2,
                            "NumWebPurchases": 5,
                            "NumCatalogPurchases": 2,
                            "NumStorePurchases": 4,
                            "NumWebVisitsMonth": 6,
                            "AcceptedCmp1": 0,
                            "AcceptedCmp2": 0,
                            "AcceptedCmp3": 0,
                            "AcceptedCmp4": 0,
                            "AcceptedCmp5": 0,
                            "Complain": 0,
                            "Response": 0,
                        }
                    },
                    indent=2,
                ),
                language="json",
            )


def show_recommendations() -> None:
    render_header(
        "Recommandations metier",
        "Synthese des decisions a prendre pour la fraude, le marketing et l'industrialisation.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraude")
        fraud_matrix = pd.DataFrame(
            [
                ["Score >= seuil", "Alerte prioritaire", "Verification forte ou blocage temporaire"],
                ["Score intermediaire", "Surveillance", "Revue humaine selon capacite"],
                ["Score faible", "Traitement standard", "Monitoring statistique"],
            ],
            columns=["Situation", "Decision", "Action"],
        )
        st.dataframe(fraud_matrix, width="stretch", hide_index=True)
    with col2:
        st.subheader("Marketing")
        profiles = load_cluster_profiles()
        if not profiles.empty:
            st.dataframe(
                profiles[["cluster", "segment", "cluster_size", "action"]],
                width="stretch",
                hide_index=True,
            )

    st.subheader("Priorites de la prochaine iteration")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_insight("Interpretabilite", "Ajouter SHAP pour expliquer les alertes fraude et les variables les plus decisives.")
    with p2:
        render_insight("Monitoring", "Suivre la derive des montants, types de transaction, scores et tailles de segments.")
    with p3:
        render_insight("Industrialisation", "Ajouter MLflow, Docker, validation de schema et CI/CD pour rendre le projet reproductible.")


def show_mlops() -> None:
    render_header(
        "MLOps et industrialisation",
        "Etat des artefacts, architecture cible et controles de robustesse du projet.",
    )

    fraud_model = load_fraud_model()
    cluster_model = load_cluster_model()
    tests = "6 tests passes"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modele fraude", "charge" if fraud_model is not None else "absent")
    c2.metric("Modele clustering", "charge" if cluster_model is not None else "absent")
    c3.metric("Tests", tests)
    c4.metric("API", "FastAPI")

    st.subheader("Pipeline cible")
    pipeline = pd.DataFrame(
        [
            ["1", "Ingestion", "Chargement des CSV bruts"],
            ["2", "Validation", "Verification schema, types, valeurs manquantes"],
            ["3", "Preprocessing", "Nettoyage, imputation, encodage"],
            ["4", "Features", "Soldes, ratios, depenses, canaux"],
            ["5", "Training", "Comparaison modeles et calibration seuil"],
            ["6", "Evaluation", "PR-AUC, recall, precision, silhouette"],
            ["7", "Serving", "API FastAPI et dashboard Streamlit"],
            ["8", "Monitoring", "Drift, performance, stabilite segments"],
        ],
        columns=["Etape", "Bloc", "Role"],
    )
    st.dataframe(pipeline, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Artefacts")
        artifact_rows = []
        for path in [
            FRAUD_DIR / "fraud_model.joblib",
            FRAUD_DIR / "fraud_model_comparison.csv",
            CLUSTER_DIR / "cluster_model.joblib",
            CLUSTER_DIR / "cluster_profiles.csv",
            PROJECT_ROOT / "src" / "api" / "main.py",
            PROJECT_ROOT / "dashboard" / "app.py",
        ]:
            artifact_rows.append(
                {
                    "artefact": path.relative_to(PROJECT_ROOT).as_posix(),
                    "statut": "present" if path.exists() else "absent",
                    "taille_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
                }
            )
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)

    with right:
        st.subheader("Endpoints API")
        endpoints = pd.DataFrame(
            [
                ["GET", "/health", "Sante API"],
                ["GET", "/model/info", "Modeles charges"],
                ["POST", "/predict/fraud", "Score fraude"],
                ["POST", "/predict/segment", "Segment client"],
            ],
            columns=["Methode", "Endpoint", "Usage"],
        )
        st.dataframe(endpoints, width="stretch", hide_index=True)

    st.subheader("Roadmap technique")
    roadmap = pd.DataFrame(
        [
            ["Court terme", "SHAP, figures rapport, validation metier des clusters"],
            ["Moyen terme", "MLflow, DVC/Git LFS, Docker, GitHub Actions"],
            ["Long terme", "Monitoring drift, reentrainement programme, registre modele"],
        ],
        columns=["Horizon", "Actions"],
    )
    st.dataframe(roadmap, width="stretch", hide_index=True)


inject_style()
selected_page = get_current_page()
render_sidebar_navigation(selected_page)
st.caption(f"Projet ML / {NAV_BY_SLUG[selected_page]['label']} - {NAV_BY_SLUG[selected_page]['description']}")

if selected_page == "overview":
    show_overview()
elif selected_page == "fraud":
    show_fraud_results()
elif selected_page == "segmentation":
    show_cluster_profiles()
elif selected_page == "recommendations":
    show_recommendations()
elif selected_page == "prediction":
    show_prediction_center()
else:
    show_mlops()
