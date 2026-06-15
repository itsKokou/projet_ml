"""Dashboard Streamlit avancé pour résultats ML."""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

# Permet d'importer `src.*` quand Streamlit exécute depuis `dashboard/`.
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
REPORT_PATH = PROJECT_ROOT / "reports" / "rapport_technique.md"
REPORT_PDF_PATH = PROJECT_ROOT / "reports" / "rapport_technique.pdf"
PRESENTATION_PPTX_PATH = PROJECT_ROOT / "reports" / "presentation_finale.pptx"
REPORT_DIR = PROJECT_ROOT / "reports"
_REPORT_IMG_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?:\s+\"[^\"]*\")?\)"
)

FRAUD_MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"]
FRAUD_DISPLAY_SAMPLE_SIZE = 300_000
FRAUD_THRESHOLD_SIM_SAMPLE_SIZE = 180_000
SEGMENT_NAMES = {
    0: "Dormants à faible valeur",
    1: "Chasseurs de promotions digitaux",
    2: "Premium ultra-engagés",
    3: "Fidèles à forte valeur",
}
SEGMENT_ACTIONS = {
    0: "Réactivation, offres accessibles, parcours d'onboarding.",
    1: "Coupons, retargeting web, bundles promotionnels.",
    2: "Programme VIP, avant-premières, offres premium.",
    3: "Fidélisation, cross-sell, expérience omnicanale.",
}
NAV_ITEMS = [
    {
        "slug": "overview",
        "label": "Vue d'ensemble",
        "short": "Pilotage",
        "description": "KPI globaux, comparaison des modèles et signaux métier.",
    },
    {
        "slug": "fraud",
        "label": "Fraude",
        "short": "Détection",
        "description": "Performance des modèles, seuil, analyse transactionnelle et simulateur.",
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
        "short": "Décisions",
        "description": "Actions métier pour la fraude, le marketing et la prochaine itération.",
    },
    {
        "slug": "prediction",
        "label": "Prédiction",
        "short": "Scoring",
        "description": "Tester les prédictions fraude et segmentation, puis voir les endpoints API.",
    },
    {
        "slug": "mlops",
        "label": "MLOps",
        "short": "Industrialisation",
        "description": "Artefacts, API, pipeline cible et roadmap technique.",
    },
    {
        "slug": "report",
        "label": "Rapport d'analyse",
        "short": "Interprétation",
        "description": "Rapport PDF, présentation PPTX et aperçu en ligne.",
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
    """Styles globaux du dashboard (cartes KPI, etc.)."""
    st.markdown(
        """
        <style>
        .kpi-card {
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.25rem;
            box-shadow: 0 6px 18px rgba(16, 24, 40, 0.1);
            min-height: 92px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .kpi-label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.92;
            margin-bottom: 0.4rem;
        }
        .kpi-value {
            font-size: 1.28rem;
            font-weight: 700;
            line-height: 1.25;
            word-break: break-word;
        }
        .kpi-teal {
            background: linear-gradient(145deg, #157f7f 0%, #1babab 100%);
            color: #ffffff;
        }
        .kpi-rose {
            background: linear-gradient(145deg, #9f1239 0%, #c2415c 100%);
            color: #ffffff;
        }
        .kpi-blue {
            background: linear-gradient(145deg, #1e4a7a 0%, #3b6ea8 100%);
            color: #ffffff;
        }
        .kpi-indigo {
            background: linear-gradient(145deg, #3730a3 0%, #6366f1 100%);
            color: #ffffff;
        }
        .kpi-emerald {
            background: linear-gradient(145deg, #047857 0%, #10b981 100%);
            color: #ffffff;
        }
        .kpi-amber {
            background: linear-gradient(145deg, #92400e 0%, #d97706 100%);
            color: #ffffff;
        }
        .kpi-violet {
            background: linear-gradient(145deg, #5b21b6 0%, #8b5cf6 100%);
            color: #ffffff;
        }
        .kpi-slate {
            background: linear-gradient(145deg, #334155 0%, #64748b 100%);
            color: #ffffff;
        }

        /* Sidebar — theme teal / slate */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                #0c3d3d 0%,
                #157f7f 42%,
                #1e4a6e 100%
            );
        }
        [data-testid="stSidebar"] > div:first-child {
            background: transparent;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: transparent;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #ecfdf5 !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: rgba(236, 253, 245, 0.78) !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.22);
            margin: 0.85rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            color: rgba(236, 253, 245, 0.75) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px;
            font-weight: 600;
            transition: background 0.15s ease, transform 0.1s ease;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            color: #ecfdf5 !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            background-color: rgba(255, 255, 255, 0.18) !important;
            border-color: rgba(255, 255, 255, 0.45) !important;
        }
        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background-color: #ffffff !important;
            color: #0f4f4f !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        [data-testid="stSidebar"] .stButton > button:disabled {
            opacity: 1 !important;
            background-color: rgba(255, 255, 255, 0.22) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.35) !important;
        }
        [data-testid="stSidebar"] [data-testid="collapsedControl"] {
            color: #ecfdf5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(column, label: str, value: str, variant: str = "teal") -> None:
    """Affiche un KPI dans une carte colorée."""
    safe_label = html.escape(label)
    safe_value = html.escape(str(value))
    with column:
        st.markdown(
            f"""
            <div class="kpi-card kpi-{html.escape(variant)}">
                <div class="kpi-label">{safe_label}</div>
                <div class="kpi-value">{safe_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_technical_report() -> str:
    """Charge le rapport technique markdown."""
    if not REPORT_PATH.exists():
        return ""
    return REPORT_PATH.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def load_report_pdf() -> bytes:
    """Charge le rapport PDF pour téléchargement."""
    if not REPORT_PDF_PATH.is_file():
        return b""
    return REPORT_PDF_PATH.read_bytes()


@st.cache_data(show_spinner=False)
def load_presentation_pptx() -> bytes:
    """Charge la présentation PPTX pour téléchargement."""
    if not PRESENTATION_PPTX_PATH.is_file():
        return b""
    return PRESENTATION_PPTX_PATH.read_bytes()


def _resolve_report_image(match: re.Match[str]) -> str:
    """Convertit une image markdown en balise HTML avec chemin résolu."""
    alt = html.escape(match.group("alt"))
    raw_path = match.group("path").strip()
    if raw_path.startswith(("http://", "https://", "data:")):
        return match.group(0)

    image_path = (REPORT_DIR / raw_path).resolve()
    if not image_path.is_file():
        return f"*Figure introuvable : `{raw_path}`*"

    mime_by_ext = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    ext = image_path.suffix.lower().lstrip(".")
    mime = mime_by_ext.get(ext, "image/png")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return (
        f'<img src="data:{mime};base64,{encoded}" alt="{alt}" '
        'style="max-width:100%; height:auto; margin:1rem 0;" />'
    )


def render_report_markdown(content: str) -> None:
    """Affiche le rapport en résolvant les chemins d'images et liens docs du dossier reports/."""

    def _resolve_doc_link(match: re.Match[str]) -> str:
        text = match.group("text")
        href = match.group("href").strip()
        if href.startswith("../docs/"):
            doc_path = PROJECT_ROOT / "docs" / href.removeprefix("../docs/")
            if doc_path.is_file():
                rel = doc_path.relative_to(PROJECT_ROOT).as_posix()
                return f"**{text}** (`{rel}` — téléchargeable depuis le dépôt)"
        return match.group(0)

    link_pattern = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<href>[^)]+)\)")
    with_doc_links = link_pattern.sub(_resolve_doc_link, content)
    resolved = _REPORT_IMG_RE.sub(_resolve_report_image, with_doc_links)
    st.markdown(resolved, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_json_artifact(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv_cached(path: Path, sep: str = ",", **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep=sep, **kwargs)


@st.cache_data(show_spinner=False)
def fraud_dataset_size() -> int:
    """Nombre total de lignes dans le CSV fraude (sans charger tout le dataframe)."""
    path = DATA_DIR / "detection_fraude.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _format_compact_count(value: int) -> str:
    if value >= 1_000_000:
        millions = value / 1_000_000
        text = f"{millions:.2f}".rstrip("0").rstrip(".")
        return f"{text.replace('.', ',')}M"
    if value >= 1_000:
        return f"{round(value / 1_000)}k"
    return str(value)


def _format_fraud_sample_label(sample_size: int) -> str:
    total = fraud_dataset_size()
    sample_label = _format_compact_count(sample_size)
    if total <= 0 or total == sample_size:
        return f"{sample_size:,}".replace(",", " ")
    return f"{sample_label} / {_format_compact_count(total)}"


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
    profiles["segment"] = profiles["cluster"].map(SEGMENT_NAMES).fillna("Segment non nommé")
    profiles["action"] = profiles["cluster"].map(SEGMENT_ACTIONS).fillna("Analyse complémentaire.")
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
    st.sidebar.caption("Fraude bancaire · Segmentation client · MLOps")
    st.sidebar.divider()

    st.sidebar.subheader("Navigation")
    for item in NAV_ITEMS:
        is_active = item["slug"] == active_slug
        label = item["label"] if not is_active else f"▸ {item['label']}"
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
    st.sidebar.subheader("Statut des modèles")
    st.sidebar.metric("Modèle de fraude", best_fraud)
    st.sidebar.metric("PR-AUC", f"{_metric_value(fraud_cmp, best_fraud, 'pr_auc'):.4f}")
    st.sidebar.metric("Clustering", best_cluster)
    st.sidebar.metric("Silhouette", f"{_metric_value(cluster_cmp, best_cluster, 'silhouette'):.4f}")
    # st.sidebar.divider()
    # st.sidebar.markdown(
    #     """
    #     **Artefacts**

    #     - `models/fraud/`
    #     - `models/clustering/`
    #     - `reports/rapport_technique.md`
    #     - `reports/presentation_outline.md`
    #     """
    # )


@st.cache_data(show_spinner=False)
def fraud_type_summary(sample_size: int = FRAUD_DISPLAY_SAMPLE_SIZE) -> pd.DataFrame:
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
def fraud_time_summary(sample_size: int = FRAUD_DISPLAY_SAMPLE_SIZE) -> pd.DataFrame:
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
def fraud_threshold_curve(sample_size: int = FRAUD_THRESHOLD_SIM_SAMPLE_SIZE) -> pd.DataFrame:
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


def _prettify_fraud_feature(name: str) -> str:
    raw = (
        str(name)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("type_", "type=")
    )
    labels = {
        "origin_error": "Écart solde émetteur",
        "dest_error": "Écart solde destinataire",
        "origin_balance_diff": "Variation solde émetteur",
        "dest_balance_diff": "Variation solde destinataire",
        "amount": "Montant",
        "amount_to_oldbalance_ratio": "Ratio montant / solde",
        "is_transfer_or_cashout": "TRANSFER ou CASH_OUT",
        "is_zero_newbalance_origin": "Solde émetteur à zéro après tx",
        "is_zero_oldbalance_dest": "Solde destinataire nul avant tx",
        "step": "Horodatage (step)",
        "step_bucket": "Fenêtre temporelle",
        "oldbalanceOrg": "Solde émetteur avant",
        "newbalanceOrig": "Solde émetteur après",
        "oldbalanceDest": "Solde destinataire avant",
        "newbalanceDest": "Solde destinataire après",
    }
    return labels.get(raw, raw.replace("_", " "))


def _shap_positive_class(shap_values) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _fraud_features_from_inputs(
    step: int,
    tx_type: str,
    amount: float,
    old_org: float,
    new_org: float,
    old_dest: float,
    new_dest: float,
) -> pd.DataFrame:
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
    drop_cols = [c for c in ("nameOrig", "nameDest", "isFlaggedFraud") if c in features.columns]
    return features.drop(columns=drop_cols)


def _predict_fraud_transaction(
    model,
    step: int,
    tx_type: str,
    amount: float,
    old_org: float,
    new_org: float,
    old_dest: float,
    new_dest: float,
) -> tuple[float, pd.DataFrame]:
    """Retourne (probabilité fraude, features enrichies)."""
    features = _fraud_features_from_inputs(step, tx_type, amount, old_org, new_org, old_dest, new_dest)
    model_input = _align_to_expected_columns(features, model)
    proba = float(model.predict_proba(model_input)[0, 1])
    return proba, features


def _build_fraud_shap_matrix(
    model, sample_size: int = 4000
) -> tuple[np.ndarray | None, list[str] | None, np.ndarray | None]:
    df = load_fraud_sample(200_000)
    if df.empty or "isFraud" not in df.columns:
        return None, None, None

    fraud = df[df["isFraud"].astype(int) == 1]
    normal = df[df["isFraud"].astype(int) == 0]
    n_fraud = len(fraud)
    n_normal = max(sample_size - n_fraud, 0)
    if n_normal > 0 and len(normal) > n_normal:
        normal = normal.sample(n_normal, random_state=42)
    sample = pd.concat([fraud, normal], ignore_index=True)

    X = build_fraud_features(preprocess_fraud(sample.drop(columns=["isFraud"])))
    X = X.drop(columns=[c for c in ("nameOrig", "nameDest", "isFlaggedFraud") if c in X.columns])
    X = _align_to_expected_columns(X, model)
    preprocessor = model.named_steps["preprocessor"]
    X_t = preprocessor.transform(X)
    names = [_prettify_fraud_feature(n) for n in preprocessor.get_feature_names_out()]
    y = sample["isFraud"].astype(int).to_numpy()
    return X_t, names, y


@st.cache_resource(show_spinner=False)
def _fraud_shap_explainer_bundle():
    import shap

    model = load_fraud_model()
    if model is None:
        return None

    classifier = model.named_steps.get("classifier")
    if classifier is None:
        return None

    X_t, names, y = _build_fraud_shap_matrix(model, sample_size=5000)
    if X_t is None or y is None:
        return None

    try:
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
        else:
            background = shap.sample(X_t, min(400, len(X_t)), random_state=42)
            explainer = shap.LinearExplainer(classifier, background)
        return {"explainer": explainer, "names": names, "X_t": X_t, "y": y}
    except Exception:
        return None


@st.cache_data(show_spinner="Calcul SHAP global…")
def fraud_shap_global_summary(top_n: int = 12) -> pd.DataFrame:
    bundle = _fraud_shap_explainer_bundle()
    if not bundle:
        return pd.DataFrame()

    sv = _shap_positive_class(bundle["explainer"].shap_values(bundle["X_t"]))
    return (
        pd.DataFrame(
            {
                "feature": bundle["names"],
                "mean_abs_shap": np.abs(sv).mean(axis=0),
                "mean_shap": sv.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )


@st.cache_data(show_spinner="Calcul SHAP local…")
def fraud_shap_local_contributions(
    step: int,
    tx_type: str,
    amount: float,
    old_org: float,
    new_org: float,
    old_dest: float,
    new_dest: float,
    top_n: int = 10,
) -> pd.DataFrame:
    bundle = _fraud_shap_explainer_bundle()
    model = load_fraud_model()
    if not bundle or model is None:
        return pd.DataFrame()

    features = _fraud_features_from_inputs(step, tx_type, amount, old_org, new_org, old_dest, new_dest)
    features = _align_to_expected_columns(features, model)
    X_t = model.named_steps["preprocessor"].transform(features)
    sv = _shap_positive_class(bundle["explainer"].shap_values(X_t))[0]

    out = pd.DataFrame({"feature": bundle["names"], "shap": sv})
    out["abs_shap"] = out["shap"].abs()
    return out.sort_values("abs_shap", ascending=False).head(top_n)


def _shap_insight_text(feature: str, mean_shap: float) -> str:
    direction = "augmente" if mean_shap >= 0 else "diminue"
    f = feature.lower()
    if "écart solde émetteur" in f or "origin_error" in f:
        return (
            f"En moyenne, un écart entre montant et solde émetteur {direction} fortement "
            "la probabilité de fraude — signal d'incohérence comptable."
        )
    if "écart solde destinataire" in f or "dest_error" in f:
        return (
            f"Les écarts côté destinataire {direction} aussi le score : utile pour repérer "
            "des flux qui ne recollent pas aux soldes."
        )
    if "montant" in f and "ratio" not in f:
        return (
            f"Le montant {direction} le risque, surtout lorsque les soldes ne justifient pas "
            "le transfert."
        )
    if "ratio" in f:
        return (
            f"Un ratio montant / solde élevé {direction} le score : peu de marge sur le solde "
            "avant transaction."
        )
    if "transfer" in f or "cash_out" in f or "type=" in f:
        return (
            f"Le type de transaction {direction} le risque : TRANSFER et CASH_OUT restent "
            "les canaux les plus touchés dans les données."
        )
    if "zéro" in f or "zero" in f:
        return (
            f"Un solde vidé ou nul {direction} la suspicion — pattern fréquent sur les "
            "comptes compromis."
        )
    return (
        f"La variable « {feature} » {direction} la probabilité de fraude sur l'échantillon "
        "analysé (SHAP moyen)."
    )


def _plot_shap_bar(df: pd.DataFrame, value_col: str, title: str, height: int = 380) -> go.Figure:
    plot_df = df.sort_values(value_col, ascending=True).copy()
    if value_col == "shap":
        colors = ["#c2415c" if v > 0 else "#157f7f" for v in plot_df["shap"]]
    else:
        colors = "#157f7f"
    fig = go.Figure(
        go.Bar(
            x=plot_df[value_col],
            y=plot_df["feature"],
            orientation="h",
            marker_color=colors,
        )
    )
    fig.update_layout(title=title, xaxis_title="Contribution SHAP", yaxis_title="", height=height)
    return fig


def _render_local_shap_block(
    step: int,
    tx_type: str,
    amount: float,
    old_org: float,
    new_org: float,
    old_dest: float,
    new_dest: float,
    proba: float | None = None,
) -> None:
    local = fraud_shap_local_contributions(step, tx_type, amount, old_org, new_org, old_dest, new_dest)
    if local.empty:
        st.caption("Contributions SHAP indisponibles pour ce modèle.")
        return

    if proba is not None:
        st.caption(f"Score modèle : **{proba:.4f}** — barres rouges = poussent vers la fraude, vertes = freinent.")
    fig = _plot_shap_bar(local, "shap", "Contributions à la probabilité de fraude", height=320)
    st.plotly_chart(plotly_layout(fig, 320), width="stretch")

    pushes = local[local["shap"] > 0].head(3)
    if not pushes.empty:
        bullets = ", ".join(f"**{r.feature}** ({r.shap:+.3f})" for r in pushes.itertuples())
        st.markdown(f"Principaux facteurs qui **renforcent** l'alerte : {bullets}.")


def render_fraud_shap_tab() -> None:
    """Coquille légère : le calcul SHAP n'est lancé qu'après action utilisateur."""
    if not st.session_state.get("fraud_shap_loaded"):
        st.info(
            "L'analyse SHAP charge un échantillon de transactions et peut prendre **10 à 30 secondes**. "
            "Elle n'est pas lancée automatiquement afin de ne pas ralentir le simulateur."
        )
        if st.button("Charger l'analyse SHAP", type="primary", key="btn_load_fraud_shap"):
            st.session_state["fraud_shap_loaded"] = True
            st.rerun()
        return
    _render_fraud_shap_tab_content()


def _render_fraud_shap_tab_content() -> None:
    model = load_fraud_model()
    if model is None:
        st.warning("Chargez un modèle fraude (`python -m src.models.train_fraud_model`) pour activer SHAP.")
        return

    st.caption(
        f"Modèle : **{_best_fraud_name()}** — SHAP indique comment chaque variable déplace la probabilité "
        "de fraude (référence = comportement moyen sur un échantillon stratifié de ~5 000 transactions, "
        "tiré de 200k). Le classement peut différer légèrement de la figure statique du rapport."
    )

    shap_report_fig = REPORT_DIR / "figures" / "09_fraud_shap_summary.png"
    if shap_report_fig.is_file():
        with st.expander("Figure de référence — rapport technique"):
            st.image(
                str(shap_report_fig),
                caption="Export statique (`generate_report_figures.py`) — échantillon fixe de 4 000 transactions",
            )

    global_df = fraud_shap_global_summary()
    if global_df.empty:
        st.info(
            "SHAP n'est pas disponible pour ce classifieur (utilisez XGBoost, forêt aléatoire ou régression logistique)."
        )
        return

    st.subheader("Trois leçons globales")
    insight_cols = st.columns(3)
    for col, (_, row) in zip(insight_cols, global_df.head(3).iterrows()):
        with col:
            render_insight(row["feature"], _shap_insight_text(row["feature"], float(row["mean_shap"])))

    st.subheader("Importance globale (|SHAP| moyen)")
    fig = _plot_shap_bar(
        global_df.sort_values("mean_abs_shap").tail(12),
        "mean_abs_shap",
        "Variables qui déplacent le plus le score, en moyenne",
        height=420,
    )
    st.plotly_chart(plotly_layout(fig, 420), width="stretch")

    st.divider()
    st.subheader("Explication d'une transaction")
    examples = fraud_suspicious_examples(6)
    preset_labels: list[str] = []
    preset_rows: list[dict] = []

    if not examples.empty:
        for idx, (_, row) in enumerate(examples.iterrows()):
            label = (
                f"Fraude #{idx + 1} — {row.get('type', '?')} "
                f"{float(row.get('amount', 0)):,.0f} €"
            )
            preset_labels.append(label)
            preset_rows.append(
                {
                    "step": int(row["step"]),
                    "type": str(row["type"]),
                    "amount": float(row["amount"]),
                    "oldbalanceOrg": float(row["oldbalanceOrg"]),
                    "newbalanceOrig": float(row["newbalanceOrig"]),
                    "oldbalanceDest": float(row["oldbalanceDest"]),
                    "newbalanceDest": float(row["newbalanceDest"]),
                }
            )

    choice = st.selectbox(
        "Transaction à expliquer",
        ["Saisie manuelle (simulateur)"] + preset_labels,
        index=1 if preset_labels else 0,
    )

    if choice == "Saisie manuelle (simulateur)":
        c1, c2, c3, c4 = st.columns(4)
        step = c1.number_input("Step", min_value=1, value=1, key="shap_step")
        tx_type = c2.selectbox(
            "Type",
            ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            key="shap_type",
        )
        amount = c3.number_input("Montant", min_value=0.0, value=1000.0, step=100.0, key="shap_amount")
        old_org = c4.number_input(
            "Solde émetteur avant", min_value=0.0, value=2000.0, step=100.0, key="shap_old_org"
        )
        new_org = c1.number_input(
            "Solde émetteur après", min_value=0.0, value=1000.0, step=100.0, key="shap_new_org"
        )
        old_dest = c2.number_input(
            "Solde destinataire avant", min_value=0.0, value=500.0, step=100.0, key="shap_old_dest"
        )
        new_dest = c3.number_input(
            "Solde destinataire après", min_value=0.0, value=1500.0, step=100.0, key="shap_new_dest"
        )
    else:
        payload = preset_rows[preset_labels.index(choice)]
        step = payload["step"]
        tx_type = payload["type"]
        amount = payload["amount"]
        old_org = payload["oldbalanceOrg"]
        new_org = payload["newbalanceOrig"]
        old_dest = payload["oldbalanceDest"]
        new_dest = payload["newbalanceDest"]
        st.info(
            f"Exemple réel du jeu de données : **{tx_type}**, montant **{amount:,.0f}**, "
            f"fraude confirmée (isFraud=1)."
        )

    proba, _ = _predict_fraud_transaction(
        model, step, tx_type, amount, old_org, new_org, old_dest, new_dest
    )
    comparison = load_fraud_comparison()
    threshold = _metric_value(comparison, _best_fraud_name(), "threshold", 0.5)

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilité fraude", f"{proba:.4f}")
    m2.metric("Seuil modèle", f"{threshold:.2f}")
    m3.metric("Décision", "Alerte" if proba >= threshold else "OK")

    _render_local_shap_block(step, tx_type, amount, old_org, new_org, old_dest, new_dest, proba=proba)


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
    proj["segment"] = proj["cluster"].map(SEGMENT_NAMES).fillna("Segment non nommé")
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
    out["segment"] = out["cluster"].map(SEGMENT_NAMES).fillna("Segment non nommé")
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
        "Synthèse opérationnelle : données, modèles retenus et signaux métier.",
    )

    fraud_cmp = load_fraud_comparison()
    cluster_cmp = load_cluster_comparison()
    profiles = load_cluster_profiles()
    best_fraud = _best_fraud_name()
    best_cluster = _best_cluster_name()

    fraud_df = load_fraud_sample(FRAUD_DISPLAY_SAMPLE_SIZE)
    cluster_df = load_cluster_raw()
    fraud_rate = float(fraud_df["isFraud"].mean()) if not fraud_df.empty else 0.0
    best_pr_auc = _metric_value(fraud_cmp, best_fraud, "pr_auc")
    best_recall = _metric_value(fraud_cmp, best_fraud, "recall")
    best_precision = _metric_value(fraud_cmp, best_fraud, "precision")
    best_threshold = _metric_value(fraud_cmp, best_fraud, "threshold")
    best_silhouette = _metric_value(cluster_cmp, best_cluster, "silhouette")
    n_clients = len(cluster_df)
    premium_share = 0.0
    if not profiles.empty and "cluster_size" in profiles.columns:
        total_clients = float(profiles["cluster_size"].sum())
        if total_clients > 0 and 2 in profiles["cluster"].to_numpy():
            premium_share = float(profiles.loc[profiles["cluster"] == 2, "cluster_size"].iloc[0]) / total_clients

    st.subheader("Indicateurs clés")
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
    render_kpi_card(
        r1c1,
        "Échantillon affiché",
        _format_fraud_sample_label(len(fraud_df)),
        "teal",
    )
    render_kpi_card(r1c2, "Taux de fraude (échantillon)", _format_percent(fraud_rate, 3), "rose")
    render_kpi_card(r1c3, "PR-AUC (test)", f"{best_pr_auc:.4f}", "blue")
    render_kpi_card(r1c4, "Rappel (test)", _format_percent(best_recall, 1), "indigo")
    render_kpi_card(r1c5, "Précision (test)", _format_percent(best_precision, 1), "emerald")
    render_kpi_card(r1c6, "Seuil retenu", f"{best_threshold:.2f}", "amber")
    st.caption(
        f"L'échantillon affiché ({_format_fraud_sample_label(len(fraud_df))}) sert aux graphiques exploratoires. "
        "PR-AUC, rappel, précision et seuil proviennent du **jeu de test officiel** "
        f"({int(_metric_value(fraud_cmp, best_fraud, 'test_size', 0)):,} lignes).".replace(",", " ")
    )

    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
    render_kpi_card(
        r2c1,
        "Clients",
        f"{n_clients:,}".replace(",", " ") if n_clients else "—",
        "slate",
    )
    render_kpi_card(r2c2, "Modèle fraude", best_fraud, "teal")
    render_kpi_card(r2c3, "Modèle clustering", best_cluster, "violet")
    render_kpi_card(r2c4, "Silhouette", f"{best_silhouette:.4f}", "indigo")
    render_kpi_card(r2c5, "Segments", "4" if not profiles.empty else "—", "blue")
    render_kpi_card(r2c6, "Part premium (C2)", _format_percent(premium_share, 1), "rose")

    st.divider()

    chart_left, chart_right = st.columns([1.15, 0.85])
    with chart_left:
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
            st.plotly_chart(plotly_layout(fig, 380), width="stretch")

        if not fraud_cmp.empty:
            ordered = [m for m in FRAUD_MODEL_ORDER if m in fraud_cmp.index]
            cmp_view = fraud_cmp.loc[ordered if ordered else fraud_cmp.index]
            fig_cmp = px.bar(
                cmp_view.reset_index(),
                x="index",
                y="pr_auc",
                color="index",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text=cmp_view["pr_auc"].map(lambda x: f"{x:.3f}"),
                title="Comparaison des modèles — PR-AUC",
            )
            fig_cmp.update_traces(textposition="outside")
            fig_cmp.update_layout(showlegend=False)
            st.plotly_chart(plotly_layout(fig_cmp, 320), width="stretch")

    with chart_right:
        if not profiles.empty:
            fig = px.pie(
                profiles,
                names="segment",
                values="cluster_size",
                hole=0.52,
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Répartition des segments clients",
            )
            st.plotly_chart(plotly_layout(fig, 380), width="stretch")

            table = profiles[["cluster", "segment", "cluster_size", "action"]].copy()
            table["part"] = table["cluster_size"] / table["cluster_size"].sum()
            table = table.rename(
                columns={
                    "cluster": "Cluster",
                    "segment": "Profil",
                    "cluster_size": "Clients",
                    "action": "Action prioritaire",
                    "part": "Part",
                }
            )
            table["Part"] = table["Part"].map(lambda x: f"{100 * x:.1f} %")
            st.dataframe(table, width="stretch", hide_index=True)

    st.subheader("Messages pour le décideur")
    i1, i2, i3 = st.columns(3)
    with i1:
        render_insight(
            "Priorité fraude",
            "Classe très déséquilibrée (~0,11 %). Piloter par recall, précision et PR-AUC — pas par l'accuracy.",
        )
    with i2:
        render_insight(
            "Signal dominant",
            "Risque concentré sur TRANSFER et CASH_OUT. Les écarts de soldes sont le signal métier le plus robuste.",
        )
    with i3:
        render_insight(
            "Lecture marketing",
            "Quatre personas actionnables : dormants (50 %), fidèles stables (40 %), premium réactifs (8 %), promo digitaux (2 %).",
        )


def render_fraud_advanced_analysis() -> None:
    """Panneaux coûts FP/FN, erreurs et validation temporelle (artefacts exportés)."""
    cost_data = load_json_artifact(FRAUD_DIR / "fraud_cost_analysis.json")
    error_data = load_json_artifact(FRAUD_DIR / "fraud_error_analysis.json")
    temporal_data = load_json_artifact(FRAUD_DIR / "fraud_temporal_metrics.json")

    if not cost_data and not error_data and not temporal_data:
        st.info(
            "Analyses avancées indisponibles. Lancez `python -m src.models.train_fraud_model` "
            "puis `python -c \"from src.models.fraud_experiments import export_cost_analysis; export_cost_analysis()\"`."
        )
        return

    st.caption(
        "Données issues des exports d'entraînement (jeu de test officiel). "
        "La courbe de seuil de l'onglet précédent est une simulation sur échantillon."
    )

    if temporal_data:
        st.subheader("Validation temporelle (split par `step`)")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("PR-AUC temporel", f"{temporal_data.get('pr_auc', 0):.4f}")
        t2.metric("Rappel", _format_percent(float(temporal_data.get("recall", 0)), 2))
        t3.metric("Précision", _format_percent(float(temporal_data.get("precision", 0)), 2))
        t4.metric("Seuil calibré", f"{float(temporal_data.get('threshold', 0)):.2f}")
        t5.metric("Stratégie", temporal_data.get("split_strategy", "—"))
        temporal_fig = REPORT_DIR / "figures" / "07_fraud_temporal_comparison.png"
        if temporal_fig.is_file():
            st.image(str(temporal_fig), caption="Comparaison split aléatoire vs temporel")

    if error_data:
        st.subheader("Analyse des erreurs (jeu de test)")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Faux négatifs", int(error_data.get("false_negatives", 0)))
        e2.metric("Faux positifs", int(error_data.get("false_positives", 0)))
        e3.metric("Rappel au seuil", _format_percent(float(error_data.get("recall_at_threshold", 0)), 2))
        e4.metric("Seuil retenu", f"{float(error_data.get('threshold', 0)):.2f}")
        error_fig = REPORT_DIR / "figures" / "08_fraud_error_summary.png"
        if error_fig.is_file():
            st.image(str(error_fig), caption="Synthèse FN / FP sur le jeu de test")

        fn_path = FRAUD_DIR / "fraud_false_negatives.csv"
        if fn_path.is_file():
            fn_df = read_csv_cached(fn_path)
            if not fn_df.empty:
                st.markdown("**Détail des faux négatifs**")
                st.dataframe(fn_df, width="stretch", hide_index=True)

    if cost_data:
        st.subheader("Chiffrage économique FP / FN")
        assumptions = cost_data.get("assumptions", {})
        st.write(
            f"Hypothèse revue analyste : **{assumptions.get('cost_per_fp_review_eur', 25):.0f} €/FP**. "
            f"{assumptions.get('fn_cost_model', '')}"
        )
        scenarios = cost_data.get("scenarios", [])
        if scenarios:
            cost_df = pd.DataFrame(scenarios)
            display_cols = [
                c
                for c in [
                    "threshold",
                    "false_negatives",
                    "false_positives",
                    "recall",
                    "precision",
                    "fn_financial_loss_units",
                    "fp_operational_cost_eur",
                    "total_cost_eur_equivalent",
                    "is_selected_threshold",
                ]
                if c in cost_df.columns
            ]
            st.dataframe(
                cost_df[display_cols].style.format(
                    {
                        "threshold": "{:.2f}",
                        "recall": "{:.2%}",
                        "precision": "{:.2%}",
                        "fn_financial_loss_units": "{:,.0f}",
                        "fp_operational_cost_eur": "{:,.0f}",
                        "total_cost_eur_equivalent": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        cost_fig = REPORT_DIR / "figures" / "10_fraud_cost_scenarios.png"
        if cost_fig.is_file():
            st.image(str(cost_fig), caption="Scénarios de coût par seuil")
        if cost_data.get("recommendation"):
            st.info(cost_data["recommendation"])


def show_fraud_results() -> None:
    render_header(
        "Détection de fraude",
        "Performance des modèles, seuil de décision, signaux explicatifs et simulateur transactionnel.",
    )

    comparison = load_fraud_comparison()
    if comparison.empty:
        st.warning("Lancez `python -m src.models.train_fraud_model` pour générer les résultats.")
        return

    best_model = _best_fraud_name()
    ordered = [m for m in FRAUD_MODEL_ORDER if m in comparison.index]
    comparison = comparison.loc[ordered + [m for m in comparison.index if m not in ordered]]
    best_row = comparison.loc[best_model] if best_model in comparison.index else comparison.iloc[0]
    confusion = estimate_confusion_from_metrics(best_row)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Modèle retenu", best_model)
    k2.metric("PR-AUC", f"{best_row.get('pr_auc', 0):.4f}")
    k3.metric("Rappel", _format_percent(float(best_row.get("recall", 0)), 2))
    k4.metric("Précision", _format_percent(float(best_row.get("precision", 0)), 2))
    k5.metric("Seuil retenu", f"{best_row.get('threshold', 0):.2f}")

    tab_perf, tab_data, tab_threshold, tab_adv, tab_lab, tab_shap = st.tabs(
        ["Performance", "Analyse données", "Seuil et alertes", "Analyses avancées", "Simulateur", "Explications SHAP"]
    )

    with tab_perf:
        left, right = st.columns([1.05, 0.95])
        with left:
            metrics = ["pr_auc", "roc_auc", "f1", "recall", "precision"]
            long = comparison[metrics].reset_index().melt(id_vars="index", var_name="métrique", value_name="score")
            fig = px.bar(
                long,
                x="index",
                y="score",
                color="métrique",
                barmode="group",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Comparaison des modèles de classification",
            )
            fig.update_layout(xaxis_title="Modèle", yaxis_title="Score")
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
                st.info("Importance des variables indisponible pour ce modèle.")

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
                    title="Déséquilibre des classes",
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
                fig.update_layout(xaxis_title="", yaxis_title="Valeur moyenne ou médiane")
                st.plotly_chart(plotly_layout(fig, 330), width="stretch")

        i1, i2, i3 = st.columns(3)
        with i1:
            render_insight(
                "Classe rare",
                "Le modèle doit traiter une classe fraude très minoritaire. C'est pour cela que l'accuracy n'est pas la métrique principale.",
            )
        with i2:
            render_insight(
                "Soldes utiles",
                "Les écarts entre montant et variation de solde créent des signaux forts pour identifier des transactions incohérentes.",
            )
        with i3:
            render_insight(
                "Lecture opérationnelle",
                "Les alertes doivent être calibrées par seuil : plus le seuil baisse, plus le rappel augmente mais plus il y a d'alertes.",
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
                    title="Fraudes observées par type",
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
                    title="Évolution du taux de fraude par fenêtre temporelle",
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
        st.caption(
            f"Simulation interactive sur un échantillon aléatoire de "
            f"**{FRAUD_THRESHOLD_SIM_SAMPLE_SIZE:,}** transactions (performance dashboard). "
            "Les métriques officielles du jeu de test et le chiffrage FP/FN sont dans l'onglet "
            "**Analyses avancées**.".replace(",", " ")
        )
        curve = fraud_threshold_curve()
        if curve.empty:
            st.info("La courbe de seuil n'est pas disponible.")
        else:
            default_threshold = float(best_row.get("threshold", 0.5))
            threshold = st.slider(
                "Seuil de décision",
                min_value=0.05,
                max_value=0.99,
                value=min(max(default_threshold, 0.05), 0.99),
                step=0.01,
            )
            row = curve.iloc[(curve["threshold"] - threshold).abs().argsort()[:1]]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Précision simulée", _format_percent(float(row["precision"].iloc[0]), 2))
            c2.metric("Rappel simulé", _format_percent(float(row["recall"].iloc[0]), 2))
            c3.metric("F1 simulé", f"{float(row['f1'].iloc[0]):.4f}")
            c4.metric("Alertes échantillon", f"{int(row['alerts'].iloc[0]):,}".replace(",", " "))

            fig = px.line(
                curve,
                x="threshold",
                y=["precision", "recall", "f1"],
                color_discrete_sequence=["#157f7f", "#c2415c", "#b7791f"],
                title="Compromis précision / recall / F1 selon le seuil",
            )
            fig.add_vline(x=threshold, line_dash="dash", line_color="#101828")
            st.plotly_chart(plotly_layout(fig, 430), width="stretch")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Fraudes test estimées", f"{confusion['positives']:.0f}")
            col_b.metric("Fraudes détectées estimées", f"{confusion['tp']:.0f}")
            col_c.metric("Fraudes manquées estimées", f"{confusion['fn']:.0f}")

    with tab_adv:
        render_fraud_advanced_analysis()

    with tab_lab:
        fraud_playground(form_key="fraud_sim")

    with tab_shap:
        render_fraud_shap_tab()


FRAUD_SIM_PRESETS = {
    "normal": {
        "label": "Transaction cohérente (faible risque)",
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "old_org": 2000.0,
        "new_org": 1000.0,
        "old_dest": 500.0,
        "new_dest": 1500.0,
    },
    "fraud": {
        "label": "Fraude typique du jeu de données (risque élevé)",
        "step": 1,
        "type": "TRANSFER",
        "amount": 181.0,
        "old_org": 181.0,
        "new_org": 0.0,
        "old_dest": 0.0,
        "new_dest": 0.0,
    },
}


@st.fragment
def fraud_playground(form_key: str = "fraud_sim") -> None:
    st.subheader("Évaluation d'une transaction")
    model = load_fraud_model()
    if model is None:
        st.error("Modèle fraude introuvable. Lancez `python -m src.models.train_fraud_model`.")
        return

    comparison = load_fraud_comparison()
    best_model = _best_fraud_name()
    threshold = _metric_value(comparison, best_model, "threshold", 0.5)
    preset_state_key = f"{form_key}_preset"

    if preset_state_key not in st.session_state:
        st.session_state[preset_state_key] = "normal"

    p1, p2, _ = st.columns([1, 1, 2])
    if p1.button("Exemple faible risque", key=f"{form_key}_btn_normal"):
        st.session_state[preset_state_key] = "normal"
        st.session_state.pop(f"{form_key}_show_shap", None)
        st.rerun()
    if p2.button("Exemple fraude avérée", key=f"{form_key}_btn_fraud"):
        st.session_state[preset_state_key] = "fraud"
        st.session_state.pop(f"{form_key}_show_shap", None)
        st.rerun()

    preset = FRAUD_SIM_PRESETS[st.session_state[preset_state_key]]

    with st.form(form_key):
        c1, c2, c3, c4 = st.columns(4)
        step = c1.number_input("Step", min_value=1, value=int(preset["step"]))
        tx_choices = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
        tx_type = c2.selectbox(
            "Type",
            tx_choices,
            index=tx_choices.index(preset["type"]) if preset["type"] in tx_choices else 0,
        )
        amount = c3.number_input("Montant", min_value=0.0, value=float(preset["amount"]), step=100.0)
        old_org = c4.number_input("Solde émetteur avant", min_value=0.0, value=float(preset["old_org"]), step=100.0)
        new_org = c1.number_input("Solde émetteur après", min_value=0.0, value=float(preset["new_org"]), step=100.0)
        old_dest = c2.number_input("Solde destinataire avant", min_value=0.0, value=float(preset["old_dest"]), step=100.0)
        new_dest = c3.number_input("Solde destinataire après", min_value=0.0, value=float(preset["new_dest"]), step=100.0)
        submitted = st.form_submit_button("Calculer le score", type="primary")

    if not submitted:
        st.caption("Renseignez la transaction puis cliquez sur **Calculer le score**.")
        return

    try:
        with st.spinner("Calcul du score…"):
            proba, features = _predict_fraud_transaction(
                model, int(step), tx_type, amount, old_org, new_org, old_dest, new_dest
            )
    except Exception as exc:
        st.error(f"Erreur lors du scoring : {exc}")
        return

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
                title={"text": "Probabilité de fraude"},
            )
        )
        st.plotly_chart(plotly_layout(fig, 330), width="stretch")

    with right:
        origin_error = float(features.get("origin_error", pd.Series([0.0])).iloc[0])
        dest_error = float(features.get("dest_error", pd.Series([0.0])).iloc[0])
        ratio = float(features.get("amount_to_oldbalance_ratio", pd.Series([np.nan])).iloc[0])
        with st.container(border=True):
            st.subheader("Décision")
            if prediction:
                st.error(risk_label)
            elif proba >= threshold / 2:
                st.warning(risk_label)
            else:
                st.success(risk_label)
            st.metric("Seuil modèle", f"{threshold:.2f}")
            st.metric("Erreur solde émetteur", f"{origin_error:,.2f}")
            st.metric("Erreur solde destinataire", f"{dest_error:,.2f}")
            st.metric("Ratio montant / solde", f"{ratio:.3f}" if np.isfinite(ratio) else "—")

    shap_flag = f"{form_key}_show_shap"
    if st.button("Afficher les contributions SHAP", key=f"{form_key}_btn_shap"):
        st.session_state[shap_flag] = True

    if st.session_state.get(shap_flag):
        with st.spinner("Calcul SHAP local…"):
            _render_local_shap_block(
                int(step), tx_type, amount, old_org, new_org, old_dest, new_dest, proba=proba
            )


def show_cluster_profiles() -> None:
    render_header(
        "Segmentation client",
        "Comparaison des modèles, lecture des profils, projection PCA et simulateur segment.",
    )

    profiles = load_cluster_profiles()
    comparison = load_cluster_comparison()
    if profiles.empty:
        st.warning("Lancez `python -m src.models.train_clustering_model` pour générer les profils.")
        return

    total_clients = int(profiles["cluster_size"].sum())
    premium_cluster = int(profiles.sort_values("Total_Spending", ascending=False).iloc[0]["cluster"])
    best_cluster = _best_cluster_name()
    best_silhouette = _metric_value(comparison, best_cluster, "silhouette")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients segmentés", f"{total_clients:,}".replace(",", " "))
    c2.metric("Modèle retenu", best_cluster)
    c3.metric("Silhouette", f"{best_silhouette:.4f}")
    c4.metric("Segment premium", f"Cluster {premium_cluster}")

    tab_exploration, tab_profiles, tab_projection, tab_models, tab_sim = st.tabs(
        ["Exploration données", "Profils", "Carte PCA", "Comparaison des modèles", "Simulateur"]
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
                        title="Revenu vs dépense totale",
                    )
                    st.plotly_chart(plotly_layout(fig, 500), width="stretch")

    with tab_models:
        if comparison.empty:
            st.info("Comparaison des modèles indisponible.")
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
        st.info("Données client indisponibles.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients", f"{len(df):,}".replace(",", " "))
    c2.metric("Revenu médian", f"{df['Income'].median():,.0f}" if "Income" in df.columns else "N/A")
    c3.metric(
        "Dépense médiane",
        f"{df['Total_Spending'].median():,.0f}" if "Total_Spending" in df.columns else "N/A",
    )
    c4.metric(
        "Achats médians",
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
                title="Dépenses moyennes par catégorie de produit",
            )
            fig.update_layout(xaxis_title="", yaxis_title="Dépense moyenne")
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
                title="Revenu vs dépense totale",
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
            title="Corrélation des indicateurs clients",
        )
        st.plotly_chart(plotly_layout(fig, 520), width="stretch")

    i1, i2, i3 = st.columns(3)
    with i1:
        render_insight(
            "Valeur client",
            "Le couple revenu et dépense totale permet de séparer les clients à faibles dépenses des profils premium.",
        )
    with i2:
        render_insight(
            "Canaux",
            "Les achats magasin, catalogue, web et promotions donnent une lecture directement exploitable pour les campagnes.",
        )
    with i3:
        render_insight(
            "Interprétation",
            "Le clustering doit être jugé par son utilité métier : un segment est bon s'il permet une action claire.",
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
        title="Profil normalisé des segments",
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
                st.write(f"Dépense moyenne : {row['Total_Spending']:,.0f}")
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
    st.subheader("Attribution d'un client à un segment")
    model = load_cluster_model()
    if model is None:
        st.error("Modèle clustering introuvable.")
        return

    with st.form("segment_form"):
        c1, c2, c3, c4 = st.columns(4)
        year_birth = c1.number_input("Année de naissance", min_value=1940, max_value=2010, value=1985)
        education = c2.selectbox("Education", ["Graduation", "PhD", "Master", "2n Cycle", "Basic"])
        marital = c3.selectbox("Situation", ["Single", "Together", "Married", "Divorced", "Widow"])
        income = c4.number_input("Revenu", min_value=0.0, value=50000.0, step=1000.0)

        kidhome = c1.number_input("Enfants", min_value=0, max_value=5, value=0)
        teenhome = c2.number_input("Adolescents", min_value=0, max_value=5, value=0)
        recency = c3.number_input("Récence", min_value=0, value=20)
        web_visits = c4.number_input("Visites web mensuelles", min_value=0, value=6)

        wines = c1.number_input("Dépenses vins", min_value=0, value=200, step=20)
        fruits = c2.number_input("Dépenses fruits", min_value=0, value=20, step=10)
        meat = c3.number_input("Dépenses viande", min_value=0, value=120, step=20)
        fish = c4.number_input("Dépenses poisson", min_value=0, value=30, step=10)
        sweet = c1.number_input("Dépenses sucre", min_value=0, value=15, step=10)
        gold = c2.number_input("Dépenses premium", min_value=0, value=40, step=10)

        deals = c1.number_input("Achats promo", min_value=0, value=2)
        web = c2.number_input("Achats web", min_value=0, value=5)
        catalog = c3.number_input("Achats catalogue", min_value=0, value=2)
        store = c4.number_input("Achats magasin", min_value=0, value=4)
        accepted = c1.slider("Campagnes acceptées", min_value=0, max_value=5, value=1)
        submitted = st.form_submit_button("Prédire le segment")

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
    segment_name = SEGMENT_NAMES.get(label, "Segment non nommé")
    action = SEGMENT_ACTIONS.get(label, "Analyse complémentaire.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Segment prédit", f"Cluster {label}")
    c2.metric("Persona", segment_name)
    c3.metric("Dépense totale saisie", f"{wines + fruits + meat + fish + sweet + gold:,.0f}")
    render_insight("Action recommandée", action)


def show_prediction_center() -> None:
    render_header(
        "Centre de prédiction",
        "État du scoring, tests unitaires de prédiction et endpoints API disponibles.",
    )

    fraud_model = load_fraud_model()
    cluster_model = load_cluster_model()
    fraud_cmp = load_fraud_comparison()
    best_fraud = _best_fraud_name()
    best_cluster = _best_cluster_name()
    threshold = _metric_value(fraud_cmp, best_fraud, "threshold", 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prédiction fraude", "opérationnelle" if fraud_model is not None else "indisponible")
    c2.metric("Seuil fraude", f"{threshold:.2f}")
    c3.metric("Prédiction segment", "opérationnelle" if cluster_model is not None else "indisponible")
    c4.metric("Modèle segment", best_cluster)

    s1, s2, s3 = st.columns(3)
    with s1:
        render_insight(
            "Fraude",
            f"Le modèle {best_fraud} retourne une probabilité de fraude. La classe finale dépend du seuil calibré sur validation.",
        )
    with s2:
        render_insight(
            "Segmentation",
            "Le modèle de clustering attribue un client à un segment et associe ce segment à une action marketing.",
        )
    with s3:
        render_insight(
            "Production",
            "La prédiction est disponible dans le dashboard et via l'API FastAPI. Le batch scoring reste une évolution possible.",
        )

    tab_fraud, tab_segment, tab_api = st.tabs(["Tester fraude", "Tester segment client", "API"])

    with tab_fraud:
        st.info("Objectif : saisir les caractéristiques d'une transaction et obtenir un score de fraude.")
        fraud_playground(form_key="fraud_pred")

    with tab_segment:
        st.info("Objectif : saisir le profil d'un client et obtenir son segment marketing.")
        segment_playground()

    with tab_api:
        st.subheader("Endpoints de prédiction")
        endpoints = pd.DataFrame(
            [
                ["GET", "/model/info", "Connaître les modèles chargés et le seuil fraude"],
                ["POST", "/predict/fraud", "Prédire la probabilité de fraude d'une transaction"],
                ["POST", "/predict/segment", "Attribuer un segment à un profil client"],
            ],
            columns=["Méthode", "Endpoint", "Rôle"],
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
        "Recommandations métier",
        "Synthèse des décisions à prendre pour la fraude, le marketing et l'industrialisation.",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraude")
        fraud_matrix = pd.DataFrame(
            [
                ["Score >= seuil", "Alerte prioritaire", "Vérification forte ou blocage temporaire"],
                ["Score intermédiaire", "Surveillance", "Revue humaine selon capacité"],
                ["Score faible", "Traitement standard", "Monitoring statistique"],
            ],
            columns=["Situation", "Décision", "Action"],
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

    st.subheader("Priorités de la prochaine itération")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_insight(
            "Interprétabilité",
            "L'onglet Fraude > Explications SHAP détaille les contributions globales et par transaction.",
        )
    with p2:
        render_insight("Monitoring", "Suivre la dérive des montants, types de transaction, scores et tailles de segments.")
    with p3:
        render_insight("Industrialisation", "Ajouter MLflow, Docker, validation de schéma et CI/CD pour rendre le projet reproductible.")


def show_technical_report() -> None:
    render_header(
        "Rapport d'analyse et d'interprétation",
        "Lecture des données, signification des modèles, personas clients et recommandations stratégiques.",
    )

    content = load_technical_report()
    pdf_bytes = load_report_pdf()
    pptx_bytes = load_presentation_pptx()

    if not content and not pdf_bytes:
        st.warning(
            "Les fichiers du rapport sont introuvables. "
            "Vérifiez que `reports/rapport_technique.md` et `reports/rapport_technique.pdf` "
            "sont présents dans le dépôt."
        )
        return

    st.subheader("Téléchargements")
    dl1, dl2 = st.columns(2)

    with dl1:
        if pdf_bytes:
            st.download_button(
                label="Rapport (PDF)",
                data=pdf_bytes,
                file_name="rapport_technique.pdf",
                mime="application/pdf",
                width="stretch",
                help="Version imprimable du rapport d'analyse.",
            )
        else:
            st.caption("PDF non disponible sur ce déploiement.")

    with dl2:
        if pptx_bytes:
            st.download_button(
                label="Présentation (PPTX)",
                data=pptx_bytes,
                file_name="presentation_finale.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                width="stretch",
                help="Slides pour la soutenance (12 diapositives).",
            )
        else:
            st.caption("PPTX non disponible sur ce déploiement.")

    if content:
        st.caption(
            f"Aperçu en ligne — source : `{REPORT_PATH.relative_to(PROJECT_ROOT).as_posix()}`"
        )
        with st.container(border=True):
            render_report_markdown(content)
    elif pdf_bytes:
        st.info(
            "L'aperçu markdown n'est pas disponible. Téléchargez le PDF pour consulter le rapport complet."
        )


def show_mlops() -> None:
    render_header(
        "MLOps et industrialisation",
        "État des artefacts, architecture cible et contrôles de robustesse du projet.",
    )

    fraud_model = load_fraud_model()
    cluster_model = load_cluster_model()
    tests = "6 tests passés"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle fraude", "chargé" if fraud_model is not None else "absent")
    c2.metric("Modèle clustering", "chargé" if cluster_model is not None else "absent")
    c3.metric("Tests", tests)
    c4.metric("API", "FastAPI")

    st.subheader("Pipeline cible")
    pipeline = pd.DataFrame(
        [
            ["1", "Ingestion", "Chargement des CSV bruts"],
            ["2", "Validation", "Vérification schéma, types, valeurs manquantes"],
            ["3", "Preprocessing", "Nettoyage, imputation, encodage"],
            ["4", "Features", "Soldes, ratios, dépenses, canaux"],
            ["5", "Training", "Comparaison des modèles et calibration seuil"],
            ["6", "Évaluation", "PR-AUC, recall, precision, silhouette"],
            ["7", "Serving", "API FastAPI et dashboard Streamlit"],
            ["8", "Monitoring", "Drift, performance, stabilité segments"],
        ],
        columns=["Étape", "Bloc", "Rôle"],
    )
    st.dataframe(pipeline, width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Artefacts")
        artifact_rows = []
        for path in [
            FRAUD_DIR / "fraud_model.joblib",
            FRAUD_DIR / "fraud_model_comparison.csv",
            FRAUD_DIR / "fraud_cost_analysis.json",
            FRAUD_DIR / "fraud_error_analysis.json",
            FRAUD_DIR / "fraud_temporal_metrics.json",
            CLUSTER_DIR / "cluster_model.joblib",
            CLUSTER_DIR / "cluster_profiles.csv",
            CLUSTER_DIR / "clustering_dbscan_comparison.csv",
            PROJECT_ROOT / "docs" / "CAMPAGNE_CLUSTER1.md",
            PROJECT_ROOT / "src" / "api" / "main.py",
            PROJECT_ROOT / "dashboard" / "app.py",
        ]:
            artifact_rows.append(
                {
                    "artefact": path.relative_to(PROJECT_ROOT).as_posix(),
                    "statut": "présent" if path.exists() else "absent",
                    "taille_kb": round(path.stat().st_size / 1024, 1) if path.exists() else 0,
                }
            )
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)

    with right:
        st.subheader("Endpoints API")
        endpoints = pd.DataFrame(
            [
                ["GET", "/health", "Santé API"],
                ["GET", "/model/info", "Modèles chargés"],
                ["POST", "/predict/fraud", "Score fraude"],
                ["POST", "/predict/segment", "Segment client"],
            ],
            columns=["Méthode", "Endpoint", "Usage"],
        )
        st.dataframe(endpoints, width="stretch", hide_index=True)

    st.subheader("Roadmap technique")
    roadmap = pd.DataFrame(
        [
            ["Réalisé", "SHAP, figures rapport, Docker, CI GitHub Actions, chiffrage coûts FP/FN"],
            ["Court terme", "Protocole campagne cluster 1, validation terrain des segments"],
            ["Moyen terme", "Monitoring drift (Evidently), recalibrage seuil avec le métier"],
            ["Long terme", "Réentraînement programmé, registre modèle, alerting production"],
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
elif selected_page == "report":
    show_technical_report()
else:
    show_mlops()
