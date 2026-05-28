"""Fonctions de visualisation reutilisables."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(df, target_col: str = "isFraud"):
    """Affiche la distribution de la variable cible."""
    ax = sns.countplot(data=df, x=target_col)
    ax.set_title(f"Distribution de {target_col}")
    return ax


def plot_numeric_histogram(df, column: str):
    """Affiche un histogramme simple pour une variable numerique."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[column], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution de {column}")
    return ax
