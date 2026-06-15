# Monitoring et exploitation — architecture cible

Document de référence pour la Phase 3 MLOps du projet M2 CDSD.

## 1. État actuel (MVP)

| Brique | Implémentation | Emplacement |
|--------|----------------|-------------|
| Health check API | ✅ | `GET /health` |
| Infos modèles | ✅ | `GET /model/info` |
| Métriques offline | ✅ | `models/fraud/*.json`, `models/clustering/*.json` |
| Vérification artefacts | ✅ | `scripts/check_ml_health.py` |
| CI automatisée | ✅ | `.github/workflows/ci.yml` |
| Conteneurisation | ✅ | `Dockerfile`, `docker-compose.yml` |

## 2. Script de santé ML

```bash
python scripts/check_ml_health.py
```

Contrôles effectués :

- présence des modèles `joblib` fraude et clustering ;
- PR-AUC fraude ≥ 0,90 ;
- silhouette clustering ≥ 0,10 ;
- cohérence du seuil fraude entre `fraud_model_comparison.json`, `fraud_error_analysis.json` et `fraud_cost_analysis.json` ;
- fichiers de métriques temporelles, d'erreurs et de coûts présents.

Le script retourne un code de sortie `0` (OK) ou `1` (alerte) — intégré dans la CI GitHub Actions.

## 3. Architecture de monitoring cible (production)

```text
                    ┌─────────────────┐
  Requêtes API  ──► │  FastAPI        │
                    │  (scoring)      │
                    └────────┬────────┘
                             │ logs JSON
                             ▼
                    ┌─────────────────┐
                    │ Collecte        │  ← volume alertes, latence, erreurs 5xx
                    │ (stdout / ELK)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ Performance │   │ Data drift  │   │ Seuils      │
  │ (PR-AUC,    │   │ (Evidently  │   │ métier      │
  │  recall)    │   │  AI)        │   │ (alertes/j) │
  └─────────────┘   └─────────────┘   └─────────────┘
```

## 4. Indicateurs recommandés

### Fraude

| KPI | Fréquence | Seuil d'alerte suggéré |
|-----|-----------|------------------------|
| Recall | Hebdomadaire | < 90 % |
| Precision | Hebdomadaire | < 95 % |
| Volume d'alertes / jour | Quotidien | > capacité équipe |
| PR-AUC sur holdout temporel | Mensuel | −5 % vs baseline |
| Dérive `step`, `type`, soldes | Mensuel | PSI > 0,2 |

### Segmentation

| KPI | Fréquence | Seuil d'alerte suggéré |
|-----|-----------|------------------------|
| Taille des clusters | Trimestriel | cluster < 2 % de la base |
| Silhouette recalculée | Trimestriel | < 0,15 |
| Stabilité des personas | Semestriel | > 20 % de migration inter-segments |

## 5. Évolutions prioritaires

1. **MLflow** — tracking centralisé des expériences (params, métriques, artefacts).
2. **Evidently AI** — rapports de drift sur les features fraude (`origin_error`, `type`, `amount`).
3. **Re-entraînement planifié** — cron mensuel si dérive détectée.
4. **Alerting** — Slack/email si `check_ml_health.py` échoue en production.

## 6. POC drift (manuel)

Comparer la distribution de `type` et `amount` entre :

- `data/raw/detection_fraude.csv` (référence) ;
- un export de production futur.

Si le taux de `TRANSFER` ou la médiane des montants varie de plus de 20 %, déclencher une revue modèle avant le prochain cycle d'entraînement.
