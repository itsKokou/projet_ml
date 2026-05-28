# Plan operationnel du projet (J1 a J7)

## Tableau de bord de suivi

### Avancement global

- [x] Statut global : En cours (J1 termine)
- [x] Progression estimee : 98 %
- [x] Date de debut : 27/05/2026
- [ ] Date cible de fin :
- [ ] Responsable(s) :

### Etat par jour

- [x] J1 termine
- [x] J2 termine
- [x] J3 termine
- [x] J4 termine
- [x] J5 termine
- [x] J6 termine
- [ ] J7 termine (presque finalise)

### Bloquants et risques en cours

- [x] Aucun bloquant pour le moment
- [ ] Bloquant 1 :
- [ ] Bloquant 2 :
- [ ] Risque principal : desequilibre extreme de `isFraud` (necessite seuil metier adapte)
- [x] Risque technique traite : `libomp` installe, `xgboost` operationnel

### Prochaines actions (48h)

- [x] Action prioritaire 1 : demarrer `01_eda_fraude.ipynb` (profilage + distributions)
- [x] Action prioritaire 2 : demarrer `03_eda_segmentation.ipynb` (manquants + depenses)
- [x] Action prioritaire 3 : fixer les hypotheses de features pour J4/J5

### Journal de suivi rapide

- [x] 27/05 - Avancement du jour : arborescence, modules, API/dashboard/tests squelettes crees
- [x] 27/05 - Decision prise : commencer par EDA fraude puis EDA segmentation
- [x] 27/05 - Avancement du jour : notebook `01_eda_fraude.ipynb` initialise avec trame complete
- [x] 27/05 - Avancement du jour : notebook `03_eda_segmentation.ipynb` initialise avec trame complete
- [x] 27/05 - Avancement du jour : notebooks J2/J3 executes, premiers constats metier valides
- [x] 27/05 - Avancement du jour : baseline fraude entrainee + seuil calibre + metriques exportees
- [x] 27/05 - Avancement du jour : tests unitaires executes (2 passed)
- [x] 27/05 - Avancement du jour : comparaison LogReg vs RandomForest exportee (JSON/CSV)
- [x] 27/05 - Avancement du jour : comparaison LogReg vs RandomForest vs XGBoost executee
- [x] 27/05 - Avancement du jour : comparaison clustering (KMeans/Agglo/GMM) + profils segments exportes
- [x] 27/05 - Avancement du jour : API branchee sur modeles reels + dashboard multi-pages + tests API
- [x] 27/05 - Avancement du jour : README finalise (reproduction, resultats, demo API)
- [x] 27/05 - Avancement du jour : plan de presentation cree (`reports/presentation_outline.md`)
- [ ] JJ/MM - Point a verifier demain :

## Objectif

Livrer un projet Data Science + MLOps propre, reproductible et defendable a la soutenance, avec :
- detection de fraude (classification desequilibree),
- segmentation client (clustering interpretable),
- une brique d'industrialisation (API et/ou dashboard).

## J1 - Initialisation propre du depot

- [x] Creer l'arborescence cible : `data/raw`, `data/processed`, `notebooks`, `src`, `models`, `dashboard`, `tests`.
- [x] Deplacer les CSV dans `data/raw/`.
- [x] Creer un `requirements.txt` minimal.
- [x] Completer le `README.md` avec installation + execution.
- [x] Initialiser les modules Python (`__init__.py`) dans `src/`.

### Livrable J1

- [x] Projet installable localement.
- [x] Structure claire et versionnable.

## J2 - EDA fraude (`01_eda_fraude.ipynb`)

- [x] Profilage des donnees (types, manquants, doublons, distributions).
- [x] Confirmation du desequilibre de `isFraud`.
- [x] Analyse par `type`, `amount`, `step`.
- [x] Production de 5 a 7 visualisations lisibles.
- [x] Section finale "insights metier" + hypotheses de features.

### Checklist qualite J2

- [x] Chaque graphique repond a une question metier.
- [x] Une conclusion exploitable est ecrite en fin de notebook.

## J3 - EDA segmentation (`03_eda_segmentation.ipynb`)

- [x] Analyse demographique/comportementale/canaux.
- [x] Analyse des valeurs manquantes de `Income`.
- [x] Variables exploratoires : `Total_Spending`, ratios de canaux.
- [x] Identification des colonnes a exclure (`ID`, `Z_CostContact`, `Z_Revenue`).
- [x] Visualisations de profils clients.

### Livrable J3

- [x] Notebook avec hypotheses de segmentation.
- [x] Liste des variables candidates pour le clustering.

## J4 - Pipeline fraude (scripts + modelisation)

- [x] Creer `src/data/preprocessing.py` et `src/features/build_features.py`.
- [x] Split `train/validation/test` (stratifie, test final gele).
- [x] Baseline : regression logistique.
- [x] Modeles comparatifs : RandomForest + XGBoost/LightGBM (si disponible).
- [x] Gestion du desequilibre (`class_weight`, seuil, eventuel SMOTE cote train uniquement).
- [x] Evaluation avec `PR-AUC`, `Recall`, `Precision`, `F1`, matrice de confusion.

### Point cle J4

- [x] Choisir un seuil de decision metier justifie (pas seulement 0.5).

## J5 - Pipeline clustering + interpretation metier

- [x] Creer `src/models/train_clustering_model.py`.
- [x] Pipeline complet : imputation, encodage, normalisation.
- [x] Tester KMeans + Agglomerative + GMM (DBSCAN en bonus).
- [x] Selection de `k` via Elbow + Silhouette + lisibilite metier.
- [x] Profiler les clusters et rediger des recommandations actionnables.
- [x] Sauvegarder pipeline + modele de clustering.

### Livrable J5

- [x] Tableau des segments + actions marketing recommandees.

## J6 - Industrialisation MVP (API + dashboard + tests)

- [x] API FastAPI minimale :
  - [x] `GET /health`
  - [x] `POST /predict/fraud`
  - [x] `POST /predict/segment`
- [x] Dashboard Streamlit (3 a 4 pages max) :
  - [x] vue d'ensemble
  - [x] resultats fraude
  - [x] profils clusters
  - [x] recommandations
- [x] Tests de base :
  - [x] chargement des donnees
  - [x] preprocessing
  - [x] prediction sur entree valide

### Objectif J6

- [x] Demonstration bout-en-bout fonctionnelle.

## J7 - Finalisation soutenance

- [ ] Nettoyer les notebooks (structure, markdown, conclusions).
- [ ] Geler les dependances dans `requirements.txt`.
- [x] Rediger une section "Reproduction" en 3 commandes max.
- [x] Preparer les slides : probleme -> methode -> resultats -> limites -> evolutions.
- [x] Verifier la coherence des features entre entrainement, API et dashboard.

### Livrables finaux J7

- [ ] Depot propre et lisible.
- [ ] Notebooks et scripts reutilisables.
- [ ] Demo API/dashboard.
- [ ] Presentation defendable.

## Fichiers prioritaires a creer (ordre recommande)

- [x] `requirements.txt`
- [x] `src/data/load_data.py`
- [x] `src/data/preprocessing.py`
- [x] `src/features/build_features.py`
- [x] `src/models/train_fraud_model.py`
- [x] `src/models/train_clustering_model.py`
- [x] `src/models/evaluate.py`
- [x] `src/api/main.py`
- [x] `dashboard/app.py`
- [x] `tests/test_preprocessing.py`

## Criteres go/no-go avant rendu

- [x] Fraude : metriques adaptees + seuil justifie metier.
- [x] Clustering : segments lisibles + actions marketing associees.
- [x] Reproductibilite : installation et execution simples.
- [x] Documentation : README clair et complet.
- [x] Demo : API ou dashboard fonctionnel.
