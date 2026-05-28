# Projet Machine Learning M2 CDSD

## Demarrage rapide (etat actuel)

### Prerequis

- Python 3.10+
- `pip`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Commandes utiles

```bash
# Lancer les tests
pytest -q

# Lancer l'API
uvicorn src.api.main:app --reload

# Lancer le dashboard
streamlit run dashboard/app.py
```

### Donnees

Les fichiers de donnees brutes sont places dans `data/raw/` :
- `data/raw/detection_fraude.csv`
- `data/raw/data_cluster.csv`

## Reproduction en 3 commandes

```bash
# 1) Installer l'environnement
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 2) Entrainer les modeles et generer les artefacts
python -m src.models.train_fraud_model && python -m src.models.train_clustering_model

# 3) Lancer la demo (API ou dashboard)
uvicorn src.api.main:app --reload
# ou
streamlit run dashboard/app.py
```

## Resultats obtenus (execution locale)

### Detection de fraude

- Meilleur modele: `xgboost` (selection via `PR-AUC`).
- Metriques test:
  - `pr_auc`: `0.9910`
  - `recall`: `0.9825`
  - `precision`: `1.0000`
  - `threshold`: `0.84`
- Artefacts generes dans `models/fraud/`:
  - `fraud_model.joblib`
  - `fraud_model_comparison.csv`
  - `fraud_model_comparison.json`
  - `fraud_best_model.json`

### Segmentation client

- Meilleur modele: `gmm_k4`.
- Metriques:
  - `silhouette`: `0.2100`
  - `davies_bouldin`: `2.4230`
  - `calinski_harabasz`: `305.41`
- Artefacts generes dans `models/clustering/`:
  - `cluster_model.joblib`
  - `clustering_model_comparison.csv`
  - `clustering_model_comparison.json`
  - `cluster_profiles.csv`

## API de demo

Endpoints exposes:
- `GET /health`
- `GET /model/info`
- `POST /predict/fraud`
- `POST /predict/segment`

Exemple de test rapide:

```bash
curl -X POST "http://127.0.0.1:8000/predict/fraud" \
  -H "Content-Type: application/json" \
  -d '{"payload":{"step":1,"type":"TRANSFER","amount":1000,"nameOrig":"C1","oldbalanceOrg":2000,"newbalanceOrig":1000,"nameDest":"C2","oldbalanceDest":500,"newbalanceDest":1500}}'
```


## 1. Vue d'ensemble

Ce projet a pour objectif de construire une demarche complete de Data Science autour de deux cas d'usage complementaires :

1. Detection automatique de transactions frauduleuses a partir de donnees bancaires.
2. Segmentation intelligente de clients a partir de donnees marketing.

Le projet doit egalement integrer une reflexion MLOps afin de montrer comment les modeles peuvent etre industrialises, suivis et ameliores dans le temps.

Le travail attendu ne se limite pas a entrainer des modeles. Il faut produire une analyse claire, des visualisations pertinentes, des interpretations metier, une organisation de code propre et une proposition d'architecture permettant de faire evoluer le projet.

## 2. Donnees disponibles

### 2.1 Detection de fraude

Fichier :

- `data/raw/detection_fraude.csv`

Objectif :

- predire la variable cible `isFraud`.

Variables principales :

- `step` : unite temporelle de la transaction.
- `type` : type de transaction (`PAYMENT`, `TRANSFER`, `CASH_OUT`, etc.).
- `amount` : montant transfere.
- `nameOrig` : identifiant du client emetteur.
- `oldbalanceOrg` : solde de l'emetteur avant transaction.
- `newbalanceOrig` : solde de l'emetteur apres transaction.
- `nameDest` : identifiant du destinataire.
- `oldbalanceDest` : solde du destinataire avant transaction.
- `newbalanceDest` : solde du destinataire apres transaction.
- `isFraud` : cible, avec `1` pour fraude et `0` pour transaction normale.
- `isFlaggedFraud` : indicateur de transaction signalee automatiquement.

Observations initiales :

- Le fichier contient environ 1 048 575 transactions.
- La classe fraude est tres minoritaire : environ 0,109 % des transactions.
- Les fraudes observees apparaissent principalement sur les types `TRANSFER` et `CASH_OUT`.
- Le probleme est donc un cas classique de classification fortement desequilibree.

### 2.2 Segmentation client

Fichier :

- `data/raw/data_cluster.csv`

Objectif :

- identifier des groupes de clients ayant des comportements similaires.

Variables principales :

- Variables demographiques : `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`.
- Variables comportementales : `Recency`, `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`.
- Variables de canaux d'achat : `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`.
- Variables marketing : `AcceptedCmp1` a `AcceptedCmp5`, `Response`, `Complain`.

Observations initiales :

- Le fichier contient 2 240 clients.
- La variable `Income` contient quelques valeurs manquantes.
- Le jeu de donnees permet de construire des profils clients : clients premium, clients dormants, clients digitaux, clients sensibles aux promotions, etc.

## 3. Objectifs du projet

### 3.1 Objectifs Data Science

- Comprendre les donnees disponibles.
- Nettoyer et preparer les donnees.
- Creer des variables pertinentes pour ameliorer les performances.
- Entrainer plusieurs modeles adaptes a chaque probleme.
- Evaluer les modeles avec des metriques pertinentes.
- Interpreter les resultats de maniere comprehensible pour un public metier.
- Proposer des recommandations actionnables.

### 3.2 Objectifs MLOps

- Organiser le code comme un vrai projet professionnel.
- Separer les etapes d'ingestion, nettoyage, entrainement, evaluation et prediction.
- Versionner les donnees, les modeles et les parametres.
- Prevoir une API ou une interface de demonstration.
- Mettre en place un minimum de monitoring.
- Documenter le projet pour faciliter sa reprise et son evolution.

## 4. Structure recommandee du depot

La structure suivante est conseillee pour faire evoluer le projet proprement :

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── detection_fraude.csv
│   │   └── data_cluster.csv
│   ├── processed/
│   └── external/
├── notebooks/
│   ├── 01_eda_fraude.ipynb
│   ├── 02_modelisation_fraude.ipynb
│   ├── 03_eda_segmentation.ipynb
│   ├── 04_clustering_clients.ipynb
│   └── 05_mlops_synthese.ipynb
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train_fraud_model.py
│   │   ├── train_clustering_model.py
│   │   └── evaluate.py
│   ├── visualization/
│   │   └── plots.py
│   └── api/
│       └── main.py
├── models/
│   ├── fraud/
│   └── clustering/
├── reports/
│   ├── figures/
│   ├── rapport_technique.pdf
│   └── presentation_finale.pptx
├── dashboard/
│   └── app.py
├── tests/
│   └── test_preprocessing.py
├── Dockerfile
├── docker-compose.yml
└── .gitignore
```

Remarque importante :

- Les gros fichiers de donnees ne doivent pas forcement etre versionnes directement dans GitHub.
- Pour un depot public ou collaboratif, utiliser plutot Git LFS, DVC ou un lien de telechargement documente.

## 5. Etape 1 - Cadrage metier

Avant de coder, il faut clarifier le probleme.

### Detection de fraude

Questions a se poser :

- Quel est le cout d'un faux positif ?
- Quel est le cout d'un faux negatif ?
- Faut-il detecter toutes les fraudes, meme au prix de beaucoup d'alertes ?
- Quel seuil de decision est acceptable pour l'equipe metier ?
- Le modele doit-il fonctionner en temps reel ou en batch ?

Priorite metier probable :

- maximiser le rappel des fraudes afin de rater le moins possible de transactions frauduleuses ;
- garder une precision suffisante pour ne pas saturer les equipes de controle.

### Segmentation client

Questions a se poser :

- Quels types d'actions marketing seront declenches par les segments ?
- Cherche-t-on des segments faciles a expliquer ou seulement performants mathematiquement ?
- Combien de segments sont exploitables par les equipes marketing ?
- Les segments doivent-ils etre stables dans le temps ?

Priorite metier probable :

- produire des segments interpretables, actionnables et utilisables dans une campagne marketing.

## 6. Etape 2 - Analyse exploratoire des donnees

### 6.1 EDA pour la detection de fraude

Analyses a realiser :

- Nombre total de transactions.
- Repartition de `isFraud`.
- Taux de fraude global.
- Repartition des types de transaction.
- Taux de fraude par type de transaction.
- Distribution des montants.
- Comparaison des montants frauduleux et non frauduleux.
- Analyse des soldes avant/apres transaction.
- Verification des incoherences :
  - `oldbalanceOrg - amount != newbalanceOrig`
  - `oldbalanceDest + amount != newbalanceDest`
- Analyse de `isFlaggedFraud`.
- Analyse temporelle avec `step`.
- Detection de clients ou destinataires apparaissant souvent dans les fraudes.

Visualisations recommandees :

- barplot de la variable cible ;
- barplot du taux de fraude par type de transaction ;
- histogramme ou boxplot des montants ;
- courbe temporelle du nombre de fraudes par `step` ;
- matrice de correlation des variables numeriques ;
- heatmap des comportements suspects.

### 6.2 EDA pour la segmentation client

Analyses a realiser :

- Distribution de l'age.
- Distribution du revenu.
- Analyse des valeurs manquantes.
- Analyse des depenses par categorie de produit.
- Depense totale par client.
- Repartition des niveaux d'education.
- Repartition des situations matrimoniales.
- Analyse de la recence d'achat.
- Analyse des canaux d'achat :
  - web ;
  - catalogue ;
  - magasin ;
  - promotions.
- Analyse de la reponse aux campagnes marketing.

Visualisations recommandees :

- histogramme des revenus ;
- boxplot des depenses par categorie ;
- barplot des niveaux d'education ;
- scatterplot revenu vs depense totale ;
- heatmap de correlation ;
- pairplot sur les variables principales ;
- PCA 2D pour visualiser les clients avant clustering.

## 7. Etape 3 - Nettoyage et preprocessing

### 7.1 Regles communes

- Verifier les doublons.
- Verifier les valeurs manquantes.
- Harmoniser les types de donnees.
- Identifier les valeurs aberrantes.
- Separer les variables numeriques et categorielles.
- Construire un pipeline reproductible avec `sklearn.pipeline`.
- Eviter toute fuite de donnees entre train et test.

### 7.2 Preprocessing pour la fraude

Actions recommandees :

- Supprimer ou transformer les identifiants `nameOrig` et `nameDest`.
- Encoder la variable `type`.
- Creer des variables de controle des soldes :
  - difference entre ancien solde et nouveau solde emetteur ;
  - difference entre ancien solde et nouveau solde destinataire ;
  - indicateur d'erreur de solde ;
  - ratio montant / solde initial.
- Standardiser ou normaliser les variables numeriques si necessaire.
- Creer un jeu d'entrainement et un jeu de test avec stratification sur `isFraud`.
- Gerer le desequilibre des classes.

Methodes possibles pour le desequilibre :

- `class_weight='balanced'` pour certains modeles.
- Sous-echantillonnage de la classe majoritaire.
- Sur-echantillonnage de la classe minoritaire.
- SMOTE, a utiliser avec prudence.
- Seuil de decision optimise selon le compromis precision/rappel.

### 7.3 Preprocessing pour le clustering

Actions recommandees :

- Traiter les valeurs manquantes de `Income`.
- Creer une variable `Age` a partir de `Year_Birth`.
- Creer une variable `Customer_Tenure` a partir de `Dt_Customer`.
- Creer une variable `Total_Spending`.
- Creer une variable `Total_Purchases`.
- Creer une variable `Children` = `Kidhome + Teenhome`.
- Creer des ratios :
  - part des achats web ;
  - part des achats magasin ;
  - part des achats catalogue ;
  - part des achats promotionnels.
- Encoder `Education` et `Marital_Status`.
- Standardiser les variables numeriques.
- Reduire la dimension avec PCA si utile.

Variables a supprimer ou a traiter avec prudence :

- `ID` : identifiant technique.
- `Z_CostContact` et `Z_Revenue` : variables constantes ou peu informatives si elles n'apportent pas de variance.

## 8. Etape 4 - Feature engineering

### 8.1 Features utiles pour la fraude

Exemples de variables a creer :

- `origin_balance_diff = oldbalanceOrg - newbalanceOrig`
- `dest_balance_diff = newbalanceDest - oldbalanceDest`
- `origin_error = origin_balance_diff - amount`
- `dest_error = dest_balance_diff - amount`
- `amount_to_oldbalance_ratio = amount / oldbalanceOrg`
- `is_transfer_or_cashout`
- `is_zero_newbalance_origin`
- `is_zero_oldbalance_dest`
- frequence d'apparition de `nameOrig`
- frequence d'apparition de `nameDest`

But :

- aider le modele a detecter les transactions incoherentes ou typiques d'une fraude.

### 8.2 Features utiles pour la segmentation

Exemples de variables a creer :

- `Age`
- `Customer_Tenure`
- `Total_Spending`
- `Total_Purchases`
- `Average_Basket`
- `Children`
- `Campaign_Acceptance_Total`
- `Web_Purchase_Ratio`
- `Store_Purchase_Ratio`
- `Catalog_Purchase_Ratio`
- `Deals_Purchase_Ratio`

But :

- passer de variables brutes a des indicateurs metier faciles a interpreter.

## 9. Etape 5 - Modelisation de la detection de fraude

### 9.1 Separation des donnees

Approche recommandee :

- separer `X` et `y` ;
- utiliser un split stratifie ;
- garder un jeu de test final non touche jusqu'a l'evaluation ;
- eventuellement creer un jeu de validation.

Exemple de repartition :

- 70 % entrainement ;
- 15 % validation ;
- 15 % test.

### 9.2 Modeles a tester

Modeles de base :

- Regression logistique.
- Decision Tree.
- Random Forest.

Modeles avances :

- XGBoost.
- LightGBM.
- CatBoost si disponible.
- Reseau de neurones simple.

Baseline importante :

- toujours commencer par un modele simple et interpretable.
- comparer ensuite avec des modeles plus puissants.

### 9.3 Evaluation

Metriques a calculer :

- accuracy ;
- precision ;
- recall ;
- F1-score ;
- ROC-AUC ;
- PR-AUC ;
- matrice de confusion.

Point important :

- avec une classe fraude tres rare, la PR-AUC et le recall sont souvent plus informatifs que l'accuracy.

Analyses a produire :

- comparaison des modeles dans un tableau ;
- courbe ROC ;
- courbe precision-rappel ;
- choix du seuil de classification ;
- analyse des faux positifs ;
- analyse des faux negatifs.

## 10. Etape 6 - Clustering client

### 10.1 Preparation des donnees

Avant le clustering :

- retirer les identifiants ;
- imputer les valeurs manquantes ;
- encoder les variables categorielles ;
- standardiser les variables ;
- selectionner les variables vraiment utiles ;
- eventuellement appliquer PCA.

### 10.2 Modeles de clustering a tester

Modeles attendus :

- K-Means.
- DBSCAN.
- Agglomerative Clustering.
- Gaussian Mixture Models.

### 10.3 Evaluation des clusters

Metriques et methodes :

- Elbow Method.
- Silhouette Score.
- Davies-Bouldin Score.
- Calinski-Harabasz Score si utile.
- Visualisation PCA 2D ou t-SNE.

Point important :

- le meilleur score mathematique n'est pas toujours le meilleur choix metier.
- il faut privilegier des segments lisibles, stables et actionnables.

### 10.4 Interpretation metier des clusters

Pour chaque cluster, produire :

- taille du segment ;
- revenu moyen ;
- age moyen ;
- depense totale moyenne ;
- canaux d'achat preferes ;
- sensibilite aux promotions ;
- taux de reponse aux campagnes ;
- comportement recent ou dormant ;
- recommandation marketing.

Exemples de profils possibles :

- Clients premium : revenu eleve, fortes depenses, faible sensibilite aux promotions.
- Clients digitaux : achats web importants, visites web frequentes.
- Clients sensibles aux promotions : nombreux achats avec offres.
- Clients dormants : forte recence, faibles achats recents.
- Clients a potentiel : revenu correct, engagement encore faible mais activable.

## 11. Etape 7 - Interpretabilite

### 11.1 Detection de fraude

Approches recommandees :

- importance des variables pour Random Forest, XGBoost ou LightGBM ;
- coefficients de regression pour la regression logistique ;
- SHAP values pour expliquer les predictions ;
- analyse locale de quelques transactions frauduleuses ;
- comparaison entre transactions bien classees et mal classees.

Questions a traiter :

- quelles variables expliquent le plus les fraudes ?
- le modele s'appuie-t-il sur des signaux coherents ?
- quels cas sont difficiles a detecter ?
- quels types de transactions generent des faux positifs ?

### 11.2 Segmentation client

Approches recommandees :

- moyenne des variables par cluster ;
- radar charts ;
- barplots par segment ;
- projection PCA coloree par cluster ;
- description textuelle de chaque segment.

Questions a traiter :

- les clusters sont-ils distincts ?
- les segments sont-ils interpretables ?
- peut-on associer une action marketing claire a chaque segment ?

## 12. Etape 8 - Dashboard interactif

Un dashboard est attendu pour rendre les resultats accessibles.

Technologies possibles :

- Streamlit, recommande pour aller vite.
- Dash, utile pour une application plus structuree.

Pages recommandees :

1. Accueil du projet.
2. Detection de fraude.
3. Segmentation client.
4. Comparaison des modeles.
5. Interpretation et recommandations.

Fonctionnalites utiles :

- upload ou chargement des donnees ;
- affichage des statistiques descriptives ;
- visualisation de la repartition des fraudes ;
- prediction d'une transaction ;
- affichage du cluster d'un client ;
- graphiques interactifs ;
- synthese des recommandations metier.

## 13. Etape 9 - API de prediction

Une API simple peut etre proposee pour montrer comment deployer les modeles.

Technologie recommandee :

- FastAPI.

Endpoints possibles :

- `GET /health` : verifier que l'API fonctionne.
- `POST /predict/fraud` : predire si une transaction est frauduleuse.
- `POST /predict/segment` : attribuer un client a un segment.
- `GET /model/info` : afficher la version du modele charge.

Bonnes pratiques :

- valider les entrees avec Pydantic ;
- charger le modele au demarrage ;
- retourner une probabilite et pas seulement une classe ;
- logger les predictions ;
- gerer les erreurs proprement.

## 14. Etape 10 - MLOps

### 14.1 Pipeline

Pipeline cible :

1. Ingestion des donnees.
2. Validation du schema.
3. Nettoyage.
4. Feature engineering.
5. Separation train/test.
6. Entrainement.
7. Evaluation.
8. Sauvegarde du modele.
9. Generation d'un rapport de performance.
10. Deploiement ou mise a jour du dashboard.

Outils possibles :

- `scikit-learn Pipeline`
- `joblib`
- `MLflow`
- `DVC`
- `Great Expectations`
- `Prefect` ou `Airflow` pour orchestration avancee

### 14.2 Versioning

Elements a versionner :

- code source avec Git ;
- dependances avec `requirements.txt` ;
- donnees avec DVC ou Git LFS ;
- modeles avec MLflow ou un dossier `models/` versionne proprement ;
- parametres d'entrainement dans des fichiers YAML ou JSON ;
- metriques dans des fichiers de resultats.

### 14.3 Monitoring

Pour le modele de fraude :

- taux de transactions predites frauduleuses ;
- distribution des probabilites ;
- precision et recall si les labels reels arrivent plus tard ;
- derive des variables ;
- derive du score ;
- augmentation des faux positifs ;
- temps de reponse de l'API.

Pour le clustering :

- stabilite des clusters ;
- evolution de la taille des segments ;
- changement des profils moyens ;
- apparition de nouveaux comportements clients ;
- baisse du silhouette score lors d'un recalcul.

### 14.4 CI/CD

Pipeline CI/CD simplifie :

1. Push du code sur GitHub ou GitLab.
2. Execution automatique des tests.
3. Verification du formatage du code.
4. Entrainement optionnel sur un echantillon.
5. Construction de l'image Docker.
6. Deploiement de l'API ou du dashboard.

Outils possibles :

- GitHub Actions.
- GitLab CI.
- Docker.
- Docker Compose.
- Render, Railway, Hugging Face Spaces ou serveur interne.

## 15. Etape 11 - Livrables attendus

Livrables minimum :

- notebooks d'analyse exploratoire ;
- notebooks de modelisation ;
- scripts Python reutilisables ;
- visualisations propres ;
- rapport technique ;
- presentation finale ;
- dashboard interactif ;
- README complet ;
- fichier `requirements.txt` ;
- architecture MLOps ;
- depot GitHub bien organise.

Livrables recommandes :

- modele sauvegarde avec `joblib` ou `pickle` ;
- fichier de metriques ;
- matrice de confusion exportee ;
- graphiques d'interpretabilite ;
- exemples de requetes API ;
- documentation d'installation ;
- documentation de reproduction des resultats.

## 16. Qualite du code

Bonnes pratiques :

- utiliser des noms de variables explicites ;
- eviter le code duplique ;
- separer les fonctions metier des notebooks ;
- garder les notebooks propres et lisibles ;
- ajouter des commentaires uniquement quand ils expliquent une logique utile ;
- creer des fonctions pour les traitements repetes ;
- documenter les choix importants ;
- fixer les seeds aleatoires ;
- ne pas melanger exploration rapide et code final.

Exemples de modules utiles :

- `load_data.py` pour charger les donnees.
- `preprocessing.py` pour nettoyer les donnees.
- `build_features.py` pour creer les variables.
- `train_fraud_model.py` pour entrainer le modele de fraude.
- `train_clustering_model.py` pour entrainer le clustering.
- `evaluate.py` pour centraliser les metriques.
- `plots.py` pour centraliser les visualisations.

## 17. Tests a prevoir

Tests simples mais utiles :

- verifier que les fichiers de donnees sont chargeables ;
- verifier que les colonnes attendues existent ;
- verifier que le preprocessing ne retourne pas de valeurs manquantes non gerees ;
- verifier que les dimensions de sortie sont coherentes ;
- verifier que le modele sauvegarde peut etre recharge ;
- verifier que l'API repond sur `/health`.

## 18. Requirements recommandes

Dependances probables :

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
jupyter
imbalanced-learn
xgboost
lightgbm
shap
joblib
streamlit
fastapi
uvicorn
mlflow
```

Selon les choix techniques, certaines dependances peuvent etre retirees.

## 19. Commandes utiles

Creer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
source .venv/bin/activate
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter :

```bash
jupyter notebook
```

Lancer un dashboard Streamlit :

```bash
streamlit run dashboard/app.py
```

Lancer une API FastAPI :

```bash
uvicorn src.api.main:app --reload
```

## 20. Roadmap conseillee

### Phase 1 - Mise en place

- Organiser les dossiers du projet.
- Creer `requirements.txt`.
- Deplacer les donnees dans `data/raw/`.
- Creer les premiers notebooks.
- Faire une premiere lecture des donnees.

### Phase 2 - Analyse exploratoire

- Produire l'EDA fraude.
- Produire l'EDA segmentation.
- Identifier les problemes de qualite des donnees.
- Generer les premieres visualisations.

### Phase 3 - Preprocessing et features

- Construire les pipelines de nettoyage.
- Creer les variables metier.
- Sauvegarder les donnees preparees.
- Documenter les choix.

### Phase 4 - Modelisation

- Entrainer les modeles de detection de fraude.
- Comparer les performances.
- Optimiser le seuil de decision.
- Entrainer les modeles de clustering.
- Choisir le nombre de clusters.

### Phase 5 - Interpretation

- Interpreter le meilleur modele de fraude.
- Analyser les erreurs du modele.
- Decrire chaque cluster client.
- Produire des recommandations metier.

### Phase 6 - Industrialisation

- Transformer le code notebook en scripts reutilisables.
- Sauvegarder les modeles.
- Creer une API ou un dashboard.
- Ajouter Docker si possible.
- Ajouter une architecture MLOps dans le rapport.

### Phase 7 - Finalisation

- Nettoyer les notebooks.
- Verifier la reproductibilite.
- Rediger le rapport technique.
- Preparer la presentation finale.
- Mettre a jour le README.
- Verifier le depot GitHub.

## 21. Pistes d'evolution

### Evolution du modele de fraude

- Ajouter une optimisation des hyperparametres avec GridSearchCV, RandomizedSearchCV ou Optuna.
- Tester des modeles specialises pour donnees desequilibrees.
- Mettre en place un seuil de decision adapte au cout metier.
- Ajouter une analyse temporelle plus fine.
- Construire des features de reseau entre emetteurs et destinataires.
- Mettre en place une detection d'anomalies non supervisee en complement.

### Evolution de la segmentation client

- Recalculer les clusters periodiquement.
- Comparer la stabilite des segments dans le temps.
- Ajouter une segmentation RFM : Recency, Frequency, Monetary.
- Construire des personas marketing.
- Relier les segments aux taux de conversion des campagnes.
- Proposer des recommandations personnalisees par segment.

### Evolution MLOps

- Ajouter MLflow pour suivre les experiences.
- Ajouter DVC pour versionner les donnees.
- Ajouter Great Expectations pour valider les donnees.
- Ajouter des tests unitaires.
- Ajouter GitHub Actions.
- Dockeriser l'API et le dashboard.
- Ajouter un monitoring de derive avec Evidently AI.
- Mettre en place un cycle de reentrainement automatique.

## 22. Criteres de reussite

Le projet sera considere comme reussi si :

- les donnees sont bien comprises et documentees ;
- les choix de preprocessing sont justifies ;
- les modeles sont compares objectivement ;
- les metriques sont adaptees au probleme ;
- les resultats sont interpretes en langage metier ;
- les visualisations sont claires et professionnelles ;
- les recommandations sont actionnables ;
- le code est propre et organise ;
- le projet est reproductible ;
- une architecture MLOps credible est proposee.

## 23. Synthese finale attendue

La presentation finale doit repondre clairement a ces questions :

1. Quelles sont les principales caracteristiques des transactions frauduleuses ?
2. Quel modele detecte le mieux la fraude et pourquoi ?
3. Quels sont les risques de faux positifs et faux negatifs ?
4. Quels profils clients ont ete identifies ?
5. Quelles actions marketing proposer pour chaque segment ?
6. Comment industrialiser la solution ?
7. Comment suivre la performance et faire evoluer les modeles ?

## 24. Prochaine etape recommandee

La prochaine etape consiste a creer la structure du depot, installer les dependances, puis commencer par deux notebooks :

- `01_eda_fraude.ipynb`
- `03_eda_segmentation.ipynb`

Ces notebooks doivent permettre de valider la comprehension des donnees avant de passer a la modelisation.
