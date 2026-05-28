# Rapport technique - Projet Machine Learning M2 CDSD

## 1. Resume executif

Ce projet traite deux cas d'usage complementaires :

- la detection de transactions frauduleuses a partir d'un jeu de donnees transactionnel ;
- la segmentation de clients a partir de donnees marketing et comportementales.

La partie fraude est un probleme de classification supervisee fortement desequilibre. Le meilleur modele obtenu est un modele XGBoost, selectionne sur la PR-AUC, avec une PR-AUC de 0.9910 sur le jeu de test.

La partie segmentation est un probleme non supervise. Le meilleur modele exploitable pour le dashboard et l'inference est un Gaussian Mixture Model avec 4 clusters. Les segments obtenus sont interpretables et peuvent etre relies a des actions marketing : clients faibles depenses, clients promotionnels/digitaux, clients premium tres reactifs et clients a forte valeur.

Le projet est structure pour pouvoir evoluer vers une approche MLOps : scripts reutilisables, modeles sauvegardes, tests automatises, API FastAPI et dashboard Streamlit.

## 2. Objectifs du projet

### 2.1 Detection de fraude

L'objectif est de predire si une transaction est frauduleuse ou non via la variable cible `isFraud`.

Les enjeux principaux sont :

- detecter le maximum de fraudes ;
- limiter les fausses alertes ;
- choisir un seuil de decision adapte a la capacite de traitement metier ;
- expliquer les predictions afin de rendre le modele utilisable par des analystes fraude.

### 2.2 Segmentation client

L'objectif est d'identifier des groupes de clients ayant des comportements similaires.

Les enjeux principaux sont :

- obtenir des segments lisibles ;
- comprendre la valeur et le comportement de chaque segment ;
- proposer des actions marketing adaptees ;
- construire une base reutilisable dans un dashboard ou une campagne.

## 3. Donnees utilisees

### 3.1 Donnees fraude

Fichier utilise :

- `data/raw/detection_fraude.csv`

Variables importantes :

- `step` : unite temporelle de la transaction ;
- `type` : type de transaction ;
- `amount` : montant transfere ;
- `oldbalanceOrg`, `newbalanceOrig` : soldes de l'emetteur ;
- `oldbalanceDest`, `newbalanceDest` : soldes du destinataire ;
- `isFraud` : variable cible ;
- `isFlaggedFraud` : indicateur de signalement automatique.

Volumetrie :

- environ 1 048 575 transactions ;
- taux de fraude proche de 0.109 % ;
- probleme fortement desequilibre.

Point metier important :

- les fraudes observees sont concentrees sur les transactions `TRANSFER` et `CASH_OUT`.

### 3.2 Donnees segmentation

Fichier utilise :

- `data/raw/data_cluster.csv`

Variables importantes :

- variables demographiques : age, education, situation matrimoniale, revenu ;
- variables de consommation : depenses par categorie ;
- variables de canal : achats web, catalogue, magasin, promotions ;
- variables marketing : acceptation des campagnes et reponse.

Volumetrie :

- 2 240 clients ;
- quelques valeurs manquantes sur `Income`.

## 4. Methodologie generale

La demarche suivie est la suivante :

1. Chargement des donnees.
2. Nettoyage de base.
3. Feature engineering.
4. Construction de pipelines.
5. Entrainement de plusieurs modeles.
6. Evaluation comparative.
7. Sauvegarde du meilleur modele.
8. Interpretation metier.
9. Mise a disposition via dashboard et API.

Le code est organise dans les dossiers suivants :

- `src/data/` : chargement et preprocessing ;
- `src/features/` : creation des variables ;
- `src/models/` : entrainement et evaluation ;
- `src/api/` : API de prediction ;
- `dashboard/` : application Streamlit ;
- `models/` : modeles et resultats sauvegardes ;
- `reports/` : rapports, plans et figures.

## 5. Preprocessing et feature engineering

### 5.1 Fraude

Les principales transformations sont :

- nettoyage des doublons ;
- remplacement de `oldbalanceOrg = 0` par une valeur manquante pour eviter les divisions par zero ;
- suppression des identifiants directs `nameOrig` et `nameDest` dans la baseline ;
- suppression de `isFlaggedFraud` pour eviter une dependance excessive a un signal deja issu d'un systeme de detection ;
- encodage de la variable `type` ;
- standardisation des variables numeriques ;
- imputation des valeurs manquantes.

Variables creees :

- `origin_balance_diff` : difference entre ancien et nouveau solde emetteur ;
- `origin_error` : incoherence entre mouvement du solde emetteur et montant ;
- `dest_balance_diff` : difference entre nouveau et ancien solde destinataire ;
- `dest_error` : incoherence entre mouvement du solde destinataire et montant ;
- `is_transfer_or_cashout` : indicateur des types de transactions a risque ;
- `is_zero_newbalance_origin` : indicateur de solde emetteur vide apres transaction ;
- `is_zero_oldbalance_dest` : indicateur de destinataire avec solde initial nul ;
- `amount_to_oldbalance_ratio` : ratio entre montant et solde initial ;
- `step_bucket` : regroupement temporel simple.

Ces variables permettent de capturer les incoherences comptables et les comportements typiques des fraudes.

### 5.2 Segmentation

Les principales transformations sont :

- imputation de `Income` par la mediane ;
- suppression de colonnes constantes comme `Z_CostContact` et `Z_Revenue` ;
- encodage des variables categorielles ;
- standardisation des variables numeriques ;
- suppression des variables non pertinentes pour l'apprentissage des segments : `ID`, `Response`, `Complain`, `Dt_Customer`.

Variables creees :

- `Age` ;
- `Customer_Tenure_days` ;
- `Total_Spending` ;
- `Total_Purchases` ;
- ratios par canal d'achat ;
- `Children` ;
- `Campaign_Acceptance_Total`.

Ces variables transforment les donnees brutes en indicateurs metier plus lisibles.

## 6. Modelisation fraude

### 6.1 Separation des donnees

Les donnees ont ete separees en trois parties :

- entrainement ;
- validation ;
- test.

La separation est stratifiee afin de conserver la proportion de fraudes dans chaque sous-ensemble.

Le jeu de test contient 157 287 transactions avec un taux de fraude de 0.1087 %, soit environ 171 fraudes.

### 6.2 Modeles testes

Trois modeles ont ete compares :

- regression logistique ;
- random forest ;
- XGBoost.

Pour tenir compte du desequilibre de classe :

- la regression logistique utilise `class_weight="balanced"` ;
- la random forest utilise `class_weight="balanced_subsample"` ;
- XGBoost utilise `scale_pos_weight`.

Un seuil de decision est selectionne sur le jeu de validation avec une contrainte de rappel minimal.

### 6.3 Resultats

| Modele | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Seuil |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regression logistique | 0.9993 | 0.6739 | 0.7251 | 0.6986 | 0.9986 | 0.7674 | 0.98 |
| Random Forest | 0.99998 | 1.0000 | 0.9825 | 0.9912 | 0.9941 | 0.9864 | 0.52 |
| XGBoost | 0.99998 | 1.0000 | 0.9825 | 0.9912 | 0.9981 | 0.9910 | 0.84 |

Le modele retenu est XGBoost car il obtient la meilleure PR-AUC.

### 6.4 Interpretation des resultats fraude

L'accuracy est tres elevee pour tous les modeles, mais cette metrique est peu informative dans un contexte tres desequilibre. Un modele qui predit toujours "non fraude" aurait deja une accuracy tres elevee.

Les metriques importantes sont donc :

- recall : capacite a retrouver les fraudes ;
- precision : proportion des alertes qui sont vraiment frauduleuses ;
- F1-score : compromis entre precision et recall ;
- PR-AUC : qualite du classement dans un contexte de classe rare.

Avec XGBoost :

- le recall est de 0.9825 ;
- la precision est de 1.0000 ;
- sur environ 171 fraudes dans le test, le modele en detecte environ 168 et en manque environ 3 ;
- aucun faux positif n'est observe au seuil retenu dans ce test.

Ces resultats sont excellents, mais doivent etre interpretes avec prudence :

- le jeu de donnees est probablement tres structure ;
- les variables de solde peuvent contenir des signaux tres forts ;
- il faut verifier la stabilite sur de nouvelles donnees ;
- il faut surveiller la derive des comportements frauduleux.

## 7. Segmentation client

### 7.1 Modeles testes

Les modeles suivants ont ete compares pour plusieurs nombres de clusters :

- K-Means ;
- Agglomerative Clustering ;
- Gaussian Mixture Model.

Les valeurs de `k` testees sont 3, 4, 5 et 6.

### 7.2 Metriques utilisees

Les metriques principales sont :

- Silhouette Score : mesure la separation et la cohesion des clusters ;
- Davies-Bouldin Score : plus il est faible, meilleure est la separation ;
- Calinski-Harabasz Score : mesure le rapport entre dispersion inter-clusters et intra-clusters.

### 7.3 Resultats clustering

Le modele retenu est `gmm_k4`.

Metriques :

- silhouette : 0.2100 ;
- Davies-Bouldin : 2.4230 ;
- Calinski-Harabasz : 305.4055 ;
- nombre de clusters : 4.

Le score silhouette reste modere. Cela signifie que les groupes ne sont pas parfaitement separes. C'est frequent en segmentation client, car les comportements marketing sont souvent progressifs plutot que strictement separes.

Le choix de 4 clusters est pertinent car il produit des segments lisibles et exploitables.

## 8. Profils clients obtenus

### Cluster 0 - Clients faible valeur / faible engagement

Taille :

- 1 132 clients.

Profil moyen :

- revenu moyen : 36 571.82 ;
- age moyen : 55.41 ans ;
- depense totale moyenne : 123.77 ;
- nombre total d'achats : 8.79 ;
- acceptation campagne : 0.15.

Interpretation :

- clients nombreux mais peu depensiers ;
- faible engagement marketing ;
- achats limites sur tous les canaux.

Actions recommandees :

- campagnes de reactivation simples ;
- offres d'entree de gamme ;
- coupons limites dans le temps ;
- communication orientee prix et praticite.

### Cluster 1 - Clients promotionnels et digitaux

Taille :

- 42 clients.

Profil moyen :

- revenu moyen : 49 681.74 ;
- age moyen : 62.33 ans ;
- depense totale moyenne : 467.48 ;
- nombre total d'achats : 20.00 ;
- achats web : 6.98 ;
- achats promotionnels : 5.71 ;
- acceptation campagne : 0.86.

Interpretation :

- petit segment mais assez actif ;
- forte sensibilite aux promotions ;
- utilisation importante du canal web ;
- bonne reactivite aux campagnes.

Actions recommandees :

- offres promotionnelles personnalisees ;
- campagnes email ou web ciblees ;
- bundles et reductions conditionnelles ;
- tests A/B sur mecanismes de couponing.

### Cluster 2 - Clients premium tres reactifs

Taille :

- 175 clients.

Profil moyen :

- revenu moyen : 80 575.03 ;
- age moyen : 57.01 ans ;
- depense totale moyenne : 1 584.17 ;
- nombre total d'achats : 20.81 ;
- achats catalogue : 5.93 ;
- achats magasin : 8.28 ;
- acceptation campagne : 2.58.

Interpretation :

- segment a tres forte valeur ;
- revenu eleve ;
- depenses tres importantes ;
- tres forte reactivite aux campagnes ;
- canal magasin et catalogue importants.

Actions recommandees :

- offres premium ;
- programme VIP ;
- avant-premieres ;
- recommandations personnalisees ;
- relation client renforcee.

### Cluster 3 - Clients forte valeur stables

Taille :

- 891 clients.

Profil moyen :

- revenu moyen : 66 696.41 ;
- age moyen : 59.26 ans ;
- depense totale moyenne : 1 032.57 ;
- nombre total d'achats : 21.17 ;
- achats magasin : 8.27 ;
- achats catalogue : 4.63 ;
- acceptation campagne : 0.38.

Interpretation :

- segment important en taille et valeur ;
- clients fideles avec fortes depenses ;
- engagement commercial fort mais reponse campagne plus faible que le cluster 2 ;
- canal magasin tres present.

Actions recommandees :

- fidelisation ;
- cross-sell ;
- offres personnalisees mais moins promotionnelles ;
- parcours omnicanal magasin/catalogue/web.

## 9. Recommandations metier

### 9.1 Fraude

Recommandations :

- utiliser XGBoost comme modele principal ;
- conserver le seuil de decision optimise, mais le recalibrer selon la capacite des analystes fraude ;
- suivre le recall pour limiter les fraudes manquees ;
- suivre la precision pour eviter une surcharge d'alertes ;
- analyser manuellement les faux negatifs ;
- mettre en place un suivi par type de transaction.

Politique possible :

- score tres eleve : blocage ou verification forte ;
- score intermediaire : revue humaine ;
- score faible : transaction normale.

### 9.2 Marketing

Recommandations :

- Cluster 0 : reactivation et offres accessibles.
- Cluster 1 : promotions digitales et coupons.
- Cluster 2 : premium, VIP, personnalisation forte.
- Cluster 3 : fidelisation et developpement de valeur.

Le clustering doit etre utilise comme un outil d'aide a la decision. Les segments doivent etre valides par les equipes marketing avant lancement de campagnes.

## 10. Industrialisation et MLOps

### 10.1 Architecture actuelle

Le projet contient deja :

- scripts d'entrainement ;
- modeles sauvegardes avec `joblib` ;
- resultats en JSON et CSV ;
- dashboard Streamlit ;
- API FastAPI ;
- tests automatiques avec pytest.

### 10.2 Architecture cible

Architecture recommandee :

1. Ingestion des donnees.
2. Validation du schema.
3. Preprocessing.
4. Feature engineering.
5. Entrainement.
6. Evaluation.
7. Sauvegarde et versioning du modele.
8. Deploiement API/dashboard.
9. Monitoring.
10. Re-entrainement.

### 10.3 Monitoring

Pour la fraude :

- taux d'alertes ;
- distribution des scores ;
- precision et recall quand les labels reels sont disponibles ;
- derive des montants ;
- derive des types de transaction ;
- temps de reponse API.

Pour la segmentation :

- taille des clusters ;
- evolution des revenus moyens par segment ;
- evolution des depenses moyennes ;
- stabilite des clusters ;
- apparition de nouveaux profils clients.

### 10.4 Versioning

Elements a versionner :

- code source avec Git ;
- dependances avec `requirements.txt` ;
- modeles avec MLflow ou un registre de modeles ;
- metriques avec fichiers JSON/CSV ;
- donnees avec DVC ou Git LFS si elles sont trop volumineuses.

## 11. Limites du projet

### 11.1 Limites fraude

- Le taux de fraude est extremement faible.
- Les resultats sont tres eleves et doivent etre confirmes sur donnees futures.
- Les fraudeurs changent de comportement, ce qui peut provoquer de la derive.
- Les variables de solde peuvent etre tres predictives dans ce jeu de donnees, mais leur qualite doit etre garantie en production.
- Le cout metier d'un faux positif et d'un faux negatif n'est pas encore chiffre.

### 11.2 Limites segmentation

- Le score silhouette est modere.
- Les clusters doivent etre valides par le metier.
- Les donnees ne contiennent pas d'historique long de campagnes.
- Les segments peuvent evoluer avec le temps.
- Certains clusters sont petits, notamment le cluster 1.

## 12. Pistes d'amelioration

### 12.1 Ameliorations fraude

- Ajouter une courbe precision-rappel dans le rapport final.
- Tester LightGBM et CatBoost.
- Optimiser les hyperparametres avec Optuna.
- Ajouter SHAP pour expliquer les predictions.
- Construire des features reseau entre emetteurs et destinataires.
- Mettre en place un systeme de seuils multiples.

### 12.2 Ameliorations segmentation

- Ajouter une segmentation RFM.
- Tester PCA avant clustering.
- Ajouter t-SNE ou UMAP pour visualisation.
- Nommer officiellement les segments avec les equipes marketing.
- Evaluer la stabilite des clusters par bootstrap.
- Relier les clusters a la conversion des campagnes.

### 12.3 Ameliorations MLOps

- Ajouter MLflow pour tracer les experiences.
- Ajouter DVC pour versionner les donnees.
- Ajouter Great Expectations pour valider les schemas.
- Ajouter GitHub Actions.
- Dockeriser l'API et le dashboard.
- Ajouter Evidently AI pour le monitoring de derive.

## 13. Conclusion

Le projet dispose maintenant d'une base solide :

- un pipeline de detection de fraude performant ;
- une segmentation client exploitable ;
- des modeles sauvegardes ;
- un dashboard interactif ;
- une API ;
- des tests automatiques ;
- une organisation compatible avec une evolution MLOps.

La prochaine etape prioritaire est de renforcer l'interpretabilite :

- SHAP pour la fraude ;
- visualisations de profils pour les clusters ;
- rapport final avec figures ;
- presentation claire orientee decision metier.
