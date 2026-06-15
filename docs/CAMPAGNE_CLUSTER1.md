# Protocole de campagne test — Cluster 1 (promo digitaux)

Segment cible : **Cluster 1** — « Chasseurs de promotions digitaux »  
Taille : **42 clients** (1,9 % de la base) — segment statistiquement fragile, validation terrain indispensable.

## 1. Hypothèse à valider

Les clients du cluster 1 présentent un **ROI marketing supérieur** à une campagne générique lorsqu'on leur propose des offres web personnalisées (coupons, flash sales).

## 2. Design expérimental proposé

| Élément | Spécification |
|---------|---------------|
| Type | A/B test |
| Groupe A (témoin) | 21 clients cluster 1 — pas de promotion |
| Groupe B (traitement) | 21 clients cluster 1 — coupon web -15 %, durée 14 jours |
| Durée | 4 semaines (1 semaine warmup + 2 semaines campagne + 1 semaine observation) |
| Canal | Email + notification web (canal dominant du segment) |

## 3. KPIs de succès

| KPI | Cible | Seuil d'échec |
|-----|-------|---------------|
| Taux de conversion promo (B) | ≥ 25 % | < 10 % |
| Panier moyen (B vs A) | +10 % vs témoin | Pas de différence significative |
| ROI campagne | > 150 % | < 100 % |
| Taux de réachat à 30 jours | ≥ 15 % | < 5 % |

## 4. Risques et garde-fous

- **Échantillon très petit (n=42)** : résultats indicatifs, pas conclusifs — prévoir test élargi si prometteur.
- **Cannibalisation des marges** : plafonner la remise à 15 % et exclure les produits premium.
- **Instabilité du segment** : recalculer les clusters trimestriellement avant chaque campagne.

## 5. Décision post-campagne

| Résultat | Action |
|----------|--------|
| KPIs au-dessus des cibles | Industrialiser les campagnes coupon web cluster 1 |
| Résultats mitigés | Fusionner cluster 1 avec cluster 0 ou 3 pour les actions promo |
| Échec | Retirer le segment des ciblages prioritaires ; réviser k ou variables |

## 6. Données à collecter

- ID client, groupe A/B, date d'envoi, ouverture email, clic, achat, montant, marge.
- Comparaison avec `Response` historique pour calibrer les attentes.
