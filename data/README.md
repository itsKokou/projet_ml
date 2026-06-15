# Données du projet

## Fichiers attendus

| Fichier | Taille approx. | Description |
|---------|----------------|-------------|
| `raw/detection_fraude.csv` | ~78 Mo | Transactions bancaires (~1 048 576 lignes) |
| `raw/data_cluster.csv` | ~220 Ko | Profils clients (2 240 lignes) |

## Git LFS (fichier fraude)

Le CSV fraude est déclaré dans `.gitattributes` pour **Git LFS** afin d'éviter de alourdir l'historique Git.

### Première installation (contributeur)

```bash
brew install git-lfs   # macOS
git lfs install
git lfs pull
```

### Migration si le fichier est déjà versionné sans LFS

```bash
git lfs install
git lfs track "data/raw/detection_fraude.csv"
git add .gitattributes
git lfs migrate import --include="data/raw/detection_fraude.csv" --everything
```

> La commande `migrate` réécrit l'historique : à coordonner avec l'équipe avant push.

### Alternative sans LFS

Télécharger le fichier manuellement et le placer dans `data/raw/detection_fraude.csv` (même schéma que l'énoncé du cours).
