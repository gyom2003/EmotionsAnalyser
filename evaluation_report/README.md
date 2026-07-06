# Rapport d'évaluation du modèle

Ce dossier contient le rapport d'évaluation du modèle de sentiment (Personne 4) :
matrices de confusion, précision/rappel/F1-score par classe, analyse des biais
et recommandations.

## Contenu

- `evaluate.py` — génère le rapport à partir des données annotées.
- `evaluation_report.pdf` — rapport final (matrices, métriques, analyse, recommandations).
- `confusion_matrix_positive.png` / `confusion_matrix_negative.png` — figures utilisées dans le rapport.

## Régénérer le rapport

```bash
pip install -r requirements-report.txt
python evaluate.py
```

Le script tente d'abord de se connecter à la base MySQL `tweets` (via `db.py`
à la racine du projet). Si la base n'est pas accessible, il utilise
automatiquement le jeu de données seed défini dans `../setup_db.sql`, afin que
le rapport reste reproductible même sans base de données démarrée.

## Méthodologie

Le jeu de données annoté étant restreint, un simple split train/test
laisserait trop peu d'exemples de validation. Le script utilise donc une
validation croisée stratifiée à 5 plis (`StratifiedKFold` + `cross_val_predict`)
sur chacun des deux classifieurs binaires (`positive`, `negative`), avec la
même architecture qu'en production (TF-IDF + `LogisticRegression`). Chaque
tweet est ainsi noté par un modèle qui ne l'a jamais vu à l'entraînement.

## Résultat principal

Avec le jeu de données actuel (~27 tweets), les deux classifieurs ne
parviennent à prédire aucun cas positif hors échantillon (rappel = 0 pour la
classe positive comme pour la classe négative) : le détail est expliqué dans
la section « Analyse des biais et des erreurs » du PDF, avec les
recommandations correspondantes.
