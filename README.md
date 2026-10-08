# Prédiction des annulations de rendez-vous

Projet de classification pour prédire l'annulation d'un rendez-vous client : **0 = fait**, **1 = annulé client**. L'étude porte sur les dossiers sans co-emprunteur.

## Préparation des données

- Suppression des doublons, des dossiers hors périmètre et des dates incohérentes.
- Traitement des valeurs manquantes et encodage des catégories.
- Création d'indicateurs : charges, recettes, crédits, mensualités et horizons RDV/retraite.

Le script produit une base nettoyée de **9 710 observations et 44 colonnes** et utilise **29 variables explicatives** pour ses modèles.

## Modèles

Le projet compare trois modèles :

- **Arbre de décision** : des règles de classification faciles à lire.
- **Random Forest** : une combinaison de plusieurs arbres.
- **XGBoost** : des arbres construits successivement pour améliorer les prédictions.

Les résultats comprennent l'AUC ROC, la précision, le rappel et la matrice de confusion.

## Lancer le projet

Pour lancer le script de nettoyage, de forêt aléatoire et d'arbre de décision, installer les dépendances :

```bash
python3 -m pip install pandas scikit-learn matplotlib dtale
```

