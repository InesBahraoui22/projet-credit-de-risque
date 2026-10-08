# Prédiction des annulations de rendez-vous

Projet de classification pour prédire l'annulation d'un rendez-vous client : **0 = fait**, **1 = annulé client**. L'étude porte sur les dossiers sans co-emprunteur.

## Préparation des données

- Suppression des doublons, des dossiers hors périmètre et des dates incohérentes.
- Traitement des valeurs manquantes et encodage des catégories.
- Création d'indicateurs : charges, recettes, crédits, mensualités et horizons RDV/retraite.

La première version produisait **9 710 observations et 44 colonnes**, avec **29 variables explicatives**. La version fusionnée ajoute cinq indicateurs aux prédicteurs ; ses effectifs sont recalculés à chaque exécution.

## Modèles

Le projet compare quatre familles de modèles :

- **Régression logistique** : une estimation du risque avec pondération des classes.
- **Arbre de décision** : des règles de classification faciles à lire.
- **Random Forest** : une combinaison de plusieurs arbres.
- **XGBoost** : des arbres construits successivement pour améliorer les prédictions.

Les résultats comprennent l'AUC ROC, la précision, le rappel et la matrice de confusion.

## Version fusionnée du notebook et de XGBoost

Ouvrir `Projet_Risque_Credit.ipynb`, ou lancer depuis la racine du dépôt :

```bash
python3 -m pip install -r requirements.txt
python3 -m projet_risque.comparaison_modeles --data ~/Downloads/data.csv
```

Cette version compare la régression logistique, la forêt aléatoire, l'arbre de décision et XGBoost (simple et pondéré), avec les mêmes variables et le même découpage chronologique 80/20. Le seuil est fixé à 0,5 ; le SMOTE n'est pas appliqué. Les scores et courbes sont enregistrés dans `resultats_fusion/`.

Le notebook fourni apporte notamment le jour et l'heure du rendez-vous, la civilité, le reste à vivre et la présence d'un découvert. Les montants manquants des charges et recettes sont remplacés par zéro avant leur somme : c'est une hypothèse de préparation. La médiane, le mode et l'encodage sont appris uniquement sur l'entraînement. Les identifiants, dates brutes, textes libres et `statut_final` sont exclus des prédicteurs. La disponibilité des variables à la prise du rendez-vous reste à confirmer.

Le fichier historique `projet_risque/projet_risque_m.py` est conservé. Pour la comparaison commune, utiliser le nouveau notebook ou la commande ci-dessus. Les anciens scores obtenus sur d'autres découpages ne sont pas directement comparables.

Sur macOS, si XGBoost signale l'absence de `libomp`, installer cette dépendance avec `brew install libomp`.
