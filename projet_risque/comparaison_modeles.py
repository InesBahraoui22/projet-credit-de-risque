"""Comparer les modèles du notebook et les deux variantes XGBoost."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                            recall_score, f1_score, classification_report,
                            confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay)
from xgboost import XGBClassifier
from .preparation_commune import preparer_donnees


def comparer(chemin_csv, sortie):
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    data, train, test, y_train, y_test = preparer_donnees(chemin_csv)
    variables = json.loads(Path(__file__).with_name('variables.json').read_text())
    categories = ['situation_fam_emp', 'contrat_emp', 'genre_emp', 'jour_rdv']
    nombres = [c for c in variables if c not in categories]
    X_train, X_test = train[variables].copy(), test[variables].copy()
    # Des tableaux numériques évitent les caractères interdits dans les noms XGBoost.
    for X in [X_train, X_test]:
        X[nombres] = X[nombres].astype(float)
    poids = int((y_train == 0).sum()) / int((y_train == 1).sum())
    modeles = {
        'logistique': LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42),
        'foret': RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=2),
        'arbre': DecisionTreeClassifier(class_weight='balanced', max_depth=4, min_samples_leaf=50, random_state=42),
        'xgboost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                eval_metric='logloss', random_state=42, n_jobs=2),
        'xgboost_equilibre': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                                scale_pos_weight=poids, eval_metric='logloss', random_state=42, n_jobs=2),
    }
    resultats = []
    pipelines = {}
    figure = Figure(figsize=(12, 5), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for nom, estimateur in modeles.items():
        numerique = [('imputation', SimpleImputer(strategy='median'))]
        if nom == 'logistique':
            numerique.append(('echelle', StandardScaler()))
        preparation = ColumnTransformer([
            ('categories', Pipeline([
                ('imputation', SimpleImputer(strategy='most_frequent')),
                ('encodage', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), categories),
            ('nombres', Pipeline(numerique), nombres),
        ])
        modele = Pipeline([('preparation', preparation), ('modele', estimateur)])
        modele.fit(X_train, y_train)
        probabilites = modele.predict_proba(X_test)[:, 1]
        assert np.isfinite(probabilites).all()
        predictions = (probabilites >= 0.5).astype(int)
        resultats.append({
            'modele': nom, 'auc_roc': roc_auc_score(y_test, probabilites),
            'average_precision': average_precision_score(y_test, probabilites),
            'precision': precision_score(y_test, predictions, zero_division=0),
            'rappel': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
        })
        rapport = classification_report(y_test, predictions, labels=[0, 1],
                                        target_names=['Fait', 'Annulé'], zero_division=0)
        matrice = confusion_matrix(y_test, predictions, labels=[0, 1])
        (sortie / f'{nom}.txt').write_text(
            f'Seuil fixé à 0.5\n{rapport}\nMatrice : lignes réelles, colonnes prédites (fait, annulé)\n{matrice}\n',
            encoding='utf-8')
        RocCurveDisplay.from_predictions(y_test, probabilites, ax=axes[0], name=nom)
        PrecisionRecallDisplay.from_predictions(y_test, probabilites, ax=axes[1], name=nom)
        pipelines[nom] = modele
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].axhline(y_test.mean(), linestyle='--', color='gray', label='Prévalence test')
    axes[1].legend()
    figure.savefig(sortie / 'comparaison.png', dpi=150)
    tableau = pd.DataFrame(resultats)
    tableau.to_csv(sortie / 'comparaison.csv', index=False)
    bilan = {
        'lignes_nettoyees': len(data), 'train': len(train), 'test': len(test),
        'variables': len(variables), 'taux_annulation_train': float(y_train.mean()),
        'taux_annulation_test': float(y_test.mean()), 'poids_xgboost': poids,
        'date_debut_test': str(test['date_aboutisant_azur'].min()),
        'seuil': 0.5, 'smote': False,
    }
    (sortie / 'bilan.json').write_text(json.dumps(bilan, indent=2, ensure_ascii=False)+'\n')
    print(tableau.round(4).to_string(index=False))
    return tableau, pipelines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path.home() / 'Downloads' / 'data.csv')
    parser.add_argument('--sortie', type=Path, default=Path('resultats_fusion'))
    args = parser.parse_args()
    comparer(args.data, args.sortie)
