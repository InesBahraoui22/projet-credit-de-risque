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



def preparer_donnees(chemin_csv):
    import pandas as pd
    import ast
    import math
    from pathlib import Path

    # Chargement de la base ; la première colonne du CSV contient l'index.
    chemin_data = Path(chemin_csv).expanduser()
    data = pd.read_csv(chemin_data, index_col=0, low_memory=False)

    print(f"Base chargée : {data.shape[0]} lignes et {data.shape[1]} colonnes")

    # Exclure les états demandés. Dans le CSV, « refus avant rdv » est
    # écrit « refus avant rendezvous ». « annuler client » reste conservé.
    etats_a_supprimer = [
        "refus entreprise", "refus client", "refus avant rendezvous",
        "refus avant rdv", "pas fait",
    ]
    etat_normalise = data["etat"].astype("string").str.strip().str.lower()
    lignes_a_supprimer = etat_normalise.isin(etats_a_supprimer)
    nb_supprimees = int(lignes_a_supprimer.sum())
    data = data.loc[~lignes_a_supprimer].copy()

    print(f"Lignes supprimées : {nb_supprimees} ; lignes restantes : {len(data)}")

    # Exclure les dossiers locataires après le filtre sur les états.
    locataires = data["type_dossier"].astype("string").str.strip().str.lower().eq("locataire").fillna(False)
    nb_locataires = int(locataires.sum())
    data = data.loc[~locataires].copy()
    print(f"Locataires supprimés : {nb_locataires} ; lignes restantes : {len(data)}")

    # Supprimer les doublons exacts sur toutes les colonnes, sans comparer l'index.
    # Un même dossier peut garder plusieurs lignes si ses rendez-vous diffèrent.
    nb_doublons = int(data.duplicated().sum())
    data = data.drop_duplicates(keep="first").copy()
    print(f"Doublons supprimés : {nb_doublons} ; lignes restantes : {len(data)}")

    # Colonnes retirées pour le nettoyage (noms tels qu'ils figurent dans le CSV).
    colonnes_a_supprimer = [
        "statut_final",
        "affectation",
        "duree",
        "type_traitement_dossier",
        "type_support",
        "profession_emp",
        "conso_date_debut",
        "conso_date_fin",
        "conso_duree",
        "anciennete_emp",
        "contrat_mariage_emp",
        "csp_emp",
        "immo_date_debut",
        "immo_date_fin",
        "immo_duree",
        "immo_libelle",
        "valeur_gage",
        "valeur_residence_principale",
        "valeur_bien_immobilier",
        "valeur_produit_epargne_placement",
        "date_acquisition",
        "valeur_acquisition",
        "nb_etablissement_ficp",
        "conso_conserve",
        "immo_conserve",
        "rc_id",
        "commentaire",
        "segmentation",
        "hebergement_gratuit_emp",
        "type_invalidite_coemp",
        "date_deja_rachat",
        "type_invalidite_emp",
        "contrat_mariage_coemp",
        "hebergement_gratuit_coemp",
    ]
    data = data.drop(columns=colonnes_a_supprimer)
    print(f"Colonnes supprimées : {len(colonnes_a_supprimer)} ; colonnes restantes : {data.shape[1]}")

    # Garder uniquement les dossiers sans co-emprunteur.
    # Un identifiant coemp_id absent indique ici l'absence de co-emprunteur.
    avec_coemprunteur = data["coemp_id"].notna()
    print(f"Dossiers avec co-emprunteur supprimés : {avec_coemprunteur.sum()}")
    data = data.loc[~avec_coemprunteur].copy()
    colonnes_coemp = [colonne for colonne in data.columns if "coemp" in colonne]
    data = data.drop(columns=colonnes_coemp)
    print(f"Dossiers sans co-emprunteur conservés : {len(data)}")

    # Une échéance de retard non nulle suffit pour indiquer un retard.
    def a_un_retard(valeur):
        if pd.isna(valeur):
            return False
        montants = ast.literal_eval(valeur) if isinstance(valeur, str) else valeur
        if not isinstance(montants, list):
            montants = [montants]
        return any(pd.notna(montant) and montant != 0 for montant in montants)

    for colonne in ["conso_echeance_de_retard", "immo_echeance_de_retard"]:
        data[colonne] = data[colonne].map(a_un_retard)

    # Vrai si au moins un des trois fichages est vrai ; faux si les trois sont faux.
    data["fichage_BDF"] = (
        data["fichage_FCC_carte"].astype("boolean")
        | data["fichage_FCC_cheque"].astype("boolean")
        | data["fichage_FICP"].astype("boolean")
    )
    data = data.drop(columns=["fichage_FCC_carte", "fichage_FCC_cheque", "fichage_FICP"])

    # Faux si le montant est absent ou nul, vrai sinon.
    for colonne in ["tresorerie", "tresorerie_sur_facture"]:
        data[colonne] = data[colonne].fillna(0).ne(0)

    print("\nÉtats conservés :")
    print(data["etat"].value_counts(dropna=False))

    # Horizon du rendez-vous : délai en jours depuis la prise de rendez-vous.
    # On garde les dates originales ; une date illisible donne une valeur manquante.
    date_prise_rdv = pd.to_datetime(data["date_aboutisant_azur"], format="%Y-%m-%d", errors="coerce")
    date_rendez_vous = pd.to_datetime(data["date_rdv"], format="%Y-%m-%d", errors="coerce")
    data["horizon_rdv"] = (date_rendez_vous - date_prise_rdv).dt.days

    # Années restantes avant la retraite à la date de prise du rendez-vous.
    # Une valeur négative signifie que la date de retraite indiquée est déjà passée.
    # Une date absente ou illisible reste manquante ; 365.25 convertit les jours
    # en années approximatives, sans arrondir les valeurs utilisées pour l'analyse.
    date_retraite = pd.to_datetime(data["date_retraite_emp"], format="%Y-%m-%d", errors="coerce")
    data["horizon_retraite"] = (date_retraite - date_prise_rdv).dt.days / 365.25

    # Supprimer les rendez-vous antérieurs à leur date de prise.
    rdv_avant_prise = data["horizon_rdv"].lt(0)
    nb_rdv_supprimes = int(rdv_avant_prise.sum())
    data = data.loc[~rdv_avant_prise].copy()
    print(f"Rendez-vous antérieurs supprimés : {nb_rdv_supprimes} ; lignes restantes : {len(data)}")

    # Les capitaux restants dus (CRD) sont stockés comme des listes de montants.
    # Exemple : "[1000, 2500]" -> 2 crédits et 3500 de capital restant dû.
    # Une liste vide donne 0 crédit et 0 de cumul ; une donnée absente/illisible
    # reste manquante. On ne confond pas montant manquant et montant nul.
    def resumer_credits(valeur):
        if pd.isna(valeur):
            return float("nan"), float("nan")
        try:
            montants = ast.literal_eval(valeur)
        except (ValueError, SyntaxError):
            return float("nan"), float("nan")
        if not isinstance(montants, list):
            return float("nan"), float("nan")
        nombre = len(montants)
        # Un montant inconnu empêche de calculer un total complet, mais pas
        # de compter les entrées de la liste.
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in montants):
            return nombre, float("nan")
        return nombre, sum(montants)

    for famille in ["conso", "immo"]:
        statistiques_credits = data[f"{famille}_crd"].map(resumer_credits)
        data[f"nombre_credits_{famille}"] = statistiques_credits.map(lambda x: x[0]).astype("Int64")
        data[f"cumul_crd_{famille}"] = statistiques_credits.map(lambda x: x[1])
        print(f"Cumuls CRD {famille} non calculables : {data[f'cumul_crd_{famille}'].isna().sum()}")

    # Somme des mensualités des crédits enregistrés, sans filtre sur « conserve ».
    # Une liste vide vaut 0 ; un montant absent ou illisible rend la somme inconnue.
    for famille in ["conso", "immo"]:
        data[f"total_mensualites_{famille}"] = data[f"{famille}_mensualite"].map(
            lambda valeur: resumer_credits(valeur)[1]
        )
    # L'addition conserve NA si l'un des deux totaux est inconnu.
    data["total_mensualites_credits"] = (
        data["total_mensualites_conso"] + data["total_mensualites_immo"]
    )

    # Charges et recette selon les postes retenus pour le projet.
    # L'addition conserve une valeur manquante si un poste est inconnu.
    # Hypothèse du notebook : un poste financier manquant contribue pour zéro.
    cols_charges = ["pension_versee_emp", "loyer_emp", "charges_loyer_emp",
                    "total_mensualites_credits", "charge_recurrente_emp", "charge_courante_emp"]
    cols_recettes = ["salaire_emp", "rev_foncier_emp", "apl_emp", "pension_alimentaire_emp",
                     "allocation_familiale_emp", "pension_invalidite_emp"]
    data["charges"] = data[cols_charges].fillna(0).sum(axis=1)
    data["recette"] = data[cols_recettes].fillna(0).sum(axis=1)

    # Faux si aucune commission n'est renseignée ou si le nombre vaut zéro.
    data["nombre_commissions_intervention"] = data["nombre_commissions_intervention"].fillna(0).ne(0)

    # Faux si le montant est absent ou nul, vrai sinon.
    for colonne in ["retard_loyer_emp", "avis_a_tiers_detenteurs_emp", "autre_dette_emp"]:
        data[colonne] = data[colonne].fillna(0).ne(0)

    # Retirer les détails après avoir calculé les totaux et l'horizon retraite.
    data = data.drop(columns=[
        "apl_emp",
        "pension_alimentaire_emp",
        "allocation_familiale_emp",
        "date_retraite_emp",
        "saisie_sur_salaire_emp",
        "salaire_emp",
        "rev_foncier_emp",
        "conge_parental_emp",
        "nature_de_projet",
        "conso_type",
        "conso_mensualite",
        "conso_crd",
        "conso_taux",
        "immo_taux",
        "immo_crd",
        "immo_garantie",
        "immo_mensualite",
        "loyer_emp",
        "charges_loyer_emp",
        "charge_recurrente_emp",
        "charge_courante_emp",
        "charge_future_eventuelle_emp",
        "pension_versee_emp",
    ])

    # Un code fixe par contrat ; INCONNU et inconnu désignent la même catégorie.
    codage_contrat = {
        "CDI": 0,
        "CDD": 1,
        "INTERIM": 2,
        "RETRAITE": 3,
        "TNS": 4,
        "PROFESSION LIBERALE": 5,
        "SANS EMPLOI": 6,
        "INCONNU": 7,
    }
    contrats = data["contrat_emp"].fillna("INCONNU").str.strip().str.upper()
    if not contrats.isin(codage_contrat).all():
        raise ValueError("Une nouvelle catégorie de contrat doit être ajoutée au codage.")
    data["contrat_emp"] = contrats.map(codage_contrat).astype(int)
    print("Codage contrat_emp :", codage_contrat)

    # Remplacer les montants manquants par zéro pour ces trois colonnes.
    for colonne in ["dette_famille_ami_emp", "retard_impot_emp", "decouvert_emp"]:
        data[colonne] = data[colonne].fillna(0)

    # Compléter l'horizon retraite avec l'hypothèse d'un départ à 64 ans.
    age_retraite_legal = 64
    naissance = pd.to_datetime(data["date_naissance_emp"], errors="coerce")
    date_dossier = pd.to_datetime(data["date_aboutisant_azur"], errors="coerce")
    age_au_t0 = (date_dossier - naissance).dt.days / 365.25
    horizon_theorique = age_retraite_legal - age_au_t0
    data["horizon_retraite"] = data["horizon_retraite"].fillna(horizon_theorique)
    print("NA horizon retraite après calcul théorique :", data["horizon_retraite"].isna().sum())

    # Variables supplémentaires du notebook, connues au moment du rendez-vous prévu.
    data["jour_rdv"] = pd.to_datetime(data["date_rdv"], errors="coerce").dt.dayofweek
    data["heure_rdv_num"] = pd.to_datetime(data["heure_rdv_debut"], format="%H:%M:%S", errors="coerce").dt.hour
    data["genre_emp"] = data["intitule_emp"].astype("string").str.strip().str.lower().map({"madame": "Madame", "monsieur": "Monsieur"}).fillna("Inconnu")
    data["reste_a_vivre"] = data["recette"] - data["charges"]
    data["est_a_decouvert"] = data["decouvert_emp"].gt(0)

    # Découpage chronologique : environ 80 % pour apprendre, 20 % pour tester.
    dates = pd.to_datetime(data["date_aboutisant_azur"], errors="coerce")
    if dates.isna().any() or data["id_dossier"].isna().any():
        raise ValueError("Le découpage nécessite une date et un identifiant pour chaque ligne.")

    # Garder tous les rendez-vous d'une même date du même côté.
    effectifs_par_date = dates.value_counts().sort_index()
    effectifs_avant_date = effectifs_par_date.cumsum().shift(fill_value=0)
    date_coupure = (effectifs_avant_date - 0.80 * len(data)).abs().idxmin()
    data_train = data.loc[dates < date_coupure].copy()
    data_test = data.loc[dates >= date_coupure].copy()

    # Écarter de l'entraînement les dossiers également présents dans le test.
    dossiers_communs = data_train["id_dossier"].isin(data_test["id_dossier"])
    nb_lignes_ecartees = int(dossiers_communs.sum())
    data_train = data_train.loc[~dossiers_communs].copy()
    data_train = data_train.sort_values("date_aboutisant_azur")
    data_test = data_test.sort_values("date_aboutisant_azur")

    assert not set(data_train["id_dossier"]) & set(data_test["id_dossier"])
    assert data_train["date_aboutisant_azur"].max() < data_test["date_aboutisant_azur"].min()

    # La cible : 0 = fait, 1 = annulé client.
    codage_etat = {"fait": 0, "annuler client": 1}
    y_train = data_train["etat"].str.strip().str.lower().map(codage_etat)
    y_test = data_test["etat"].str.strip().str.lower().map(codage_etat)
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Un état ne correspond ni à fait ni à annuler client.")

    print(f"\nDébut du test : {date_coupure:%d/%m/%Y}")
    print(f"Lignes écartées de l'entraînement pour éviter les dossiers communs : {nb_lignes_ecartees}")
    for nom, groupe, cible in [("Entraînement", data_train, y_train),
                               ("Test", data_test, y_test)]:
        print(f"{nom} : {len(groupe)} lignes ; {100 * cible.mean():.2f} % d'annulations")

    # Apprendre les valeurs de remplacement uniquement sur l'entraînement.
    mode_situation = data_train["situation_fam_emp"].mode()
    mediane_horizon = data_train["horizon_retraite"].median()
    if mode_situation.empty or pd.isna(mediane_horizon):
        raise ValueError("Pas assez de valeurs renseignées dans l'entraînement pour imputer.")
    situation_frequente = mode_situation.iloc[0]

    # Utiliser les mêmes valeurs pour la base affichée, l'entraînement et le test.
    for groupe in [data, data_train, data_test]:
        groupe["situation_fam_emp"] = groupe["situation_fam_emp"].fillna(situation_frequente)
        groupe["horizon_retraite"] = groupe["horizon_retraite"].fillna(mediane_horizon)
        groupe.drop(columns=["date_naissance_emp", "id_dossier", "type_dossier",
                              "local_agence", "emp_id"], inplace=True)
        # Un horizon négatif est ramené à zéro.
        groupe["horizon_retraite"] = groupe["horizon_retraite"].clip(lower=0)

    print("Situation familiale utilisée :", situation_frequente)
    print(f"Médiane utilisée pour l'horizon retraite : {mediane_horizon:.2f} ans")
    print("NA restants pour horizon_retraite :", data["horizon_retraite"].isna().sum())
    print("Colonne date_naissance_emp supprimée")
    print("Horizons retraite négatifs restants :", data["horizon_retraite"].lt(0).sum())


    return data, data_train, data_test, y_train, y_test


def comparer(chemin_csv, sortie):
    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    data, train, test, y_train, y_test = preparer_donnees(chemin_csv)
    variables = [
        'situation_fam_emp',
        'contrat_emp',
        'nb_enfants_emp',
        'assistante_maternelle_emp',
        'pension_invalidite_emp',
        'retard_loyer_emp',
        'dette_famille_ami_emp',
        'retard_impot_emp',
        'decouvert_emp',
        'autre_dette_emp',
        'avis_a_tiers_detenteurs_emp',
        'tresorerie',
        'tresorerie_sur_facture',
        'conso_echeance_de_retard',
        'immo_echeance_de_retard',
        'nombre_rejets',
        'nombre_commissions_intervention',
        'fichage_BDF',
        'horizon_rdv',
        'horizon_retraite',
        'nombre_credits_conso',
        'cumul_crd_conso',
        'nombre_credits_immo',
        'cumul_crd_immo',
        'total_mensualites_conso',
        'total_mensualites_immo',
        'total_mensualites_credits',
        'charges',
        'recette',
        'jour_rdv',
        'heure_rdv_num',
        'genre_emp',
        'reste_a_vivre',
        'est_a_decouvert',
    ]
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
