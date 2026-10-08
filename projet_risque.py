import pandas as pd
import ast
import math
from pathlib import Path

# Chargement de la base ; la première colonne du CSV contient l'index.
chemin_data = Path.home() / "Downloads" / "data.csv"
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
data["charges"] = (
    data["pension_versee_emp"]
    + data["loyer_emp"]
    + data["charges_loyer_emp"]
    + data["total_mensualites_credits"]
    + data["charge_recurrente_emp"]
    + data["charge_courante_emp"]
)
data["charges"] = data["charges"].fillna(0)

data["recette"] = (
    data["salaire_emp"]
    + data["rev_foncier_emp"]
    + data["apl_emp"]
    + data["pension_alimentaire_emp"]
    + data["allocation_familiale_emp"]
    + data["pension_invalidite_emp"]
)

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

# Découpage chronologique : environ 70 % pour apprendre, 30 % pour tester.
dates = pd.to_datetime(data["date_aboutisant_azur"], errors="coerce")
if dates.isna().any() or data["id_dossier"].isna().any():
    raise ValueError("Le découpage nécessite une date et un identifiant pour chaque ligne.")

# Garder tous les rendez-vous d'une même date du même côté.
effectifs_par_date = dates.value_counts().sort_index()
effectifs_avant_date = effectifs_par_date.cumsum().shift(fill_value=0)
date_coupure = (effectifs_avant_date - 0.70 * len(data)).abs().idxmin()
data_train = data.loc[dates < date_coupure].copy()
data_test = data.loc[dates >= date_coupure].copy()

# Écarter de l'entraînement les dossiers également présents dans le test.
dossiers_communs = data_train["id_dossier"].isin(data_test["id_dossier"])
nb_lignes_ecartees = int(dossiers_communs.sum())
data_train = data_train.loc[~dossiers_communs].copy()
data_train = data_train.sort_values("date_aboutisant_azur")
data_test = data_test.sort_values("date_aboutisant_azur")

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
    groupe.drop(columns=["date_naissance_emp"], inplace=True)

print("Situation familiale utilisée :", situation_frequente)
print(f"Médiane utilisée pour l'horizon retraite : {mediane_horizon:.2f} ans")
print("NA restants pour horizon_retraite :", data["horizon_retraite"].isna().sum())
print("Colonne date_naissance_emp supprimée")

# Aperçu des données : toutes les colonnes, par groupes adaptés au terminal.
print("\nLes 5 premières lignes :")
with pd.option_context("display.max_columns", None, "display.width", 120):
    print(data.head())

# Bilan par colonne pour préparer le nettoyage, sans modifier la base.
bilan = pd.DataFrame({
    "type": data.dtypes,
    "valeurs_manquantes": data.isna().sum(),
    "pourcentage_manquant": (data.isna().mean() * 100).round(2),
    "valeurs_distinctes": data.nunique(),
}).sort_values("valeurs_manquantes", ascending=False)

print("\nBilan des colonnes (les plus incomplètes en premier) :")
print(bilan.to_string())
print(f"\nNombre de lignes entièrement dupliquées (hors index) : {data.duplicated().sum()}")

# Résumé des variables numériques, comme summary() dans R.
# Les valeurs infinies sont comptées séparément et exclues des statistiques.
numeriques = data.select_dtypes(include="number")
finies = numeriques.replace([float("inf"), float("-inf")], float("nan"))
resume_numerique = finies.describe().T.rename(columns={
    "count": "N valides", "mean": "Moyenne", "std": "Écart-type",
    "min": "Minimum", "25%": "Q1", "50%": "Médiane", "75%": "Q3",
    "max": "Maximum",
})
resume_numerique["NA"] = numeriques.isna().sum()
resume_numerique["Infinis"] = numeriques.isin([float("inf"), float("-inf")]).sum()
resume_numerique["Zéros"] = numeriques.eq(0).sum()
resume_numerique["Négatifs"] = finies.lt(0).sum()
print("\nSUMMARY — Variables numériques :")
print(resume_numerique.round(2).to_string())

# Résumé des textes et booléens : nombre de modalités, plus fréquente et NA.
qualitatives = data.select_dtypes(exclude="number")
resume_qualitatif = qualitatives.describe().T.rename(columns={
    "count": "N valides", "unique": "Modalités",
    "top": "Modalité fréquente", "freq": "Fréquence",
})
resume_qualitatif["NA"] = qualitatives.isna().sum()
print("\nSUMMARY — Variables qualitatives :")
print(resume_qualitatif.to_string())

# Repérage exploratoire des valeurs extrêmes par la règle des boîtes à moustaches.
# IQR = Q3 - Q1. On signale les valeurs hors [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
# Les identifiants, codes et variables binaires ne sont pas adaptés à cette règle.
codes = {"id_dossier", "emp_id", "coemp_id", "rc_id", "cp_emp", "cp_coemp",
         "csp_emp", "csp_coemp"}
alertes = []
for colonne in finies.columns:
    serie = finies[colonne].dropna()
    if colonne in codes or serie.nunique() <= 2:
        continue
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    borne_basse, borne_haute = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    suspects = (serie < borne_basse) | (serie > borne_haute)
    alertes.append({
        "Variable": colonne, "Borne basse": borne_basse, "Borne haute": borne_haute,
        "Valeurs à examiner": int(suspects.sum()),
        "% des valeurs renseignées": round(100 * suspects.mean(), 2),
        "Minimum": serie.min(), "Maximum": serie.max(),
        "Remarque": "IQR nul : règle peu informative" if iqr == 0 else "",
    })
alertes_aberrantes = pd.DataFrame(alertes).sort_values("Valeurs à examiner", ascending=False)
print("\nValeurs potentiellement aberrantes — à vérifier, pas à supprimer automatiquement :")
print(alertes_aberrantes.round(2).to_string(index=False))

# Rapport local consultable dans le navigateur, sans modifier data.
sortie_resume = Path(__file__).resolve().parent / "resume_donnees"
sortie_resume.mkdir(exist_ok=True)
resume_numerique.to_csv(sortie_resume / "numeriques.csv", index_label="Variable")
resume_qualitatif.to_csv(sortie_resume / "qualitatives.csv", index_label="Variable")
alertes_aberrantes.to_csv(sortie_resume / "valeurs_a_examiner.csv", index=False)

rapport = """<!doctype html><html lang="fr"><meta charset="utf-8">
<title>Résumé des données</title><style>
body{font-family:Arial,sans-serif;margin:30px;color:#243746}
table{border-collapse:collapse;font-size:13px}th,td{padding:7px 12px;border:1px solid #ddd}
th{background:#e9f1f5}tr:nth-child(even){background:#f6f8fa}
.tableau{overflow:auto;max-height:70vh}thead th{position:sticky;top:0}
h2{margin-top:36px}a{color:#176b91}p{max-width:1000px;line-height:1.5}
</style><h1>Résumé de la base après les filtres demandés</h1>
<p>Les valeurs manquantes sont affichées « NA ». Les alertes IQR signalent des valeurs
à examiner, pas nécessairement des erreurs. Quand Q1 = Q3, notamment pour les colonnes
contenant beaucoup de zéros, la règle est peu informative. Aucune valeur n'est modifiée.</p>
<p>Les statistiques sont descriptives : les montants stockés comme texte ou listes ne sont
pas encore analysés comme des nombres. Les dates et valeurs sentinelles nécessitent un contrôle séparé.</p>
<nav><a href="#na">Valeurs manquantes</a> · <a href="#nombres">Numériques</a> ·
<a href="#categories">Qualitatives</a> · <a href="#alertes">Valeurs à examiner</a></nav>
"""
rapport += f"<p>{len(data):,} lignes — {data.shape[1]} colonnes.</p>"
for identifiant, titre, tableau in [
    ("na", "Types et valeurs manquantes", bilan),
    ("nombres", "Résumé numérique", resume_numerique),
    ("categories", "Résumé qualitatif", resume_qualitatif),
    ("alertes", "Valeurs potentiellement aberrantes (IQR)", alertes_aberrantes),
]:
    rapport += f'<h2 id="{identifiant}">{titre}</h2><div class="tableau">'
    rapport += tableau.to_html(na_rep="NA", float_format=lambda x: f"{x:,.2f}", escape=True)
    rapport += "</div>"
rapport += "</html>"
chemin_resume = sortie_resume / "summary.html"
chemin_resume.write_text(rapport, encoding="utf-8")
print(f"\nRésumé consultable dans le navigateur : {chemin_resume}")

# Premier modèle : forêt aléatoire sans SMOTE.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             RocCurveDisplay, ConfusionMatrixDisplay)
from matplotlib.figure import Figure

# Variables candidates : leur disponibilité à la prise du RDV doit être confirmée.
# Les identifiants, textes libres et dates brutes ne sont pas des prédicteurs ici.
variables_modele = [
    "type_dossier", "situation_fam_emp", "contrat_emp", "nb_enfants_emp",
    "assistante_maternelle_emp", "pension_invalidite_emp",
    "retard_loyer_emp", "dette_famille_ami_emp", "retard_impot_emp",
    "decouvert_emp", "autre_dette_emp", "avis_a_tiers_detenteurs_emp",
    "tresorerie", "tresorerie_sur_facture",
    "conso_echeance_de_retard", "immo_echeance_de_retard",
    "nombre_rejets", "nombre_commissions_intervention", "fichage_BDF",
    "horizon_rdv", "horizon_retraite",
    "nombre_credits_conso", "cumul_crd_conso",
    "nombre_credits_immo", "cumul_crd_immo",
    "total_mensualites_conso", "total_mensualites_immo",
    "total_mensualites_credits", "charges", "recette",
]
X_train = data_train[variables_modele].copy()
X_test = data_test[variables_modele].copy()

# contrat_emp reste une catégorie, même si ses modalités sont codées en chiffres.
variables_categories = ["type_dossier", "situation_fam_emp", "contrat_emp"]
preparation = ColumnTransformer(
    [("categories", OneHotEncoder(handle_unknown="ignore"), variables_categories)],
    remainder="passthrough",
)
modele_rf = Pipeline([
    ("preparation", preparation),
    ("foret", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
])
modele_rf.fit(X_train, y_train)

# Probabilité d'annulation et décision avec un seuil fixé à 0.5.
probabilites_rf = modele_rf.predict_proba(X_test)[:, 1]
predictions_rf = (probabilites_rf >= 0.5).astype(int)
auc_rf = roc_auc_score(y_test, probabilites_rf)
ap_rf = average_precision_score(y_test, probabilites_rf)
rapport_rf = classification_report(
    y_test, predictions_rf, labels=[0, 1],
    target_names=["Fait", "Annulé client"], zero_division=0,
)
matrice_rf = confusion_matrix(y_test, predictions_rf, labels=[0, 1])
print(f"\nRandom Forest sans SMOTE — AUC ROC : {auc_rf:.3f}")
print(f"Average precision : {ap_rf:.3f}")
print(rapport_rf)
print("Matrice : lignes = réalité, colonnes = prédiction ; ordre = fait, annulé")
print(matrice_rf)

# Enregistrer les résultats et les graphiques avant l'ouverture de D-Tale.
sortie_modele = Path(__file__).resolve().parent / "resultats_modelisation"
sortie_modele.mkdir(exist_ok=True)
(sortie_modele / "random_forest.txt").write_text(
    f"Forêt sans SMOTE — seuil 0.5\n"
    f"Train : {len(X_train)} ; test : {len(X_test)}\n"
    f"AUC ROC : {auc_rf:.4f}\nAverage precision : {ap_rf:.4f}\n\n"
    + rapport_rf + "\nMatrice (ordre : fait, annulé) :\n" + str(matrice_rf),
    encoding="utf-8",
)
figure = Figure(figsize=(11, 4.5), constrained_layout=True)
axes = figure.subplots(1, 2)
RocCurveDisplay.from_predictions(y_test, probabilites_rf, ax=axes[0], name="Random Forest")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
axes[0].set_title("Courbe ROC — test chronologique")
ConfusionMatrixDisplay(matrice_rf, display_labels=["Fait", "Annulé"]).plot(
    ax=axes[1], colorbar=False, cmap="Blues",
)
axes[1].set_title("Matrice de confusion — seuil 0.5")
figure.savefig(sortie_modele / "random_forest.png", dpi=150)
print(f"Résultats du modèle : {sortie_modele}")

import dtale

# Afficher les nouvelles variables en premier pour les retrouver facilement.
colonnes_en_tete = ["fichage_BDF", "horizon_rdv", "horizon_retraite",
                    "nombre_credits_conso", "cumul_crd_conso",
                    "nombre_credits_immo", "cumul_crd_immo",
                    "total_mensualites_conso", "total_mensualites_immo",
                    "total_mensualites_credits", "charges", "recette"]
data_affichage = data[colonnes_en_tete + [
    colonne for colonne in data.columns if colonne not in colonnes_en_tete
]].copy()
print(f"\nFichier exécuté : {Path(__file__).resolve()}")
print("D-Tale : dossiers sans co-emprunteur uniquement.")


dtale.show(
    data_affichage,
    name="Sans coemprunteur",
    host="127.0.0.1",
    open_browser=True,
    subprocess=False,
    background_mode="missing",
)
