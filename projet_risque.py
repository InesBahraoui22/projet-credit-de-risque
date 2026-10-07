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
print(f"\nHorizons RDV négatifs à vérifier : {data['horizon_rdv'].lt(0).sum()}")

# Années restantes avant la retraite à la date de prise du rendez-vous.
# Une valeur négative signifie que la date de retraite indiquée est déjà passée.
# Une date absente ou illisible reste manquante ; 365.25 convertit les jours
# en années approximatives, sans arrondir les valeurs utilisées pour l'analyse.
date_retraite = pd.to_datetime(data["date_retraite_emp"], format="%Y-%m-%d", errors="coerce")
data["horizon_retraite"] = (date_retraite - date_prise_rdv).dt.days / 365.25

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
