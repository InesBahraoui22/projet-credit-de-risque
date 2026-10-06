import pandas as pd
from pathlib import Path

print("Ca marche")
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

print("\nÉtats conservés :")
print(data["etat"].value_counts(dropna=False))

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

dtale.show(
    data,
    host="127.0.0.1",
    open_browser=True,
    subprocess=False,
    background_mode="missing",
)
