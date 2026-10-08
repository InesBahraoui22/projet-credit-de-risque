#%%
import pandas as pd
df = pd.read_csv("data1.csv", low_memory=False)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")
# %%
df.to_html("Data_clean.html", index=False)
# %%
# nombre de lignes et de colonnes
print("Dimension:",df.shape)
print("information générales:")
df.info() # type de variable
print("valeur manquantes:")
df.isnull().sum()

# %%
# laisser que les valeur fait et annuler client dans la colonne etat
etats_a_supprimer = [
    "refus entreprise",
    "refus client",
    "refus avant rendezvous",
    "refus avant rdv",
    "pas fait"
]

df["etat"] = df["etat"].astype("string").str.strip().str.lower()

df = df[~df["etat"].isin(etats_a_supprimer)].copy()

#%%
df["type_dossier"] = (
    df["type_dossier"]
    .astype("string")
    .str.strip()
    .str.lower()
)

df = df[df["type_dossier"] != "locataire"].copy()

# %%
df = df[df["type_dossier"] != "locataire"]
# %%
df["type_dossier"].value_counts(dropna=False)
# %%
df.to_html("Data_clean.html", index=False)
# %%
print(df.shape)
print(df.columns.tolist())# %%

# %%
df = df.drop_duplicates().copy()
# %%
print("Dimension:",df.shape)

#%%
df.shape

#%%
df = df[df["coemp_id"].isna()].copy()# %%
colonnes_a_supprimer = [
    "hebergement_gratuit_emp",
    "type_invalidite_coemp",
    "date_deja_rachat",
    "type_invalidite_emp",
    "csp_emp",
    "hebergement_gratuit_coemp",
    "rc_id",
    "date_retraite_coemp",
    "contrat_mariage_emp",
    "nature_de_projet",
    "contrat_coemp",
    "cp_coemp",
    "situation_fam_coemp",
    "date_naissance_coemp",
    "assistante_maternelle_coemp",
    "allocation_familiale_coemp"
]

df = df.drop(columns=colonnes_a_supprimer)
# %%
colonnes_a_supprimer_2 = [
    "coemp_id",
    "intitule_coemp",
    "consentement_coemp",
    "salaire_coemp",
    "rev_foncier_coemp",
    "conge_parental_coemp",
    "apl_coemp",
    "pension_alimentaire_coemp",
    "pension_invalidite_coemp",
    "anciennete_coemp"
]

df = df.drop(columns=colonnes_a_supprimer_2, errors="ignore")
print(df.columns.tolist())
# %%
colonnes_coemp = [
    "profession_coemp",
    "csp_coemp",
    "contrat_mariage_coemp"
]

avec_coemp = df[df[colonnes_coemp].notna().any(axis=1)].copy()

sans_coemp = df[df[colonnes_coemp].isna().all(axis=1)].copy()
# %%
print("Avec co-emprunteur :", avec_coemp.shape[0])
print("Sans co-emprunteur :", sans_coemp.shape[0])
# %%
df = df.drop(columns=["segmentation"])
# %%
df = df.drop(columns=["commentaire"], errors="ignore")
# %%
"commentaire" in df.columns
# %%
df["total_mensualite"] = (
    df["conso_mensualite"].fillna(0)
    + df["immo_mensualite"].fillna(0)
)
# %%
df[["conso_mensualite", "immo_mensualite", "total_mensualite"]].head()
#%%

df["conso_crd"] = pd.to_numeric(df["conso_crd"], errors="coerce")
df["immo_crd"] = pd.to_numeric(df["immo_crd"], errors="coerce")

# %%
df["nombre_credit"] = (
    (df["conso_crd"].fillna(0) > 0).astype(int)
    + (df["immo_crd"].fillna(0) > 0).astype(int)
)
# %%

print(df["conso_crd"].dtype)
print(df["immo_crd"].dtype)

# %%
df["conso_crd"] = pd.to_numeric(df["conso_crd"], errors="coerce")
df["immo_crd"] = pd.to_numeric(df["immo_crd"], errors="coerce")
# %%
print(df["conso_crd"].dtype)
print(df["immo_crd"].dtype)
# %%
df["nombre_credit"] = (
    (df["conso_crd"].fillna(0) > 0).astype(int)
    + (df["immo_crd"].fillna(0) > 0).astype(int)
)
# %%
df["cumul_credit"] = (
    df["conso_crd"].fillna(0)
    + df["immo_crd"].fillna(0)
)
# %%
df[["conso_crd", "immo_crd", "nombre_credit", "cumul_credit"]].head(10)
# %%
df[["conso_crd", "immo_crd"]].head(20)
#%%
colonnes_charges = [
    "pension_versee_emp",
    "loyer_emp",
    "charges_loyer_emp",
    "total_mensualite",
    "charge_recurrente_emp",
    "charge_courante_emp"
]

for col in colonnes_charges:
    df[col] = pd.to_numeric(df[col], errors="coerce")




# %%
df["charges"] = (
    df["pension_versee_emp"].fillna(0)
    + df["loyer_emp"].fillna(0)
    + df["charges_loyer_emp"].fillna(0)
    + df["total_mensualite"].fillna(0)
    + df["charge_recurrente_emp"].fillna(0)
    + df["charge_courante_emp"].fillna(0)
)
# %%

colonnes_charges = [
    "pension_versee_emp",
    "loyer_emp",
    "charges_loyer_emp",
    "total_mensualite",
    "charge_recurrente_emp",
    "charge_courante_emp"
]

for col in colonnes_charges:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# %%
df["charges"] = (
    df["pension_versee_emp"].fillna(0)
    + df["loyer_emp"].fillna(0)
    + df["charges_loyer_emp"].fillna(0)
    + df["total_mensualite"].fillna(0)
    + df["charge_recurrente_emp"].fillna(0)
    + df["charge_courante_emp"].fillna(0)
)
df[
    [
        "pension_versee_emp",
        "loyer_emp",
        "charges_loyer_emp",
        "total_mensualite",
        "charge_recurrente_emp",
        "charge_courante_emp",
        "charges"
    ]
].head(10)

#%%
df = df.drop(
    columns=[
        "loyer_emp",
        "charges_loyer_emp",
        "charge_recurrente_emp",
        "charge_courante_emp",
        "charge_future_eventuelle_emp",
        "pension_versee_emp"
    ],
    errors="ignore"
)

# %%
colonnes_recette = [
    "salaire_emp",
    "apl_emp",
    "pension_alimentaire_emp",
    "allocation_familiale_emp",
    "pension_invalidite_emp"
]

for col in colonnes_recette:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["recette"] = (
    df["salaire_emp"].fillna(0)
    + df["apl_emp"].fillna(0)
    + df["pension_alimentaire_emp"].fillna(0)
    + df["allocation_familiale_emp"].fillna(0)
    + df["pension_invalidite_emp"].fillna(0)
)
# %%
df[
    [
        "salaire_emp",
        "apl_emp",
        "pension_alimentaire_emp",
        "allocation_familiale_emp",
        "pension_invalidite_emp",
        "recette"
    ]
].head(10)
# %%
df = df.drop(
    columns=["conso_conserve", "immo_conserve"],
    errors="ignore"
)
# %%
print("conso_conserve" in df.columns)
print("immo_conserve" in df.columns)
# %%
df["dette"] = (
    (df["dette_famille_ami_emp"].fillna(0) > 0)
    | (df["autre_dette_emp"].fillna(0) > 0)
)
# %%
df["dette"].value_counts(dropna=False)

# %%
df["heure_rdv_debut"] = pd.to_numeric(
    df["heure_rdv_debut"],
    errors="coerce"
).astype("Int64")
# %%
df["heure_rdv_debut"].head(10)

# %%
df["charges"] = df["charges"].fillna(0)
df["charges"].isna().sum()
# %%
df["tresorerie"] = pd.to_numeric(df["tresorerie"], errors="coerce")
df["tresorerie_sur_facture"] = pd.to_numeric(
    df["tresorerie_sur_facture"],
    errors="coerce"
)

df["tresorerie"] = df["tresorerie"].fillna(0) > 0
df["tresorerie_sur_facture"] = df["tresorerie_sur_facture"].fillna(0) > 0
# %%
df[["tresorerie", "tresorerie_sur_facture"]].head(10)
# %%
df = df.drop(columns=["nature_de_projet"], errors="ignore")
# %%
df = df.drop(
    columns=[
        "conso_date_debut",
        "conso_date_fin",
        "conso_duree"
    ],
    errors="ignore"
)
# %%
print("conso_date_debut" in df.columns)
print("conso_date_fin" in df.columns)
print("conso_duree" in df.columns)
# %%
df = df.drop(
    columns=[
        "immo_garantie",
        "immo_date_debut",
        "immo_date_fin",
        "immo_duree",
        "immo_libelle",
        "contrat_mariage_emp",
        "csp_emp",
        "anciennete_emp"
    ],
    errors="ignore"
)
# %%
df["conso_echeance_de_retard"] = (
    pd.to_numeric(df["conso_echeance_de_retard"], errors="coerce")
    .fillna(0) > 0
)

df["immo_echeance_de_retard"] = (
    pd.to_numeric(df["immo_echeance_de_retard"], errors="coerce")
    .fillna(0) > 0
)
# %%
df["fichage_BDF"] = (
    df["fichage_FCC_carte"].fillna(False)
    | df["fichage_FCC_cheque"].fillna(False)
    | df["fichage_FICP"].fillna(False)
)
# %%
df[
    [
        "fichage_FCC_carte",
        "fichage_FCC_cheque",
        "fichage_FICP",
        "fichage_BDF"
    ]
].head(10)

#%%
df = df.drop(
    columns=[
        "fichage_FCC_carte",
        "fichage_FCC_cheque",
        "fichage_FICP"
    ],
    errors="ignore"
)
# %%
df = df.drop(
    columns=[
        "valeur_gage",
        "valeur_residence_principale",
        "valeur_bien_immobilier",
        "valeur_produit_epargne_placement",
        "date_acquisition",
        "valeur_acquisition",
        "nb_etablissement_ficp"
    ],
    errors="ignore"
)
# %%
print(df.columns.tolist())
# %%
df.shape
# %%
df.to_html("Data_clean.html", index=False)
# %%
df = df.drop(
    columns=[
        "nature_de_projet",
        "conso_type",
        "conso_mensualite",
        "conso_crd"
    ],
    errors="ignore"
)
print(df.columns.tolist())
# %%
df["nombre_commissions_intervention"] = (
    pd.to_numeric(
        df["nombre_commissions_intervention"],
        errors="coerce"
    )
    .fillna(0)
    > 0
)
df["nombre_commissions_intervention"].value_counts(dropna=False)
# %%
df = df.drop(
    columns=[
        "conso_taux",
        "immo_taux",
        "immo_crd",
        "immo_garantie",
        "immo_mensualite"
    ],
    errors="ignore"
)
print(df.columns.tolist())
# %%
df = df.drop(
    columns=[
        "salaire_emp",
        "rev_foncier_emp",
        "conge_parental_emp",
        "apl_emp",
        "pension_alimentaire_emp",
        "allocation_familiale_emp",
        "date_retraite_emp",
        "saisie_sur_salaire_emp"
    ],
    errors="ignore"
)
print(df.columns.tolist())
# %%
colonnes_bool = [
    "retard_loyer_emp",
    "avis_a_tiers_detenteurs_emp",
    "autre_dette_emp"
]

for col in colonnes_bool:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
for col in colonnes_bool:
    print("\n", col)
    print(df[col].value_counts(dropna=False))
# %%
df = df.drop(
    columns=[
        "valeur_gage",
        "valeur_residence_principale",
        "valeur_bien_immobilier",
        "valeur_produit_epargne_placement",
        "date_acquisition",
        "valeur_acquisition",
        "nb_etablissement_ficp"
    ],
    errors="ignore"
)
print(df.columns.tolist())
# %%
df.shape

# %%
df.to_html("Data_clean.html", index=False)
# %%
colonnes_coemp = [
    col for col in df.columns
    if "coemp" in col
]

df = df.drop(columns=colonnes_coemp)
#%%
df.shape
# %%
df.to_html("Data_clean.html", index=False)
#%%
df["contrat_emp"] = df["contrat_emp"].fillna("inconnu")
df["contrat_emp"].value_counts(dropna=False)
# %%
df = df.drop(
    columns=[
        "affectation",
        "duree",
        "type_traitement_dossier",
        "type_support",
        "profession_emp"
    ],
    errors="ignore"
)

# %%
print(df.columns.tolist())
# %%
df.shape
#%%
df = df[
    df["date_rdv"] >= df["date_aboutisant_azur"]
].copy()
# %%
nb_avant = len(df)

df = df[
    df["date_rdv"] >= df["date_aboutisant_azur"]
].copy()

print("Lignes supprimées :", nb_avant - len(df))
print("Lignes restantes :", len(df))
# %%
colonnes_na_zero = [
    "dette_famille_ami_emp",
    "retard_impot_emp",
    "decouvert_emp"
]

df[colonnes_na_zero] = df[colonnes_na_zero].fillna(0)
# %%
df[colonnes_na_zero].isna().sum()
# %%
df.shape
# %%
df.to_html("Data_clean.html", index=False)

# %%

## horizon retraite

# ============================================
# Traitement de horizon_retraite AVANT le split
# ============================================

age_retraite_legal = 64

# Conversion en dates
df["date_aboutisant_azur"] = pd.to_datetime(
    df["date_aboutisant_azur"],
    errors="coerce"
)

df["date_naissance_emp"] = pd.to_datetime(
    df["date_naissance_emp"],
    errors="coerce"
)

# Age au moment du dossier
age_au_t0 = (
    df["date_aboutisant_azur"]
    - df["date_naissance_emp"]
).dt.days / 365.25

# Horizon avant retraite
horizon_theorique = age_retraite_legal - age_au_t0

# Création de la nouvelle variable
df["horizon_retraite"] = horizon_theorique

# Valeurs négatives -> 0
df.loc[
    df["horizon_retraite"] < 0,
    "horizon_retraite"
] = 0

print(
    "NA restants pour horizon_retraite :",
    df["horizon_retraite"].isna().sum()
)

# Supprimer la date de naissance après le calcul
df = df.drop(
    columns=["date_naissance_emp"],
    errors="ignore"
)
print("Colonne date_naissance_emp supprimée")

#%%
## split 70 % / 30 % :
X = df.drop(
    columns=["etat", "id_dossier", "emp_id", "Unnamed: 0"],
    errors="ignore"
)

y = df["etat"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Train :", X_train.shape)
print("Test :", X_test.shape)

# Remplacement des derniers NA de horizon_retraite
# avec la médiane calculée uniquement sur le TRAIN

mediane_horizon = X_train["horizon_retraite"].median()

X_train["horizon_retraite"] = (
    X_train["horizon_retraite"]
    .fillna(mediane_horizon)
)

X_test["horizon_retraite"] = (
    X_test["horizon_retraite"]
    .fillna(mediane_horizon)
)

print(
    "NA horizon_retraite train :",
    X_train["horizon_retraite"].isna().sum()
)

print(
    "NA horizon_retraite test :",
    X_test["horizon_retraite"].isna().sum()
)

# %%
df["date_aboutisant_azur"] = pd.to_datetime(
    df["date_aboutisant_azur"],
    errors="coerce"
)

df = df.sort_values("date_aboutisant_azur").copy()
df[["date_aboutisant_azur", "etat"]].head()

# %%
# faire le split

seuil = int(len(df) * 0.70)

train = df.iloc[:seuil].copy()
test = df.iloc[seuil:].copy()
print("Train :", train.shape)
print("Test :", test.shape)
# %%
# Étape 3 — Séparer les variables explicatives X et la cible y

y_train = train["etat"]
y_test = test["etat"]

X_train = train.drop(
    columns=[
        "etat",
        "id_dossier",
        "emp_id",
        "Unnamed: 0"
    ],
    errors="ignore"
)

X_test = test.drop(
    columns=[
        "etat",
        "id_dossier",
        "emp_id",
        "Unnamed: 0"
    ],
    errors="ignore"
)

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print("y_train :", y_train.shape)
print("y_test :", y_test.shape)
print("Train :")
print(y_train.value_counts())

print("\nTest :")
print(y_test.value_counts())
# %%
# preparer les variable categorial
X_train = X_train.drop(
    columns=["date_aboutisant_azur"],
    errors="ignore"
)

X_test = X_test.drop(
    columns=["date_aboutisant_azur"],
    errors="ignore"
)
print(X_train.select_dtypes(include=["datetime64[ns]"]).columns.tolist())
print(X_test.select_dtypes(include=["datetime64[ns]"]).columns.tolist())

# %%
print("NA dans X_train :", X_train.isna().sum().sum())
print("NA dans X_test :", X_test.isna().sum().sum())
print(
    X_train.isna().sum()
    .sort_values(ascending=False)
    .head(20)
)
# %%
X_train["situation_fam_emp"] = X_train["situation_fam_emp"].fillna("inconnu")
X_test["situation_fam_emp"] = X_test["situation_fam_emp"].fillna("inconnu")
mediane_horizon = X_train["horizon_retraite"].median()

X_train["horizon_retraite"] = X_train["horizon_retraite"].fillna(mediane_horizon)
X_test["horizon_retraite"] = X_test["horizon_retraite"].fillna(mediane_horizon)
print(
    "total_mensualite :",
    X_train["total_mensualite"].isna().mean() * 100,
    "%"
)

print(
    "heure_rdv_debut :",
    X_train["heure_rdv_debut"].isna().mean() * 100,
    "%"
)
# %%
X_train = X_train.drop(
    columns=["total_mensualite", "heure_rdv_debut"],
    errors="ignore"
)

X_test = X_test.drop(
    columns=["total_mensualite", "heure_rdv_debut"],
    errors="ignore"
)
# %%
print("NA dans X_train :", X_train.isna().sum().sum())
print("NA dans X_test :", X_test.isna().sum().sum())
# %%
print(
    X_train.isna().sum()
    .sort_values(ascending=False)
    .head(20)
)
# %%
colonnes_cat = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns

print(colonnes_cat.tolist())
# %%
X_train = pd.get_dummies(
    X_train,
    columns=colonnes_cat,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=colonnes_cat,
    dtype=int
)
# %%
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)
# %%
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print(
    "Colonnes texte restantes :",
    X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
)
# %%
# Nettoyer les noms des colonnes pour XGBoost
X_train.columns = (
    X_train.columns
    .astype(str)
    .str.replace("[", "_", regex=False)
    .str.replace("]", "_", regex=False)
    .str.replace("<", "_", regex=False)
)

X_test.columns = (
    X_test.columns
    .astype(str)
    .str.replace("[", "_", regex=False)
    .str.replace("]", "_", regex=False)
    .str.replace("<", "_", regex=False)
)
print(X_train.columns.tolist())
# %%
modele_xgb.fit(X_train, y_train)
# %%
y_pred = modele_xgb.predict(X_test)
# %%
y_proba = modele_xgb.predict_proba(X_test)[:, 1]
# %%
# evaluation du modèle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision :", precision_score(y_test, y_pred))
print("Recall :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("ROC-AUC :", roc_auc_score(y_test, y_proba))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report :")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["fait", "annuler client"]
    )
)
# %%
## recall 0,01 1% des annuler qui predit mon premmier modèle
# gérer le déséquilibre des classes
#Avec XGBoost, on peut donner plus d importance à la classe 1 = annuler client.

# calcule du poids

nb_fait = (y_train == 0).sum()
nb_annule = (y_train == 1).sum()

poids_classe_1 = nb_fait / nb_annule

print("Poids classe annuler client :", poids_classe_1)

# %%
from xgboost import XGBClassifier

modele_xgb_equilibre = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=poids_classe_1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

modele_xgb_equilibre.fit(X_train, y_train)
# %%
y_pred_eq = modele_xgb_equilibre.predict(X_test)
y_proba_eq = modele_xgb_equilibre.predict_proba(X_test)[:, 1]
# %%
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_eq))

print("\nClassification report :")
print(
    classification_report(
        y_test,
        y_pred_eq,
        target_names=["fait", "annuler client"]
    )
)

print(
    "ROC-AUC :",
    roc_auc_score(y_test, y_proba_eq)
)
# %%
# ajuster le seuil de décision

from sklearn.metrics import precision_score, recall_score, f1_score

for seuil in [0.3, 0.4, 0.5, 0.6, 0.7]:

    y_pred_seuil = (y_proba_eq >= seuil).astype(int)

    precision = precision_score(y_test, y_pred_seuil)
    recall = recall_score(y_test, y_pred_seuil)
    f1 = f1_score(y_test, y_pred_seuil)

    print(
        "Seuil :", seuil,
        "| Precision :", round(precision, 3),
        "| Recall :", round(recall, 3),
        "| F1 :", round(f1, 3)
    )
# %%


## test pour 80 / 20

# 80 % train / 20 % test
seuil = int(len(df) * 0.80)

train = df.iloc[:seuil].copy()
test = df.iloc[seuil:].copy()

print("Train :", train.shape)
print("Test :", test.shape)

y_train = train["etat"]
y_test = test["etat"]

X_train = train.drop(
    columns=[
        "etat",
        "id_dossier",
        "emp_id",
        "Unnamed: 0"
    ],
    errors="ignore"
)

X_test = test.drop(
    columns=[
        "etat",
        "id_dossier",
        "emp_id",
        "Unnamed: 0"
    ],
    errors="ignore"
)

print(
    "Train :",
    train["date_aboutisant_azur"].min(),
    "->",
    train["date_aboutisant_azur"].max()
)

print(
    "Test :",
    test["date_aboutisant_azur"].min(),
    "->",
    test["date_aboutisant_azur"].max()
)
# %%
colonnes_dates = X_train.select_dtypes(
    include=["datetime64[ns]", "datetime64"]
).columns

print("Colonnes dates :", colonnes_dates.tolist())


# %%
X_train = X_train.drop(
    columns=colonnes_dates,
    errors="ignore"
)

X_test = X_test.drop(
    columns=colonnes_dates,
    errors="ignore"
)

# %%
X_train = X_train.drop(
    columns=["date_aboutisant_azur"],
    errors="ignore"
)

X_test = X_test.drop(
    columns=["date_aboutisant_azur"],
    errors="ignore"
)
# %%
print(
    X_train.select_dtypes(
        include=["datetime64[ns]", "datetime64"]
    ).columns.tolist()
)

print(
    X_test.select_dtypes(
        include=["datetime64[ns]", "datetime64"]
    ).columns.tolist()
)
# %%
print("NA dans X_train :", X_train.isna().sum().sum())
print("NA dans X_test :", X_test.isna().sum().sum())
# %%
print(
    X_train.isna().sum()
    .sort_values(ascending=False)
    .head(20)
)
# %%
X_train = X_train.drop(
    columns=["total_mensualite", "heure_rdv_debut"],
    errors="ignore"
)

X_test = X_test.drop(
    columns=["total_mensualite", "heure_rdv_debut"],
    errors="ignore"
)
# %%
X_train["situation_fam_emp"] = (
    X_train["situation_fam_emp"].fillna("inconnu")
)

X_test["situation_fam_emp"] = (
    X_test["situation_fam_emp"].fillna("inconnu")
)
# %%
mediane_horizon = X_train["horizon_retraite"].median()

X_train["horizon_retraite"] = (
    X_train["horizon_retraite"].fillna(mediane_horizon)
)

X_test["horizon_retraite"] = (
    X_test["horizon_retraite"].fillna(mediane_horizon)
)
# %%
print("NA dans X_train :", X_train.isna().sum().sum())
print("NA dans X_test :", X_test.isna().sum().sum())
# %%
colonnes_cat = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns

print(colonnes_cat.tolist())
# %%
X_train = pd.get_dummies(
    X_train,
    columns=colonnes_cat,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=colonnes_cat,
    dtype=int
)

# %%
colonnes_a_retirer = [
    "date_creation",
    "date_rdv",
    "date_aboutisant_azur",
    "statut_final"
]

X_train = X_train.drop(
    columns=colonnes_a_retirer,
    errors="ignore"
)

X_test = X_test.drop(
    columns=colonnes_a_retirer,
    errors="ignore"
)
# %%
colonnes_cat = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

print(colonnes_cat)
# %%
X_train = pd.get_dummies(
    X_train,
    columns=colonnes_cat,
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=[
        col for col in colonnes_cat
        if col in X_test.columns
    ],
    dtype=int
)
# %%
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)
# %%
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print(
    "Colonnes texte restantes :",
    X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
)
#%%
# Nettoyer les noms de colonnes pour XGBoost
X_train.columns = (
    X_train.columns
    .astype(str)
    .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
)

X_test.columns = (
    X_test.columns
    .astype(str)
    .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
)

colonnes_probleme = [
    col for col in X_train.columns
    if "[" in col or "]" in col or "<" in col
]

print(colonnes_probleme)

# %%

X_train_xgb = X_train.to_numpy(dtype=float)
X_test_xgb = X_test.to_numpy(dtype=float)
print(X_train_xgb.shape)
print(X_test_xgb.shape)
# %%
from xgboost import XGBClassifier

poids_classe_1 = (y_train == 0).sum() / (y_train == 1).sum()

modele_xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=poids_classe_1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

modele_xgb.fit(X_train_xgb, y_train)
# %%
y_pred = modele_xgb.predict(X_test_xgb)

y_proba = modele_xgb.predict_proba(X_test_xgb)[:, 1]
# %%
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report :")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["fait", "annuler client"]
    )
)

print("\nROC-AUC :", roc_auc_score(y_test, y_proba))
# %%

# plot AUC-ROC
import matplotlib.pyplot as plt

modeles = [
    "70/30\nInitial",
    "70/30\nÉquilibré",
    "80/20\nÉquilibré"
]

recall_scores = [
    8 / 618,   # ancien modèle 70/30
    0.757,     # 70/30 équilibré
    0.800      # 80/20 équilibré
]

plt.figure(figsize=(8, 5))

bars = plt.bar(modeles, recall_scores)

plt.ylim(0, 1)
plt.ylabel("Recall - Annuler client")
plt.title("Comparaison du Recall des 3 modèles")

for bar, score in zip(bars, recall_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.3f}",
        ha="center"
    )

plt.show()

# %%
# maintenant pour le recall

import matplotlib.pyplot as plt

modeles = [
    "70/30\nInitial",
    "70/30\nÉquilibré",
    "80/20\nÉquilibré"
]

recall_scores = [
    8 / 618,   # 70/30 initial
    0.757,     # 70/30 équilibré
    0.800      # 80/20 équilibré
]

plt.figure(figsize=(8, 5))

bars = plt.bar(modeles, recall_scores)

plt.ylim(0, 1)
plt.ylabel("Recall - Annuler client")
plt.xlabel("Modèles")
plt.title("Comparaison du Recall des 3 modèles")

for bar, score in zip(bars, recall_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score + 0.02,
        f"{score:.3f}",
        ha="center"
    )

plt.show()
