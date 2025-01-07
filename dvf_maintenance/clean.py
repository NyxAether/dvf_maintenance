from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_alt_df_cols = [
    "id_mutation",
    "jour_mutation",
    "mois_mutation",
    "annee_mutation",
    "nature_mutation",
    "valeur_fonciere",
    "adresse_numero",
    "adresse_suffixe",
    "adresse_nom_voie",
    "adresse_code_voie",
    "code_postal",
    "nom_commune",
    "code_departement",
    "id_parcelle",
    "surface_carrez_total",
    "surface_reelle_bati_total",
    "surface_terrain_total",
    "nombre_lots",
    "nombre_maisons",
    "surface_carrez_maisons",
    "surface_reelle_bati_maisons",
    "surface_terrain_maisons",
    "nombre_appartements",
    "surface_carrez_appartements",
    "surface_reelle_bati_appartements",
    "surface_terrain_appartements",
    "nombre_dependances",
    "surface_carrez_dependances",
    "surface_reelle_bati_dependances",
    "surface_terrain_dependances",
    "nombre_pieces_principales",
    "nature_culture",
    "nature_culture_speciale",
    "longitude",
    "latitude",
]


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les valeurs manquantes.
        ATTENTION : cette méthode supprime les ventes de locaux commerciaux.
        Cela changera dans le futur

    Args:
        df (pd.DataFrame): DataFrame à nettoyer

    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    df_clean = df.drop(
        columns=["ancien_code_commune", "ancien_nom_commune", "ancien_id_parcelle"]
    )
    # Elimination des locaux industriels et commerciaux
    mutation_commerciale = df_clean[df_clean.code_type_local == 4].id_mutation
    df_clean = df_clean[~df_clean.id_mutation.isin(mutation_commerciale)]
    coltofill = [
        "valeur_fonciere",
        "nombre_pieces_principales",
        "surface_reelle_bati",
        "surface_terrain",
        "lot1_surface_carrez",
        "lot2_surface_carrez",
        "lot3_surface_carrez",
        "lot4_surface_carrez",
        "lot5_surface_carrez",
        "lot1_numero",
        "lot2_numero",
        "lot3_numero",
        "lot4_numero",
        "lot5_numero",
        "adresse_numero",
        "code_postal",
        "numero_volume",
        "longitude",
        "latitude",
    ]
    df_clean[coltofill] = df_clean[coltofill].fillna(0)
    df_clean["code_type_local"] = df_clean["code_type_local"].fillna(3)
    df_clean["type_local"] = df_clean["type_local"].fillna("Dépendance")

    coltofill = [
        "adresse_nom_voie",
        "adresse_code_voie",
        "code_nature_culture",
        "adresse_suffixe",
        "nature_culture",
        "code_nature_culture_speciale",
        "nature_culture_speciale",
    ]
    df_clean[coltofill] = df_clean[coltofill].fillna("<EMPTY>")
    return df_clean


def reduce_get_id(df: pd.DataFrame) -> np.signedinteger[Any]:
    """Retourne l'indice de ligne d'un bien pour une mutation.
    Si la mutation compte plusieurs biens, l'indice du bien retourné
    sera celui qui est une maison plutôt qu'un appartement.
    Si il reste plusieurs canditats, le bien avec
    la plus grande surface réelle sera retourné

    Args:
        df (pd.DataFrame): DataFrame contenant un ou plusieurs biens de la même mutation

    Returns:
        int: L'indice de la ligne du bien principale de la mutation
    """
    type_count = df.code_type_local.value_counts()
    if 1 in type_count:
        # Maison
        if type_count[1] == 1:
            return df.index.get_loc(df.index[df.code_type_local == 1][0])
        else:
            return np.argmax(df.surface_reelle_bati)
    elif 2 in type_count:
        # Appartement
        if type_count[2] == 1:
            return df.index.get_loc(df.index[df.code_type_local == 2][0])
        else:
            return np.argmax(df.surface_reelle_bati)
    return np.argmax(df.surface_reelle_bati)


def fusion_data(df: pd.DataFrame) -> pd.Series:
    """Fusionne les mutations pour les formater en une seule ligne.
    Les surfaces sont sommées et le nombres de lots est pris en compte.
    Néanmoins du contenu est inévitablement perdu

    Args:
        df (pd.DataFrame): DataFrame qui contient un ou
        plusieurs biens d'une même mutation

    Returns:
        pd.Series: Série correspondant à la nouvelle ligne fusionnée
    """

    # Recherche du bien principal dans la mutation
    iprinc = reduce_get_id(df)
    mut_princ = df.iloc[iprinc]
    values = []
    # id_mutation
    values.append(mut_princ.id_mutation)
    date = mut_princ.date_mutation.split("-")
    # jour_mutation
    values.append(date[2])
    # mois_mutation
    values.append(date[1])
    # annee_mutation
    values.append(date[0])
    # nature_mutation
    values.append(mut_princ.nature_mutation)
    # valeur_fonciere
    values.append(mut_princ.valeur_fonciere)
    # adresse_numero
    values.append(mut_princ.adresse_numero)
    # adresse_suffixe
    values.append(mut_princ.adresse_suffixe)
    # adresse_nom_voie
    values.append(mut_princ.adresse_nom_voie)
    # adresse_code_voie
    values.append(mut_princ.adresse_code_voie)
    # code_postal
    values.append(mut_princ.code_postal)
    # nom_commune
    values.append(mut_princ.nom_commune)
    # code_departement
    values.append(mut_princ.code_departement)
    # id_parcelle
    values.append(mut_princ.id_parcelle)
    # surface_carrez_total
    cols_carrez = [
        "lot1_surface_carrez",
        "lot2_surface_carrez",
        "lot3_surface_carrez",
        "lot4_surface_carrez",
        "lot5_surface_carrez",
    ]
    values.append(df[cols_carrez].sum().sum())
    # surface_reelle_bati_total
    values.append(df.surface_reelle_bati.sum())
    # surface_terrain_total
    values.append(df.surface_terrain.sum())
    # nombre_lots
    values.append(df.nombre_lots.sum())

    # nombre_maisons
    # surface_carrez_maisons
    # surface_reelle_bati_maisons
    # surface_terrain_maisons
    def sum_surface(indice: int) -> None:
        """Somme les surfaces de plusieurs biens

        Args:
            indice (int): Id de la mutation
        """
        # nombre_local
        type_locaux = df.code_type_local.value_counts()
        values.append(0 if indice not in type_locaux else type_locaux[indice])
        # surface_carrez_local
        values.append(df[df.code_type_local == indice][cols_carrez].sum().sum())
        # surface_reelle_bati_local
        values.append(df[df.code_type_local == indice].surface_reelle_bati.sum().sum())
        # surface_terrain_local
        values.append(df[df.code_type_local == indice].surface_terrain.sum().sum())

    sum_surface(1)
    # nombre_appartements
    # surface_carrez_appartements
    # surface_reelle_bati_appartements
    # surface_terrain_appartements
    sum_surface(2)
    # nombre_dependences
    # surface_carrez_dependences
    # surface_reelle_bati_dependences
    # surface_terrain_dependences
    sum_surface(3)
    # nombre_pieces_principales
    values.append(df.nombre_pieces_principales.sum())
    # nature_culture
    values.append(mut_princ.nature_culture)
    # nature_culture_speciale
    values.append(mut_princ.nature_culture_speciale)
    # longitude
    values.append(mut_princ.longitude)
    # latitude
    values.append(mut_princ.latitude)
    return pd.Series({k: v for k, v in zip(_alt_df_cols, values)}, index=_alt_df_cols)


def applyParallel(dfGrouped: Iterable, func: Callable) -> pd.DataFrame:
    retLst = Parallel(n_jobs=-1)(delayed(func)(group) for _, group in dfGrouped)
    return pd.DataFrame(retLst, columns=_alt_df_cols)


def convert_type(df: pd.DataFrame) -> pd.DataFrame:
    """Change en ligne les types de certaines colonnes
    pour corriger certains imports invalides du csv

    Args:
        df (pd.DataFrame): Dataframe à traiter
    Returns:
        pd.DataFrame: Dataframe avec les types corrects
    """
    surface_columns = (c for c in df.columns if c.startswith("surface_"))
    nombre_columns = (c for c in df.columns if c.startswith("nombre_"))

    df.code_departement = df.code_departement.astype("str")

    df = df.astype(
        {
            **{c: "uint16" for c in nombre_columns},
            **{c: "float32" for c in surface_columns},
            **dict(
                code_postal="uint32",
                code_departement="category",
                nature_mutation="category",
                adresse_suffixe="category",
                nature_culture="category",
                nature_culture_speciale="category",
                jour_mutation="uint8",
                mois_mutation="uint8",
                annee_mutation="uint16",
                adresse_numero="uint16",
            ),
        },
    )
    return df


def clean(df_path: str, new_path: str, format_input: str = "csv") -> None:
    """Nettoie un fichier dvf issu csv original en un fichier parquet

    Args:
        df_path (str): Chemin du fichier csv contenant les données
        new_path (str): Chemin du fichier parquet de sortie
        format_input (str, optional): Format de l'entree. Default "csv".
    """

    match format_input:
        case "csv":
            df = pd.read_csv(df_path, low_memory=False)
        case "parquet":
            df = pd.read_parquet(df_path)
        case "pickle":
            df = pd.read_pickle(df_path)
        case _:
            raise ValueError(f"Format {format_input} non reconnu")
    df = missing_values(df)
    df = applyParallel(df.groupby("id_mutation"), fusion_data)
    df.set_index("id_mutation", inplace=True)
    df = convert_type(df)
    df.to_parquet(new_path, compression="brotli")
