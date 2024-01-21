import pandas as pd


def store(df_path: str, new_path: str, format_input: str = "csv") -> None:
    """Sauvegarde un fichier dvf au format parquet

    Args:
        df_path (str): Chemin du fichier contenant les données
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
    df.to_parquet(new_path, compression="brotli")
