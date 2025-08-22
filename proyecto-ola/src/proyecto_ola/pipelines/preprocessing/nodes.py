import logging
from typing import Tuple, Dict
import pandas as pd

logger = logging.getLogger(__name__)

def clean_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    preprocessing_params: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cleaned_train = clean_data(train_df, subset_name="train", dataset_name=dataset_name, params=preprocessing_params)
    cleaned_test  = clean_data(test_df, subset_name="test", dataset_name=dataset_name, params=preprocessing_params)

    return cleaned_train, cleaned_test


def clean_data(
    df: pd.DataFrame,
    subset_name: str,
    dataset_name: str,
    params: Dict,
) -> pd.DataFrame:

    if df.empty:
        logger.warning(f"[Preprocessing] DataFrame vacío en subset {subset_name} del dataset {dataset_name}.")

    out = df.copy()

    # Eliminar filas vacías
    out = out.dropna()

    # Eliminar columnas completamente vacías
    if params.get("drop_empty_cols", True):
        empty_cols = out.columns[out.isna().all()].tolist()
        if empty_cols:
            out = out.drop(columns=empty_cols)

    # Eliminar filas duplicadas
    if params.get("drop_duplicates", True):
        before = out.shape[0]
        out = out.drop_duplicates()
        after = out.shape[0]
        if before != after:
            logger.info(f"[Preprocessing] Se eliminaron {before - after} filas duplicadas en subset {subset_name} del dataset {dataset_name}.")

    # Eliminar columnas constantes
    if params.get("drop_constant_cols", True):
        nunique = out.nunique(dropna=False)
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            out = out.drop(columns=constant_cols)

    # Eliminar penúltima columna si se solicita
    if params.get("drop_penultimate", True) and out.shape[1] >= 2:
        col_to_drop = out.columns[-2]
        out = out.drop(columns=[col_to_drop])

    logger.info(f"[Preprocessing] Dataset limpiado (alias): {dataset_name} - Tipo: {subset_name}\n")

    return out
