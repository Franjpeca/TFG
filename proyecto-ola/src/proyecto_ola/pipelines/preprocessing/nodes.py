import logging
from typing import Tuple
import pandas as pd

logger = logging.getLogger(__name__)

def clean_pair(
                train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                dataset_name: str,                    # <-- argumento extra
                drop_penultimate: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    logger.info(f"[Preprocessing] Limpiando dataset {dataset_name} (train/test)...\n")
    return clean_data(train_df, drop_penultimate), clean_data(test_df, drop_penultimate)

def clean_data(df: pd.DataFrame, drop_penultimate: bool = True) -> pd.DataFrame:
    if df.empty:
        logger.warning("[Preprocessing] DataFrame vacio!")
    out = df.dropna()
    if drop_penultimate and out.shape[1] >= 2:
        col_to_drop = out.columns[-2]
        out = out.drop(columns=[col_to_drop])
    return out