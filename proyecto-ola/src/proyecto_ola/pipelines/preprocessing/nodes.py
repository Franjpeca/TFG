import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_data_all(*datasets):
    cleaned_list = []
    logger.info("[Preprocessing] Limpiando los datasets ...\n")

    for dataset in datasets:
        cleaned_data = clean_data(dataset)
        cleaned_list.append(cleaned_data)

    return tuple(cleaned_list)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        logger.warning("[Preprocessing] DataFrame vacio!\n")

    data = data.dropna()
    data = data.drop(columns=[data.columns[-2]])
    
    return data


def merge_data(*cleaned_datasets):
    train_datasets = cleaned_datasets[::2]
    test_datasets = cleaned_datasets[1::2]

    merged_train = pd.concat(train_datasets, axis=0, ignore_index=True)
    merged_test = pd.concat(test_datasets, axis=0, ignore_index=True)

    return merged_train, merged_test