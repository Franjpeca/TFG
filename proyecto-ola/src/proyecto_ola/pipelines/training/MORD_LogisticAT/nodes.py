import sys
import os
import pandas as pd
import numpy as np
import logging
import mord
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logger = logging.getLogger(__name__)

def Train_MORD_LogisticAT(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    logger.info(f"[Training] Entrenando LogisticAT con GridSearch (MAE) con el dataset: {dataset_id} ...")
    logger.info(f"[Training] Model id: {model_id} ...\n")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    search = GridSearchCV(
        estimator=mord.LogisticAT(),
        param_grid=params,
        cv=cv,
        scoring="neg_mean_absolute_error",  # MAE negativo
        n_jobs=-1
    )
    search.fit(X, y_mapped)

    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model