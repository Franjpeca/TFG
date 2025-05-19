import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import mord

logger = logging.getLogger(__name__)

def Train_MORD_OrdinalRidge(dataset, params, param_type, cv_settings, dataset_id):
    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    # Mapear etiquetas si son strings
    if y_raw.dtype == 'O':
        label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        y_mapped = y_raw.map(label_mapping).astype(int)
    else:
        y_mapped = y_raw

    # Aseguramos que sean arrays para evitar warnings
    X = np.asarray(X)
    y_mapped = np.asarray(y_mapped)

    logger.info(f"\n[Training] Entrenando OrdinalRidge con GridSearch (MAE) con el dataset: {dataset_id} ...")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    search = GridSearchCV(
        estimator=mord.OrdinalRidge(),
        param_grid=params,
        cv=cv,
        scoring="neg_mean_absolute_error",  # MAE negativo
        n_jobs=-1
    )
    search.fit(X, y_mapped)

    best_model = search.best_estimator_
    if 'label_mapping' in locals():
        best_model.label_mapping = label_mapping

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")

    return best_model