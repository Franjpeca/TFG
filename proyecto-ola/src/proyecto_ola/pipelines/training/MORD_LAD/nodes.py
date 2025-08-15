import sys
import os
import pandas as pd
import numpy as np
import logging
import mord
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def Train_MORD_LAD(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    logger.info(f"\n[Training] Entrenando LAD con GridSearch (MAE) con el dataset: {dataset_id} ...")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", mord.LAD()),
    ])

    param_grid = {f"model__{k}": v for k, v in params.items()}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    search.fit(X, y_mapped)
    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")

    return best_model