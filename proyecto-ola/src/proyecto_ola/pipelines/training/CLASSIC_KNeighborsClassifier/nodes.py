import sys
import os
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

def Train_CLASSIC_KNeighborsClassifier(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    logger.info(f"[Training] Entrenando KNeighborsClassifier con GridSearch (MAE) con el dataset: {dataset_id} ...\n")
    logger.info(f"[Training] Model id: {model_id} ...\n")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])

    logger.info(f"[Training] Realizando validación cruzada (neg_mean_absolute_error)...\n")
    search = GridSearchCV(
        estimator=pipe,
        param_grid={f"model__{k}": v for k, v in params.items()},
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1
    )

    search.fit(X, y_mapped)
    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model