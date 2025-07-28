import logging
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from orca_python.classifiers.NNOP import NNOP

logger = logging.getLogger(__name__)


def Train_ORCA_NNOP(dataset, params, cv_settings, model_id, dataset_id):
    X_raw = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    y = y_raw.map(label_mapping).astype(int).values

    # Normalizacion (peta si no lo pongo)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    logger.info(f"\n[Training] Entrenando ORCA-NNOP con GridSearch (MAE) con el dataset: {dataset_id} ...")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    torch.manual_seed(cv_settings["random_state"])
    np.random.seed(cv_settings["random_state"])

    search = GridSearchCV(
        estimator=NNOP(),
        param_grid=params,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = scaler

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")

    return best_model