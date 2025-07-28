import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
sys.path.append('/home/fran/TFG/proyecto-ola/orca-python')

import orca_python
from orca_python.classifiers import OrdinalDecomposition

logger = logging.getLogger(__name__)

def Train_ORCA_OrdinalDecomposition(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    y = pd.Series(y_raw).map(label_mapping).astype(int).values

    logger.info(f"\n[Training] Entrenando ORCA-OrdinalDecomposition con GridSearch (MAE) con el dataset: {dataset_id} ...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    torch.manual_seed(cv_settings["random_state"])
    np.random.seed(cv_settings["random_state"])

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    search = GridSearchCV(
        estimator=OrdinalDecomposition(),
        param_grid=params,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    search.fit(X_scaled, y)

    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = scaler

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")

    return best_model