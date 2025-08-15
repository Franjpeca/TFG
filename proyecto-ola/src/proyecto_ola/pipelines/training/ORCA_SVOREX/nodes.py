import sys
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from orca_python.classifiers import SVOREX

logger = logging.getLogger(__name__)


def Train_ORCA_SVOREX(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_fit = dataset.iloc[:, -1].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}).astype(int).values

    logger.info(f"\n[Training] Entrenando ORCA-SVOREX con GridSearch (MAE) con el dataset: {dataset_id} ...")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", SVOREX())
    ])

    param_grid = {f"model__{k}": v for k, v in params.items()}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    search.fit(X, y_fit)

    best_model = search.best_estimator_

    logger.info(f"[Training] Mejor MAE: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo: {best_model}")

    return best_model