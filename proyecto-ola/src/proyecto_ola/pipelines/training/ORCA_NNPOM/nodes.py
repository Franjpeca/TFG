
import logging
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from orca_python.classifiers.NNPOM import NNPOM
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
logger = logging.getLogger(__name__)


def Train_ORCA_NNPOM(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    y = y_raw.map(label_mapping).astype(int).values

    logger.info(f"\n[Training] Entrenando ORCA-NNPOM con GridSearch (MAE) con el dataset: {dataset_id} ...")

    scaler = RobustScaler()

    torch.manual_seed(cv_settings["random_state"])
    np.random.seed(cv_settings["random_state"])

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    pipe = Pipeline(steps=[
        ("scaler", scaler),
        ("model", NNPOM()),
    ])

    param_grid = {f"model__{k}": v for k, v in params.items()}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    search.fit(X, y)

    best_model = search.best_estimator_

    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]
    
    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")

    return best_model