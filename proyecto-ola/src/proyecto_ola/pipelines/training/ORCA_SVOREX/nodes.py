import sys
import logging
import pandas as pd
import numpy as np
import torch
import os, random

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from orca_python.classifiers import SVOREX

logger = logging.getLogger(__name__)

def seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def Train_ORCA_SVOREX(dataset, params, cv_settings, model_id, dataset_id):
    random_state = params.get("random_state", 42)
    seed_everywhere(random_state)

    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_fit = dataset.iloc[:, -1].map({"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}).astype(int).values

    logger.info(f"[Training] Entrenando ORCA-SVOREX con GridSearch (MAE) con el dataset: {dataset_id} ...")
    logger.info(f"[Training] Model id: {model_id} ...\n")

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

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model