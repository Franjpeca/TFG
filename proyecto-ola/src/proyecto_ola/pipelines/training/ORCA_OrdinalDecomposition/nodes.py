import logging
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from orca_python.classifiers import OrdinalDecomposition

from proyecto_ola.utils.nodes_utils import seed_everywhere, qwk_scorer

logger = logging.getLogger(__name__)

def Train_ORCA_OrdinalDecomposition(dataset, params, cv_settings, model_id, dataset_id):
    random_state = params.get("random_state", 42)
    jobs = cv_settings.get("n_jobs", 1)

    seed_everywhere(random_state)

    X = dataset.iloc[:, :-1].to_numpy(dtype=np.float64)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    y = pd.Series(y_raw).map(label_mapping).astype(int).values

    logger.info(f"[Training] Entrenando ORCA-OrdinalDecomposition con GridSearch (QWK) con el dataset: {dataset_id} ...")
    logger.info(f"[Training] Model id: {model_id} ...\n")

    torch.manual_seed(cv_settings["random_state"])
    np.random.seed(cv_settings["random_state"])

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=cv_settings.get("shuffle", True),
        random_state=cv_settings["random_state"]
    )

    pipe = Pipeline(steps=[
        ("scaler", RobustScaler()),
        ("model", OrdinalDecomposition()),
    ])

    param_grid = {f"model__{k}": v for k, v in params.items()}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=qwk_scorer,
        n_jobs=jobs
    )
    search.fit(X, y)

    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]

    best_score = search.best_score_

    logger.info(f"[Training] Mejor QWK obtenido: {best_score:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model
