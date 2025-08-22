import logging
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from orca_python.classifiers.NNOP import NNOP
from sklearn.pipeline import Pipeline

from proyecto_ola.utils.nodes_utils import seed_everywhere

logger = logging.getLogger(__name__)




def Train_ORCA_NNOP(dataset, params, cv_settings, model_id, dataset_id):
    random_state = params.get("random_state", 42)
    seed_everywhere(random_state)

    X_raw = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    y = y_raw.map(label_mapping).astype(int).values

    logger.info(f"[Training] Entrenando ORCA-NNOP con GridSearch (MAE) con el dataset: {dataset_id} ...")
    logger.info(f"[Training] Model id: {model_id} ...\n")

    cv = StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=True,
        random_state=cv_settings["random_state"]
    )

    torch.manual_seed(cv_settings["random_state"])
    np.random.seed(cv_settings["random_state"])

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", NNOP()),
    ])

    param_grid = {f"model__{k}": v for k, v in params.items()}

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    search.fit(X_raw, y)

    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]

    logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model