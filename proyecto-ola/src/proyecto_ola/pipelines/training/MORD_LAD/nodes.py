import logging
import numpy as np
import mord

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from proyecto_ola.utils.nodes_utils import seed_everywhere, qwk_scorer

logger = logging.getLogger(__name__)

def Train_MORD_LAD(dataset, params, cv_settings, model_id, dataset_id):
    random_state = params.get("random_state", 42)
    seed_everywhere(random_state)

    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    logger.info(f"[Training] Entrenando LAD con GridSearch (QWK) con el dataset: {dataset_id} ...\n")
    logger.info(f"[Training] Model id: {model_id} ...\n")

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
        scoring=qwk_scorer,
        n_jobs=-1
    )

    search.fit(X, y_mapped)
    best_model = search.best_estimator_
    best_model.label_mapping = label_mapping
    best_model.scaler = best_model.named_steps["scaler"]

    best_score = search.best_score_

    logger.info(f"[Training] Mejor QWK obtenido: {best_score:.5f}")
    logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}\n\n")

    return best_model
