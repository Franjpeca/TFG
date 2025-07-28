import sys
import logging
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ORCA
sys.path.append("/home/fran/TFG/proyecto-ola/orca-python")
from orca_python.classifiers import SVOREX

logger = logging.getLogger(__name__)

def Train_ORCA_SVOREX(dataset, params, cv_settings, model_id, dataset_id):
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]

    # mapeo de etiquetas de 1 a 5 (SVOREX lo requiere)
    label_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    y = pd.Series(y_raw).map(label_mapping).astype(int).values

    logger.info(f"\n[Training] Entrenando ORCA-SVOREX con GridSearch (MAE) sobre el dataset: {dataset_id} ...")

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
        estimator=SVOREX(),
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