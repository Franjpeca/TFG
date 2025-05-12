import sys
import os
import pandas as pd
import numpy as np
import logging
import mord
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')

logger = logging.getLogger(__name__)

def MORD_LogisticAT(dataset, params, param_type, cv_settings):

    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    if param_type == "param_grid":
        logger.info("[Training] Entrenando LogisticAT con GridSearch...")

        cv = StratifiedKFold(
            n_splits=cv_settings["n_splits"],
            shuffle=True,
            random_state=cv_settings["random_state"]
        )

        search = GridSearchCV(
            estimator=mord.LogisticAT(),
            param_grid=params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )
        search.fit(X, y_mapped)
        best_model = search.best_estimator_
        best_model.label_mapping = label_mapping
        logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")
        return best_model

    else:
        logger.info("[Training] Entrenando LogisticAT con GridSearch...")
        model = mord.LogisticAT(**params)
        model.fit(X, y_mapped)
        model.label_mapping = label_mapping
        logger.info(f"[Training] Modelo entrenado:\n\t{model}")
        return model