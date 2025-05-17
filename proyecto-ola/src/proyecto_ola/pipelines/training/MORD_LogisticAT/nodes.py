import sys
import os
import pandas as pd
import numpy as np
import logging
import mord
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.path.append('/home/fran/TFG/proyecto-ola/orca-python')

logger = logging.getLogger(__name__)

def Train_MORD_LogisticAT(dataset, params, param_type, cv_settings, dataset_id):
    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    if param_type == "param_grid":
        logger.info(f"\n[Training] Entrenando LogisticAT con GridSearch (MAE) con el dataset: {dataset_id} ...")

        cv = StratifiedKFold(
            n_splits=cv_settings["n_splits"],
            shuffle=True,
            random_state=cv_settings["random_state"]
        )

        search = GridSearchCV(
            estimator=mord.LogisticAT(),
            param_grid=params,
            cv=cv,
            scoring="neg_mean_absolute_error",  # MAE (negativo)
            n_jobs=-1
        )
        search.fit(X, y_mapped)
        best_model = search.best_estimator_
        best_model.label_mapping = label_mapping

        # Mostrar el mejor MAE encontrado (recuerda cambiar el signo)
        logger.info(f"[Training] Mejor MAE obtenido: {-search.best_score_:.5f}")
        logger.info(f"[Training] Mejor modelo obtenido:\n\t{best_model}")
        return best_model

    else:  # Sin grid search, entrenamiento directo
        logger.info("[Training] Entrenando LogisticAT sin GridSearch (MAE)...")
        model = mord.LogisticAT(**params)
        model.fit(X, y_mapped)
        model.label_mapping = label_mapping

        # Calcular MAE en training
        y_pred = model.predict(X)
        mae = mean_absolute_error(y_mapped, y_pred)
        logger.info(f"[Training] MAE en training: {mae:.5f}")
        logger.info(f"[Training] Modelo entrenado:\n\t{model}")
        return model