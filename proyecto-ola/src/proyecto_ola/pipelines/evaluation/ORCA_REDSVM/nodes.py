import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

from proyecto_ola.utils.nodes_utils import amae

logger = logging.getLogger(__name__)

def Predict_ORCA_REDSVM(model, dataset, model_id, dataset_id):
    logger.info(f"[Evaluating] Prediciendo con el modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\t{dataset_id}\n\n")
    
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    y_pred = model.predict(X.to_numpy())

    y_pred_list = [int(v) - 1 for v in np.asarray(y_pred).tolist()]
    y_true_list = [int(v) for v in np.asarray(y).tolist()]

    return (
        {"y_pred": y_pred_list, "y_true": y_true_list},
        y_true_list,
        model.get_params(),
    )

def Evaluate_ORCA_REDSVM(y_true, y_pred, model_params, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"[Evaluating] Evaluando modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    logger.info(f"[Evaluating] Carpeta de ejecución:\n\t{execution_folder}\n")

    if isinstance(y_pred, dict) and "y_pred" in y_pred:
        if "y_true" in y_pred:
            y_true = y_pred["y_true"]
        y_pred = y_pred["y_pred"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    logger.info(f"[Evaluating] Predicciones (primeros 10): {y_pred[:10]}")
    logger.info(f"[Evaluating] Distribución real (y): {dict(pd.Series(y_true).value_counts().sort_index())}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {dict(pd.Series(y_pred).value_counts().sort_index())}\n")

    nominal_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted")
    }

    ordinal_metrics = {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
        "amae": amae(y_true, y_pred)
    }

    model_id_str = f"{model_type}(" + ", ".join(f"{k}={v}" for k, v in model_params.items()) + ")"

    try:
        combo_id = "_".join(model_id.split("_")[1:3])
    except IndexError:
        combo_id = "unknown"

    results = {
        "model_id": model_id_str,
        "grid_id": combo_id,
        "dataset_id": dataset_id,
        "execution_folder": execution_folder,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics
    }

    logger.info(f"[Evaluating] model_id: {model_id_str}")
    logger.info(f"[Evaluating] Métricas nominales:\n\t{nominal_metrics}")
    logger.info(f"[Evaluating] Métricas ordinales:\n\t{ordinal_metrics}\n\n")
    
    return results
