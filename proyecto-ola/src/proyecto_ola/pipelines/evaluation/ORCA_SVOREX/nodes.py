import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

# ORCA
sys.path.append("/home/fran/TFG/proyecto-ola/orca-python")
from orca_python.classifiers import SVOREX

logger = logging.getLogger(__name__)

def amae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    per_class_errors = []
    for c in classes:
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue
        per_class_errors.append(np.mean(np.abs(y_true[idx] - y_pred[idx])))
    return np.mean(per_class_errors)

def Evaluate_ORCA_SVOREX(model, dataset, model_id, model_type, dataset_id):
    logger.info(f"\n[Evaluating] Evaluando modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    # separar X / y y mapear etiquetas de 1 a 5 (igual que en entrenamiento)
    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]
    label_mapping = model.label_mapping  # {"A":1, ..., "E":5}
    y = pd.Series(y_raw).map(label_mapping).astype(int).values

    # escalar
    X_scaled = model.scaler.transform(X)

    # predecir
    y_pred = model.predict(X_scaled)  # vectores de 1..5

    logger.info(f"[Evaluating] Predicciones (primeros 10): {y_pred[:10]}")
    real_dist = dict(pd.Series(y).value_counts().sort_index())
    pred_dist = dict(pd.Series(y_pred).value_counts().sort_index())
    logger.info(f"[Evaluating] Distribución real (y): {real_dist}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {pred_dist}")

    # métricas nominales
    nominal_metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score":  f1_score(y, y_pred, average="weighted")
    }
    # métricas ordinales
    ordinal_metrics = {
        "qwk":  cohen_kappa_score(y, y_pred, weights="quadratic"),
        "mae":  mean_absolute_error(y, y_pred),
        "amae": amae(y, y_pred)
    }

    logger.info(f"[Evaluating] Métricas nominales:\n\t{nominal_metrics}")
    logger.info(f"[Evaluating] Métricas ordinales:\n\t{ordinal_metrics}")

    return {
        "model_id":    f"{model_type}(" + ", ".join(f"{k}={v}" for k, v in model.get_params().items()) + ")",
        "dataset_id":  dataset_id,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics
    }