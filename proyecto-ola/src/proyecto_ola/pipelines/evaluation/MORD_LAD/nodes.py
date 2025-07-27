import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ORCA path
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')

# Logger
logger = logging.getLogger(__name__)

# Metodo de ayuda, no es un nodo como tal
def amae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)

    per_class_errors = []
    for c in classes:
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue
        class_error = np.mean(np.abs(y_true[idx] - y_pred[idx]))
        per_class_errors.append(class_error)

    return np.mean(per_class_errors)

def Predict_MORD_LAD(model, dataset, model_id, dataset_id):
    logger.info(f"\n[Evaluating] Prediciendo con el modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Predicción con redondeo y recorte al rango esperado
    y_pred_raw = model.predict(X.values)
    y_pred = np.clip(np.round(y_pred_raw), 0, 4).astype(int)

    logger.info(f"[Evaluating] DEBUG Predicciones (primeros 10): {y_pred[:10]}")
    return y_pred.tolist()

def Evaluate_MORD_LAD(model, dataset, y_pred, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"\n[Evaluating] Evaluando modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    logger.info(f"[Evaluating] Carpeta de ejecución:\n\t{execution_folder}")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    y_pred = np.asarray(y_pred)

    logger.info(f"[Evaluating] DEBUG Distribución real (y): {dict(pd.Series(y).value_counts().sort_index())}")
    logger.info(f"[Evaluating] DEBUG Distribución predicha (y_pred): {dict(pd.Series(y_pred).value_counts().sort_index())}")

    # Métricas nominales
    nominal_metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }

    # Métricas ordinales
    ordinal_metrics = {
        "qwk": cohen_kappa_score(y, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y, y_pred),
        "amae": amae(y, y_pred),
    }

    results = {
        "model_id": f"{model_type}(" + ", ".join(f"{k}={v}" for k, v in model.get_params().items()) + ")",
        "dataset_id": dataset_id,
        "execution_folder": execution_folder,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics,
    }

    logger.info(f"[Evaluating] Métricas nominales:\n\t{nominal_metrics}")
    logger.info(f"[Evaluating] Métricas ordinales:\n\t{ordinal_metrics}")
    return results