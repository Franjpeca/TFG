import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

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
        class_error = np.mean(np.abs(y_true[idx] - y_pred[idx]))
        per_class_errors.append(class_error)

    return np.mean(per_class_errors)

def Predict_CLASSIC_LinearRegression(model, dataset, model_id, dataset_id):
    logger.info(f"[Evaluating] Prediciendo con el modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\t{dataset_id}\n\n")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # mantener exactamente la misma forma: si es objeto, codificamos
    # usando el mapping del entrenamiento si existe; si no, LabelEncoder
    if y.dtype == 'O':
        if hasattr(model, "label_mapping"):
            mapping = model.label_mapping
            y = y.map(mapping).astype(int)
        else:
            y = LabelEncoder().fit_transform(y)

    # Predicción especial con redondeo y recorte al rango 0-4
    y_pred_raw = model.predict(X.values)
    y_pred = np.clip(np.round(y_pred_raw), 0, 4).astype(int)

    return y_pred.tolist(), y.tolist(), model.get_params()

def Evaluate_CLASSIC_LinearRegression(y_true, y_pred, model_params, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"[Evaluating] Evaluando modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    logger.info(f"[Evaluating] Carpeta de ejecución:\n\t{execution_folder}\n")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    logger.info(f"[Evaluating] Distribución real (y): {dict(pd.Series(y_true).value_counts().sort_index())}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {dict(pd.Series(y_pred).value_counts().sort_index())}\n")

    nominal_metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }

    ordinal_metrics = {
        "qwk": cohen_kappa_score(y_true, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y_true, y_pred),
        "amae": amae(y_true, y_pred),
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