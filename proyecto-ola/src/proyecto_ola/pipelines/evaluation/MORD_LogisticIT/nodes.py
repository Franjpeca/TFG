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

def Predict_MORD_LogisticIT(model, dataset, model_id, dataset_id):
    logger.info(f"\n[Evaluating] Prediciendo con el modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    y_pred = model.predict(X)

    y_pred_list = [int(v) for v in y_pred.tolist()]
    y_true_list = [int(v) for v in np.asarray(y).tolist()]

    return (
        {"y_pred": y_pred_list, "y_true": y_true_list},
        y_true_list,
        model.get_params(),
    )

def Evaluate_MORD_LogisticIT(y_true, y_pred, model_params, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"\n[Evaluating] Evaluando el modelo:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    logger.info(f"[Evaluating] Carpeta de ejecución:\n\t{execution_folder}")

    if isinstance(y_pred, dict) and "y_pred" in y_pred:
        if "y_true" in y_pred:
            y_true = y_pred["y_true"]
        y_pred = y_pred["y_pred"]

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    logger.info(f"[Evaluating] Distribución real (y): {dict(pd.Series(y_true).value_counts().sort_index())}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {dict(pd.Series(y_pred).value_counts().sort_index())}")

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

    results = {
        "model_id": model_id_str,
        "dataset_id": dataset_id,
        "execution_folder": execution_folder,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics,
    }

    logger.info(f"[Evaluating] model_id: {model_id_str}")
    logger.info(f"[Evaluating] Métricas nominales:\n\t{nominal_metrics}")
    logger.info(f"[Evaluating] Métricas ordinales:\n\t{ordinal_metrics}")
    return results