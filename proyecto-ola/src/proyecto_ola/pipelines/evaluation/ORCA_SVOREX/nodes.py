import sys
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

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

def Predict_ORCA_SVOREX(model, dataset, model_id, dataset_id):
    logger.info(f"\n[Evaluating] Prediciendo con SVOREX:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y_raw = dataset.iloc[:, -1]
    y = pd.Series(y_raw).map(model.label_mapping).astype(int).values

    X_scaled = model.scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return y_pred, y.tolist(), model.get_params()

def Evaluate_ORCA_SVOREX(y_true, y_pred, model_params, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"\n[Evaluating] Evaluando SVOREX:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")
    logger.info(f"[Evaluating] Carpeta de ejecución:\n\t{execution_folder}")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    logger.info(f"[Evaluating] Predicciones (primeros 10): {y_pred[:10]}")
    real_dist = dict(pd.Series(y_true).value_counts().sort_index())
    pred_dist = dict(pd.Series(y_pred).value_counts().sort_index())
    logger.info(f"[Evaluating] Distribución real (y): {real_dist}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {pred_dist}")

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

    return {
        "model_id": model_id_str,
        "dataset_id": dataset_id,
        "execution_folder": execution_folder,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics
    }
