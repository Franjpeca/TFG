import sys
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ORCA path
sys.path.append('/home/fran/TFG/proyecto-ola/orca-python')

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


def Predict_ORCA_REDSVM(model, dataset, model_id, dataset_id):
    logger.info(f"\n[Evaluating] Prediciendo con ORCA-REDSVM:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_scaled = model.scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return y_pred # No hace falta to list porque en principio devuelve ya una lista


def Evaluate_ORCA_REDSVM(model, dataset, y_pred, model_id, model_type, dataset_id):
    logger.info(f"\n[Evaluating] Evaluando modelo ORCA-REDSVM:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    logger.info(f"[Evaluating] Predicciones (primeros 10): {y_pred[:10]}")
    logger.info(f"[Evaluating] Distribución real (y): {dict(pd.Series(y).value_counts().sort_index())}")
    logger.info(f"[Evaluating] Distribución predicha (y_pred): {dict(pd.Series(y_pred).value_counts().sort_index())}")

    nominal_metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average="weighted")
    }

    ordinal_metrics = {
        "qwk": cohen_kappa_score(y, y_pred, weights="quadratic"),
        "mae": mean_absolute_error(y, y_pred),
        "amae": amae(y, y_pred)
    }

    results = {
        "model_id": f"{model_type}(" + ", ".join(f"{k}={v}" for k, v in model.get_params().items()) + ")",
        "dataset_id": dataset_id,
        "nominal_metrics": nominal_metrics,
        "ordinal_metrics": ordinal_metrics
    }

    logger.info(f"[Evaluating] Métricas nominales:\n\t{nominal_metrics}")
    logger.info(f"[Evaluating] Métricas ordinales:\n\t{ordinal_metrics}")

    return results