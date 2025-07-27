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

def Predict_ORCA_NNPOM(model, dataset, model_id, dataset_id):
    logger.info(f"\n[Evaluating] Prediciendo con ORCA-NNPOM:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    X = dataset.iloc[:, :-1].values.astype(np.float32)
    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    X_scaled = model.scaler.transform(X)
    y_pred = model.predict(X_scaled)

    return y_pred.tolist()

def Evaluate_ORCA_NNPOM(model, dataset, y_pred, model_id, model_type, dataset_id, execution_folder):
    logger.info(f"\n[Evaluating] Evaluando modelo ORCA-NNPOM:\n\t{model_id}")
    logger.info(f"[Evaluating] Dataset usado:\n\t{dataset_id}")

    y = dataset.iloc[:, -1]

    if y.dtype == 'O':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    nominal_metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }

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