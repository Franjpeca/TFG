import os
import random
import numpy as np

from sklearn.metrics import cohen_kappa_score

try:
    import torch
except ImportError:
    torch = None

def seed_everywhere(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False



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


def qwk_scorer(estimator, X, y_true):
    # 0..K-1 según y_true del fold
    y_min, y_max = int(np.min(y_true)), int(np.max(y_true))
    labels = list(range(y_min, y_max + 1))

    y_pred = estimator.predict(X)
    # Si es continuo (regresor), redondea y recorta
    if np.issubdtype(np.asarray(y_pred).dtype, np.floating):
        y_pred = np.rint(y_pred).astype(int)
        y_pred = np.clip(y_pred, y_min, y_max)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)