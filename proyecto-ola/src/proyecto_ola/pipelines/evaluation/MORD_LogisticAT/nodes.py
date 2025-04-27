import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder



def MORD_LogisticAT(model, dataset, model_id, model_type):
    print(f"Evaluando modelo {model_id} de tipo {model_type}...")

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Convertir las etiquetas en números si no lo han sido
    if y.dtype == 'O':  # Si las etiquetas son cadenas (categóricas)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    y_pred = model.predict(X)

    results = {
        "model_id": f"{model_type}(" + ", ".join(f"{k}={v}" for k, v in model.get_params().items()) + ")",
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred, average="weighted"),
    }

    print(f"Mostrando resultados {results}")
    return results