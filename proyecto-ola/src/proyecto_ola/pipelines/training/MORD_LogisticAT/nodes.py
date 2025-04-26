import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord


def MORD_LogisticAT(dataset, params):
    print(f"Lanzando LogisticAT con {params} !!!!")

    # 1) Separa X e y (pero aún sin mapear)
    X = dataset.iloc[:, :-1]
    y_raw = dataset.iloc[:, -1]

    # 2) Mapea las clases → enteros
    label_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    y_mapped = y_raw.map(label_mapping).astype(int)

    # 3) Entrena con y_mapped
    trained_model = mord.LogisticAT(**params)
    trained_model.fit(X, y_mapped)
    trained_model.label_mapping = label_mapping
    return trained_model