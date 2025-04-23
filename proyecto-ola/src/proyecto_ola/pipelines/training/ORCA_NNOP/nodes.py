import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import orca_python
from orca_python.classifiers import OrdinalDecomposition

def ORCA_NNOP(dataset, params):
    
    # Crear el modelo vacío con los parámetros
    model = orca_python.classifiers.NNOP(
        learning_rate=params["learning_rate"], 
        hidden_layer_size=params["hidden_layer_size"], 
        max_iter=params["max_iter"],
        alpha=params["alpha"]
    )

    # Comentamos el .fit() por ahora, ya que no entrenamos el modelo
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacío (sin entrenar)
    return model