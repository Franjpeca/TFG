import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/fran/TFG/proyecto-ola/orca-python')
import mord

def MORD_MulticlassLogistic(dataset, params):

    # Crear el modelo vacío con los parámetros
    model = mord.MulticlassLogistic(
        alpha=params["alpha"], 
        max_iter=params["max_iter"], 
        tol=params["tol"]
    )

    # Comentamos el .fit() por ahora, ya que no entrenamos el modelo
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacío (sin entrenar)
    return model