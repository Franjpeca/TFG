import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord

def MORD_OrdinalRidge(dataset, params):
    
    # Crear el modelo vacío con los parámetros
    model = mord.OrdinalRidge(
        alpha=params["alpha"], 
        max_iter=params["max_iter"]
    )

    # Comentamos el .fit() por ahora, ya que no entrenamos el modelo
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacío (sin entrenar)
    return model