import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord

def LogisticIT(dataset, params):
    
    # Crear el modelo vacío con los parámetros
    model = mord.LogisticIT(
        alpha=params["alpha"], 
        max_iter=params["max_iter"], 
        tol=params["tol"], 
        fit_intercept=params["fit_intercept"]
    )

    # Comentamos el .fit() por ahora, el modelo no se entrena
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacío (sin entrenar)
    return model