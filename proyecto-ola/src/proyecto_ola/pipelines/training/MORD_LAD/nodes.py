import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord

def MORD_LAD(dataset, params):

    # Crear el modelo vacío (sin entrenar) utilizando los parámetros
    model = mord.LAD(
        alpha=params["alpha"], 
        max_iter=params["max_iter"]
    )

    print(dataset)

    # Aquí dejamos comentado el .fit(), ya que no queremos entrenar el modelo aún
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacío (sin entrenar)
    return model