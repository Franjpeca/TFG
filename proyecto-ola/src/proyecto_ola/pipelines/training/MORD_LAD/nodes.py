import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/home/fran/TFG/proyecto-ola/orca-python')
import mord

def MORD_LAD(dataset, params):

    # Crear el modelo vacio (sin entrenar) utilizando los parametros
    model = mord.LAD(
        alpha=params["alpha"], 
        max_iter=params["max_iter"]
    )

    print(dataset)

    # Aqui dejamos comentado el .fit(), ya que no queremos entrenar el modelo aun
    # model.fit(train, test)  # Comentado por ahora, no entrenamos el modelo

    # Devolvemos el modelo vacio (sin entrenar)
    return model