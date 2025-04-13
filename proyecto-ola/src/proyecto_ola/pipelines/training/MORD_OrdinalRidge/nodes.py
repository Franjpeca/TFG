import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import orca_python
from orca_python.classifiers import OrdinalDecomposition

def MORD_OrdinalRidge(*datasets):
    # Imprimir un mensaje indicando que la función fue llamada
    print("\n ==> Funcion MORD_OrdinalRidge")
    
    # Crear un DataFrame vacío
    empty_dataframe = pd.DataFrame()

    # Devuelve el DataFrame vacío dentro de una lista o tupla
    return [empty_dataframe]