import sys
import os
import pandas as pd
from orca_python import classifiers.OrdinalDecomposition

def orca_functions_node(*datasets):
    # Imprimir un mensaje indicando que la función fue llamada
    print("La función orca_functions_node ha sido ejecutada, pero no se procesó ningún dato.")
    
    # Crear un DataFrame vacío
    empty_dataframe = pd.DataFrame()

    # Devuelve el DataFrame vacío dentro de una lista o tupla
    return [empty_dataframe]