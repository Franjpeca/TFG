import sys
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')
import mord


def MORD_LogisticAT(dataset, params):
    print("ESTOY DENTRO DEL NODO !!!!")
    model = mord.LogisticAT(**params)  # los kwargs llegan directamente
    print(params)
    #model.fit(dataset["X"], dataset["y"])
    return model