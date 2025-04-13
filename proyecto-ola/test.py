# test.py
import numpy as np
import sys
sys.path.append('/mnt/c/Users/francisco.perez/Desktop/TFG/proyecto-ola/orca-python')

from orca_python.classifiers import OrdinalDecomposition

# Paso 1: Crear el conjunto de datos de entrenamiento y prueba
X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])  # Ejemplo de características de entrenamiento
y_train = np.array([0, 1, 2, 3])  # Etiquetas ordinales (por ejemplo, 0, 1, 2, 3)

X_test = np.array([[5, 6], [6, 7]])  # Ejemplo de características de prueba

# Paso 2: Crear el clasificador OrdinalDecomposition
model = OrdinalDecomposition()  # Sin necesidad de pasar 'model_type'

# Paso 3: Entrenar el modelo
model.fit(X_train, y_train)

# Paso 4: Realizar predicciones sobre los datos de prueba
predictions = model.predict(X_test)

# Paso 5: Imprimir los resultados
print("Predicciones:", predictions)