#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

# Este script contiene todos los ejercicios de la asignación práctica. Si deseas ejecutar una parte específica, solo comenta/descomenta las líneas necesarias para esa parte.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection
import warnings

# Suprimir solo las advertencias relacionadas con longdouble
warnings.filterwarnings("ignore", category=UserWarning, message=".*longdouble.*")

# Especificar que numpy no muestre advertencias
np.seterr(all='ignore')

## Cargar el dataset ##
data = pd.read_csv('dataset3.csv', header=None)

# Verificación de NaN y Inf
print(data.isna().sum())  # Verificar valores NaN
print((data == np.inf).sum())  # Verificar valores Inf

# Reemplazar Inf por NaN y eliminar filas con NaN
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Asegurarse de que X e y están definidos correctamente
X = np.array(data.iloc[:, :-1].astype(np.float64))  # Suponiendo que la última columna es 'y'
y = np.array(data.iloc[:, -1].astype(np.float64))

# Estandarización
scale = preprocessing.StandardScaler()  # Creación del objeto. Línea para las preguntas 8, 9 y 11
X_train_standarizated = scale.fit_transform(X)

## División de datos ##
# Estos valores pueden variar dependiendo de la pregunta
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_train_standarizated, y, stratify=y, test_size=0.25, random_state=10)

## Entrenamiento del modelo SVM ##
svm_model = svm.SVC(kernel='linear', C=1000)  # Línea para la pregunta 8

## Parte de K-fold ##
Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 3, num=9, base=2)
# Usamos SVM como estimador, y utilizamos C y Gamma como hiperparámetros.
optimo = model_selection.GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs, gamma=Gs), n_jobs=-1, cv=5)  # Línea para la pregunta 9

## Entrenamiento ##
optimo.fit(x_train, y_train)

# Entrenamiento del modelo sin GridSearch (Ejercicio 11 parte)
svm_model.fit(x_train, y_train)

## CCR Score ##
print("CCR: ", optimo.score(x_test, y_test))

## Parte gráfica ##
# Graficar los puntos
X = X_train_standarizated
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Graficar los vectores de soporte, las regiones de clase, el hiperplano separador y los márgenes
plt.axis('tight')
# |-> Graficar los vectores de soporte
plt.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
            marker='+', s=100, zorder=10, cmap=plt.cm.Paired)

# |-> Extraer los límites
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()

# |-> Crear una cuadrícula con todos los puntos y luego obtener la función de decisión del SVM
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])

# |-> Graficar los resultados en un contorno
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'], levels=[-1, 0, 1])

plt.xlabel('x1')
plt.ylabel('x2')

plt.show()
