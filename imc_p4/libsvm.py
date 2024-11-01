#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:14:36 2017

@author: pedroa
"""

# This script contains all exercise of the practical assignment. If you want to execute an specific exercise part, just comment/uncomment the lines
# that are needed for that exercise.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import svm
from sklearn import preprocessing
from sklearn import model_selection

## Load the dataset ##
data = pd.read_csv('dataset3.csv',header=None)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# Standarization
scale = preprocessing.StandardScaler() # Creation of the object. Line for question 8, 9 and 11


X_train_standarizated = scale.fit_transform(X,y)



## Data splitting ##   
# # This values can be different depending on the question
x_train , x_test , y_train , y_test = model_selection.train_test_split(X_train_standarizated, y, stratify = y, test_size = 0.25, random_state = 10)



## Train the SVM model ##

svm_model = svm.SVC(kernel='linear', C=1000)   # Linea for question 8
#svm_model = svm.SVC(kernel='rbf', C=100, gamma=0.2)    # Linea for question 8
#svm_model = svm.SVC(kernel = 'rbf')



## K-fold part ##
Cs = np.logspace(-5, 15, num=11, base=2)
Gs = np.logspace(-15, 3, num=9, base=2)
# We are using SVM as estimator, and we use C and Gamma as hyperparameters. 
optimo = model_selection.GridSearchCV(estimator=svm_model, param_grid=dict(C=Cs,gamma=Gs), n_jobs=-1,cv=5) # Line for question 9


## Training ##
optimo.fit(x_train, y_train)


# Fit the model without GridSearch # Exercise 11 part
svm_model.fit(x_train, y_train)



## CCR Score ##
print("CCR: ", optimo.score(x_test, y_test))



## This is the graphica part ##
### Graph and plot part ###
# Plot the points
X = X_train_standarizated
plt.figure(1)
plt.clf()
plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired)

# Plot the support vectors class regions, the separating hyperplane and the margins
plt.axis('tight')
# |->Plot support vectors
plt.scatter(svm_model.support_vectors_[:,0], svm_model.support_vectors_[:,1], 
            marker='+', s=100, zorder=10, cmap=plt.cm.Paired)
# |-> Extract the limits
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
# |-> Create a grid with all the points and then obtain the SVM 
#    score for all the points
XX, YY = np.mgrid[x_min:x_max:500j, y_min:y_max:500j]
Z = svm_model.decision_function(np.c_[XX.ravel(), YY.ravel()])
# |-> Plot the results in a countour
Z = Z.reshape(XX.shape)
plt.pcolormesh(XX, YY, Z > 0)
plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], 
            linestyles=['--', '-', '--'], levels=[-1, 0, 1])

plt.xlabel('x1')
plt.ylabel('x2')



plt.show()