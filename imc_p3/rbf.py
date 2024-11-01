#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:37:04 2021

IMC: lab assignment 3

@author: pagutierrez
"""
##### Documentacion #####

# Encima de cada funcion se encontraran los enlaces a la documentacion de las funciones de los modulos usados

#########################

# TODO Include all neccesary imports
import pickle # Módulo de python que permite representar objetos en cadenas de bytes. Permite almacenamiento en ficheros o BBDD
import os # Módulo que permite acceder a rutinas o funciones relacionadas con el sistema operativo, sobre todo para lectura y escritura de ficheros
import click # Módulo para la creacion de interfaces en líneas de comandos.

import numpy as np # Modulo numpy, da variables, estructuras y funciones matemáticas
import pandas as pd # Modulo pandas, para manipulación y análisis de datos
import scipy as sp # Modulo scipy, permite realizar algoritmos matematicos
from sklearn.model_selection import train_test_split # Sklearn, para algoritmos y metodos relacionados con el aprendizaje automatico
import sklearn.linear_model 
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error




@click.command() #"Decorator", perimte que la funcion tenga argumentos/opciones dadas por consola
@click.option('--train_file', '-t', default=None, required=False, # Se indica las opciones, si tiene valor por defecto y si es obligatorio
              help=u'Name of the file with training data.') # Información en caso de que se muestre la ayuda por consola

# TODO Include the rest of parameters...

@click.option('--test_file', '-T', default=None, required=False,
              help=u'Name of the file with test data.')
@click.option('--classification',  '-c', is_flag=True,
              help=u'The problem considered is a classification problem.')
@click.option('--ratio_rbf',  '-r', default=0.1, required=False, type=float,
              help=u'Ratio of RBF neurons (as a fraction of 1) with respect to the total number of patterns.')              
@click.option('--l2',  '-l', is_flag=True,
              help=u'Use L2 regularization instead of L1 (logistic regression).')        
@click.option('--eta', '-e', default=None, required=False, type=float,
              help=u'Value of the regularization parameter for logistic regression.')
@click.option('--outputs', '-o', default=1, required=False, type=int,
              help=u'Number of columns that will be used as target variables (all at the end).')
########

@click.option('--pred', '-p', is_flag=True, default=False, show_default=True,
              help=u'Use the prediction mode.') # KAGGLE
@click.option('--model', '-m', default="", show_default=False,
              help=u'Directory name to save the models (or name of the file to load the model, if the prediction mode is active).') # KAGGLE
def train_rbf_total(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model, pred):
    """ 5 executions of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 5 executions with different seeds.
    """

    if not pred:    

        if train_file is None: # Se termina el programa si no hay datos de entrenamiento
            print("You have not specified the training file (-t)")
            return
            
        if test_file is None: # Si el usuario no introduce datos para el test, se usaran los de entrenamiento
            print("No test file introduced, using training data as test data")
            test_file = train_file

        if eta is None: # Valor por defecto de parametro eta
            eta = 0.01

        # np.empty da vectores vacíos del tamaño indicado
        train_mses = np.empty(5) # Error de tipo MSE y el porcentaje de patrones bien clasificados (CCR) para el conjunto de entrenamiento
        train_ccrs = np.empty(5)
        test_mses = np.empty(5) # Lo mismo pero para el conjunto de test
        test_ccrs = np.empty(5)
    
        for s in range(1,6,1):
            print("-----------")
            print("Seed: %d" % s)
            print("-----------")
            np.random.seed(s)
            train_mses[s-1], test_mses[s-1], train_ccrs[s-1], test_ccrs[s-1] = \
                train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, \
                             model and "{}/{}.pickle".format(model, s) or "")
            print("Training MSE: %f" % train_mses[s-1])
            print("Test MSE: %f" % test_mses[s-1])
            print("Training CCR: %.2f%%" % train_ccrs[s-1])
            print("Test CCR: %.2f%%" % test_ccrs[s-1])

        print(" ******************")
        print("Summary of results")
        print(" ******************")
        print("Training MSE: %f +- %f" % (np.mean(train_mses), np.std(train_mses)))
        print("Test MSE: %f +- %f" % (np.mean(test_mses), np.std(test_mses)))
        print("Training CCR: %.2f%% +- %.2f%%" % (np.mean(train_ccrs), np.std(train_ccrs)))
        print("Test CCR: %.2f%% +- %.2f%%" % (np.mean(test_ccrs), np.std(test_ccrs)))
            
    else:
        # KAGGLE
        if model is None:
            print("You have not specified the file with the model (-m).")
            return

        # Obtain the predictions for the test set
        predictions = predict(test_file, model)

        # Print the predictions in csv format
        print("Id,Category")
        for prediction, index in zip(predictions, range(len(predictions))):
            s = ""            
            s += str(index)
            
            if isinstance(prediction, np.ndarray):
                for output in prediction:
                    s += ",{}".format(output)
            else:
                s += ",{}".format(int(prediction))
                
            print(s)
            
##### Documentacion #####
# sklearn.metrics.mean_squared_error : https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
# sklearn.preprocessing.OneHotEncoder : https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
def train_rbf(train_file, test_file, classification, ratio_rbf, l2, eta, outputs, model_file=""):
    """ One execution of RBFNN training
    
        RBF neural network based on hybrid supervised/unsupervised training.
        We run 1 executions.

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        classification: bool
            True if it is a classification problem
        ratio_rbf: float
            Ratio (as a fraction of 1) indicating the number of RBFs
            with respect to the total number of patterns
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
        outputs: int
            Number of variables that will be used as outputs (all at the end
            of the matrix)
        model_file: string
            Name of the file where the model will be written

        Returns
        -------
        train_mse: float
            Mean Squared Error for training data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        test_mse: float 
            Mean Squared Error for test data 
            For classification, we will use the MSE of the predicted probabilities
            with respect to the target ones (1-of-J coding)
        train_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
        test_ccr: float
            Training accuracy (CCR) of the model 
            For regression, we will return a 0
    """
    # Lectura de los ficheros de datos
    train_inputs, train_outputs, test_inputs, test_outputs = read_data(train_file, 
                                                                        test_file,
                                                                        outputs)

    #TODO: Obtain num_rbf from ratio_rbf
    num_rbf = round(ratio_rbf*(len(train_inputs.index))) # Obtenemos el numero de patrones y multiplicamos por el ratio, despues redondeamos
    print("Number of RBFs used: %d" %(num_rbf))

    # 1. Realizamos el clustering a traves de kmeans
    kmeans, distances, centers = clustering(classification, train_inputs, 
                                              train_outputs, num_rbf)

    # 2. Ajuste de los radios RBF
    radii = calculate_radii(centers, num_rbf)
    
    # 3. Arendizaje de los pesos
    # Obtencion de la matriz de salidas de las neuronas RBF
    r_matrix = calculate_r_matrix(distances, radii)

    if not classification:
        # Para el caso de regresion, se usa la matriz de Moore Penrose
        coefficients = invert_matrix_regression(r_matrix, train_outputs)
    else:
        # Modelo de regresion logistica
        logreg = logreg_classification(r_matrix, train_outputs, l2, eta)

    """
    TODO: Obtain the distances from the centroids to the test patterns
          and obtain the R matrix for the test set
    """
    # Como se ha visto en la funcion de obtencion de KMeans, con .transform puedo obtener las distancias con los centros de los patrones de test
    # Los radios son los mismos en ambas matrices
    r_matrix_test = calculate_r_matrix(kmeans.transform(test_inputs), radii)

    # # # # KAGGLE # # # #
    if model_file != "":
        save_obj = {
            'classification' : classification,            
            'radii' : radii,
            'kmeans' : kmeans
        }
        if not classification:
            save_obj['coefficients'] = coefficients
        else:
            save_obj['logreg'] = logreg

        dir = os.path.dirname(model_file)
        if not os.path.isdir(dir):
            os.makedirs(dir)

        with open(model_file, 'wb') as f:
            pickle.dump(save_obj, f)
    
    # # # # # # # # # # #

    if not classification:
        """
        TODO: Obtain the predictions for training and test and calculate
              the MSE
        """
         # Multiplicamos cada matriz por los coeficientes, obteniendo las salidas
        out_prediction_train = np.matmul(r_matrix, coefficients)
        out_prediction_test = np.matmul(r_matrix_test, coefficients)
        # Calculamos el error cuadratico medio (MSE)
        train_mse = mean_squared_error(out_prediction_train, train_outputs)
        test_mse = mean_squared_error(out_prediction_test, test_outputs)

        # En regresion no calculamos los CCR
        train_ccr = 0
        test_ccr = 0

    else:
        """
        TODO: Obtain the predictions for training and test and calculate
              the CCR. Obtain also the MSE, but comparing the obtained
              probabilities and the target probabilities
        """
        # Clase que permite representar los datos categoricos en formato de array con 0 y 1
        encoder = OneHotEncoder()
        train_encoder = encoder.fit_transform(train_outputs).toarray() # Lo pasamos en formato array
        test_encoder = encoder.fit_transform(test_outputs).toarray()

        # Calculo del CCR mediante el metodo score (Hace un predict de forma interna)
        train_ccr = logreg.score(r_matrix, train_outputs.values) * 100
        test_ccr = logreg.score(r_matrix_test, test_outputs.values) * 100

        train_mse = mean_squared_error(train_encoder, logreg.predict_proba(r_matrix))
        test_mse = mean_squared_error(test_encoder, logreg.predict_proba(r_matrix_test))

    return train_mse, test_mse, train_ccr, test_ccr

##### Documentacion #####
# pandas.read_csv : https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# pandas.DataFrame.iloc : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
def read_data(train_file, test_file, outputs):
    """ Read the input data
        It receives the name of the input data file names (training and test)
        and returns the corresponding matrices

        Parameters
        ----------
        train_file: string
            Name of the training file
        test_file: string
            Name of the test file
        outputs: int
            Number of variables to be used as outputs
            (all at the end of the matrix).
              
        Returns
        -------
        train_inputs: array, shape (n_train_patterns,n_inputs)
            Matrix containing the inputs for the training patterns
        train_outputs: array, shape (n_train_patterns,n_outputs)
            Matrix containing the outputs for the training patterns
        test_inputs: array, shape (n_test_patterns,n_inputs)
            Matrix containing the inputs for the test patterns
        test_outputs: array, shape (n_test_patterns,n_outputs)
            Matrix containing the outputs for the test patterns
    """

    #TODO: Complete the code of the function
    matriz_train = pd.read_csv(train_file, header=None) # Se vuelcan los datos en un dataframe
    matriz_test = pd.read_csv(test_file, header=None)

    train_inputs = matriz_train.iloc[:,0:-outputs] # Tomamos las primeras columnas como entradas
    train_outputs = matriz_train.iloc[:,-outputs:] # Tomamos las ultimas columnas como salida

    test_inputs = matriz_test.iloc[:,0:-outputs] # Lo mismo para los datos de test
    test_outputs = matriz_test.iloc[:,-outputs:]

    return train_inputs, train_outputs, test_inputs, test_outputs

##### Documentacion #####
# sklearn train_test_split : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
def init_centroids_classification(train_inputs, train_outputs, num_rbf):
    """ Initialize the centroids for the case of classification
        This method selects in a stratified num_rbf patterns.

        Parameters
        ----------
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        centroids: array, shape (num_rbf,n_inputs)
            Matrix with all the centroids already selected
    """
    
    #TODO: Complete the code of the function
    # Generamos patrones estratificados de forma aleatoria segun las clases indicadas en las salidas, tanto para los inputs como outputs
    train_test_list = train_test_split(train_inputs, train_size=num_rbf, stratify=train_outputs, random_state=None)
    # Tras esto, tendremos que tomar los resultados que nos interesen
    centroids = train_test_list[0]

    return centroids 

##### Documentacion #####
# sklearn.cluster.KMeans : https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def clustering(classification, train_inputs, train_outputs, num_rbf):
    """ It applies the clustering process
        A clustering process is applied to set the centers of the RBFs.
        In the case of classification, the initial centroids are set
        using the method init_centroids_classification(). 
        In the case of regression, the centroids have to be set randomly.

        Parameters
        ----------
        classification: bool
            True if it is a classification problem
        train_inputs: array, shape (n_patterns,n_inputs)
            Matrix with all the input variables
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        kmeans: sklearn.cluster.KMeans
            KMeans object after the clustering
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        centers: array, shape (num_rbf,n_inputs)
            Centers after the clustering
    """

    #TODO: Complete the code of the function
    if classification: 
        # Centroides de forma aleatoria y estratificada
        centroids = init_centroids_classification(train_inputs, train_outputs, num_rbf)
        # Realizamos el algoritmo usando los centroides anteriores y los patrones de entranmiento
        kmeans = KMeans(init=centroids, n_clusters=num_rbf, n_init=1, max_iter=500).fit(train_inputs)
        #kmeans = KMeans(init="k-means++", n_clusters=num_rbf, n_init=1, max_iter=500).fit(train_inputs)
    else:
        # Realizamos lo mismo pero tomando centros de forma aleatoria
        kmeans = KMeans(init="random", n_clusters=num_rbf, n_init=1, max_iter=500).fit(train_inputs)
        #kmeans = KMeans(init="k-means++", n_clusters=num_rbf, n_init=1, max_iter=500).fit(train_inputs)
    # Gracias al metodo .fit hemos podido obtener los centros, ahora simplemente accedemos a ellos
    centers = kmeans.cluster_centers_
    
    # El metodo .transform permite obtener las distancias de los patrones con los centros
    distances = kmeans.transform(train_inputs)

    return kmeans, distances, centers

##### Documentacion #####
# scipy.spatial.distance.pdist : https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
# scipy.spatial.distance.squareform : https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
def calculate_radii(centers, num_rbf):
    """ It obtains the value of the radii after clustering
        This methods is used to heuristically obtain the radii of the RBFs
        based on the centers

        Parameters
        ----------
        centers: array, shape (num_rbf,n_inputs)
            Centers from which obtain the radii
        num_rbf: int
            Number of RBFs to be used in the network
            
        Returns
        -------
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
    """

    #TODO: Complete the code of the function

    # Esta es la parte de la raiz cuadrada de la sumatoria de las diferencias al cuadrado
    # Hay tantas como neuronas RBF    
    dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(centers))

    # Ahora para cada una de las sumatorias anteriores, dividimos las mismas por 2·n-1
    radii = []
    for i in range(0, num_rbf):
        radii.append([sum(dist[i])/(2*(num_rbf-1))])

    return radii

##### Documentacion #####
# numpy.ndarray.shape : https://numpy.org/doc/stable/reference/generated/numpy.ndarray.shape.html
# numpy.hstack : https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
# numpy.ones : https://numpy.org/doc/stable/reference/generated/numpy.ones.html
def calculate_r_matrix(distances, radii):
    """ It obtains the R matrix
        This method obtains the R matrix (as explained in the slides),
        which contains the activation of each RBF for each pattern, including
        a final column with ones, to simulate bias
        
        Parameters
        ----------
        distances: array, shape (n_patterns,num_rbf)
            Matrix with the distance from each pattern to each RBF center
        radii: array, shape (num_rbf,)
            Array with the radius of each RBF
            
        Returns
        -------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
    """

    #TODO: Complete the code of the function
    # En distances, tendremos en cada columna cada una de las neuronas y para cada fila, todas las distancias de los patrones para esa neurona con
    # el centro de la misma
    r_matrix = np.power(distances, 2) # Elevamos al cuadrado las distancias

    for i in range(r_matrix.shape[1]): # El numero de columnas son el numero de neuronas RBF que tenemos
        # Donde [:, i] toma todas las filas de la fila i, osea, todas las salidas para cada neurona 
        r_matrix[:, i] = np.exp(-1*r_matrix[:, i] / (2 * np.power(radii[i], 2))) # Tomamos para cada columna, su radio correspondiente (radii[i])

    # Finalmente añadimos la columna de unos que actua como sesgo
    r_matrix = np.hstack((r_matrix, np.ones((r_matrix.shape[0], 1))))

    return r_matrix

##### Documentacion #####
# numpy.linalg.pinv : https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html
# numpy.matmul : https://numpy.org/doc/stable/reference/generated/numpy.matmul.html 
def invert_matrix_regression(r_matrix, train_outputs):
    """ Inversion of the matrix for regression case
        This method obtains the pseudoinverse of the r matrix and multiplies
        it by the targets to obtain the coefficients in the case of linear
        regression
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
              
        Returns
        -------
        coefficients: array, shape (n_outputs,num_rbf+1)
            For every output, values of the coefficients for each RBF and value 
            of the bias 
    """

    #TODO: Complete the code of the function
    # La obtencion de la pseudo-inversa se puede obtener con una funcion de la libreria Numpy
    penroseMatrix = np.linalg.pinv(r_matrix)
    # Ahora solo queda multiplicar ambas matrices y el resultado seran los coeficientes
    coefficients = np.matmul(penroseMatrix, train_outputs.values)

    return coefficients

##### Documentacion #####
# sklearn.linear_model.LogisticRegression : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
def logreg_classification(matriz_r, train_outputs, l2, eta):
    """ Performs logistic regression training for the classification case
        It trains a logistic regression object to perform classification based
        on the R matrix (activations of the RBFs together with the bias)
        
        Parameters
        ----------
        r_matrix: array, shape (n_patterns,num_rbf+1)
            Matrix with the activation of every RBF for every pattern. Moreover
            we include a last column with ones, which is going to act as bias
        train_outputs: array, shape (n_patterns,n_outputs)
            Matrix with the outputs of the dataset
        l2: bool
            True if we want to use L2 regularization for logistic regression 
            False if we want to use L1 regularization for logistic regression
        eta: float
            Value of the regularization factor for logistic regression
              
        Returns
        -------
        logreg: sklearn.linear_model.LogisticRegression
            Scikit-learn logistic regression model already trained
    """

    #TODO: Complete the code of the function
    # Tenemos el caso en el que aplicamos L2 y el caso en el que aplicamos L1
    # Creamos la clase con los valores correspondientes
    if l2:
        logreg = sklearn.linear_model.LogisticRegression(penalty='l2', C=(1/eta), solver='liblinear')
    else:
        logreg = sklearn.linear_model.LogisticRegression(penalty='l1', C=(1/eta), solver='liblinear')

    # Ajustamos segun los datos de entrenamiento. Indicamos los patrones y las salidas acordes a estos patrones
    logreg.fit(matriz_r, train_outputs.values.ravel()) # El .ravel() es necesario para transformar la columna de salidas en formato de fila

    return logreg

def predict(test_file, model_file):
    """ Performs a prediction with RBFNN model
        It obtains the predictions of a RBFNN model for a test file, using two files, one
        with the test data and one with the model

        Parameters
        ----------
        test_file: string
            Name of the test file
        model_file: string
            Name of the file containing the model data

        Returns
        -------
        test_predictions: array, shape (n_test_patterns,n_outputs)
            Predictions obtained with the model and the test file inputs
    """
    test_df = pd.read_csv(test_file, header=None)
    test_inputs = test_df.values[:, :]
    
    with open(model_file, 'rb') as f:
        saved_data = pickle.load(f)
    
    radii = saved_data['radii']
    classification = saved_data['classification']
    kmeans = saved_data['kmeans']
    
    test_distancias = kmeans.transform(test_inputs)    
    test_r = calculate_r_matrix(test_distancias, radii)    
    
    if classification:
        logreg = saved_data['logreg']
        test_predictions = logreg.predict(test_r)
    else:
        coeficientes = saved_data['coefficients']
        test_predictions = np.dot(test_r, coeficientes)
        
    return test_predictions

if __name__ == "__main__":
    train_rbf_total()
