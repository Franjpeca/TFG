import numpy as np
import pandas as pd


### ======= Repaso de Pandas ======= ###

### Series ###
# Creacion de una serie
a = [1, 7, 2]
var = pd.Series(a)
# Asignación de etiquetas una serie
var.index = ["a", "b", "c"]



### DataFrame ###
# Creación de un diccionario
datos = {
  "calorias": [420, 380, 390],
  "duracion": [50, 40, 45]
}
# Lo cargamos en un dataframe
df = pd.DataFrame(datos)



### Acceder a filas y columnas ###
print(df.loc[0]) # Muestra el patron de esa linea

print(df['calorias']) # Muestro esa característica con todos los elementos

print(df[10, 'calorias']) # Muestro el valor de calorias del patron 10.
# Es acceder a una celda



### Cargar un CSV ###
df = pd.read_csv('data.csv')



### Mostrar un DataFrame ###
print(df) # No lo muestra al completo
print(df.to_string()) # Muestra todo al completo



### Operaciones para visualizar datos ###
print(df.index()) # Numero de filas que tiene el dataframe

print(df.head(5)) # Muestra las 5 primeras filas
print(df.tail(5)) # Muestra las 5 ultimas filas

print(df.info()) # Muestra un resumen de información:
# Numero de filas y columnas
# Nombre de cada columna
# Elementos no nulos de cada una de las columnas
# Tipos de datos que guardan esa columna



### Limpieza de datos ###

# Borrado de datos vacios
aux = df.dropna()
# Rellenado de datos vacios con un valor en concreto
aux = df.fillna(10)
# Lo mismo pero para una columna en concreto
aux = df["calorias"].fillna(10)

# Transformar a tipo de fecha
df['fecha'] = pd.to_datetime(df['fecha']) # Es una funcion de pandas, no del dataframe

# Cambiar valores dados uno limite
for x in df.index: # Aqui estoy accediendo a una celda en concreto
  if df.loc[x, "Duration"] > 120: # Siempre que sea mayor que el threshold...
    df.loc[x, "Duration"] = 120   # se reemplazara por este

# Duplicados
print(df.duplicated()) # Devuelve true en caso de que haya duplicados

df.drop_duplicates(inplace = True) # Borra duplicados



### Uso de elementos estadisticos basicos ###
media = df["calorias"].mean()
mediana = df["calorias"].median()
moda = df["calorias"].mode()

# Matriz de correlacion para las características
df.corr()
# Cercano a 1: Relacionadas, si sube una, sube la otra
# Cercano a -1: Relacionadas pero, si sube una, baja la otra
# Cercano a 0: Poco relaciondas




### ======= Repaso de Numpy ======= ###

# La libreria numpy esta implementada de tal forma que permite un manejo mas rapido de la información, a diferencia de 
# las estructuras base de python. Por lo que son mas optimas para calculos complejos

# Parte de esto se logra por como numpy almacena los datos en memoria. A diferencia de python, almacena los arrays de forma secuencia en memoria

# Las listas en python se pueden modificar, las tuplas no.


### Array de Numpy ###
# Podemos mandar una lista, una tupla...
v = np.array((1,2,3,4)) # Mandamos una tupla
v = np.arra([1,2,3,4]) # Mandamos una lista

# Dimensiones
v = np.array(5) # Dimension 1
v = np.array([1,2,3]) # Dimension 2
v = np.array([[1,2,3],["a","b",4,5]]) # Dimension 2
# etc

# Se va guardadon por filas cada array en una fila

# Ver la dimension
print(v.ndim) 

# Acceder a elementos del array
print(v[0]) # Para 1 dimension
print(v[0,1]) # Para 2 dimensiones
# etc

# Podemos "tomar rodajas" de los arrays
print(v[1:5]) # Toma los elementos del 1 al 4
print(v[4:]) # Toma los elementos del 4 al final
print(v[1:4:2]) # Toma los elementos del 1 al 4 pero de 2 en 2, toma el 1, luego el 3


# View y copia
v2 = v.view() # Todo lo que se modifique en v2 se modifica en v
v3 = v.copy() # Aunque se modifique v3, no se modifica v


# Shape
print(v.shape()) # Indica el numero de elementos en cada dimension


# Iterar
for x in v:
  print(x) # Se itera como en python normal
# Tambien con lo anterior se itera en uno de mas dimensiones
# Este es para iterar en cada uno de los elementos, primero iteramos en la primera 
# dimension de arrays y luego en el siguiente (elemento) 
for x in v:
  for y in x:
    print(y)



### Join ###

# Concatenar
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
resultado = np.concatenate((arr1, arr2))
print(resultado) 
# [1 2 3 4 5 6]

# Stackear
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
resultado = np.vstack((arr1, arr2))
print(resultado)
#[[1 2 3]
# [4 5 6]]

# Stackear por columnas
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([[4], [5], [6]])
resultado = np.hstack((arr1, arr2))
print(resultado)
#[[1 4]
# [2 5]
# [3 6]]


### Encontrar valores ###

# Where
# Busca el INDICE del array (posicion) donde estan los elementos que coinciden en valor con el pasado (el 4)
arr = np.array([1, 2, 3, 4, 5, 4, 4])
resultado = np.where(arr == 4)
print(resultado)
# (array([3, 5, 6]),) 



### Ordenar ###

# Ordena sin modificar el original
arr = np.array(['banana', 'cherry', 'apple'])
print(np.sort(arr))


### Filtrado ###

# Toma los valores que tienen el filtro como true
arr = np.array([41, 42, 43, 44])
filtro = [True, False, True, False]
nuevo_arr = arr[filtro]
print(nuevo_arr)
# 41 43

# Lo mismo pero el filtro se crea en base a una condicion
arr = np.array([1, 2, 3, 4, 5, 6, 7])
filtro = arr > 4
nuevo_arr = arr[filtro]
print(nuevo_arr)
# 5 6 7



### ======= Repaso de Matplotlib ======= ###

import matplotlib.pyplot as plt

# Defino los puntos, para ello pongo las corodenadas X e Y
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

# Esto une los puntos con lineas
plt.plot(xpoints, ypoints)

# Esto muestra directamente los puntos
plt.plot(xpoints, ypoints, 'o')

# Con esto ponemos nombre a los ejes
plt.xlabel("Eje X")
plt.ylabel("Eje y")
# Titulo
plt.title("Titulo del grafico")

# Si queremos mostrar dos graficos debemos de volver a usar plt.plot
# Los graficos estaran uno al lado del otro

plt.show()


# Podemos mostrar un grafico de dispersion
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
# Creamos el grafico de puntos
plt.scatter(x, y)
plt.show()
# Si volvemos a llamar a la funcion, entonces se dibujan uno encima del otro con diferentes colores 

# Esto mostrara el grafico de dispersion con el tamaño del de los puntos
plt.scatter(x, y, s=sizes)


# Para graficos de barras
plt.bar(x,y)

# Genera numeros de forma aleatorias segun la distribucion normal
x = np.random.normal(170, 10, 250)

# Genera el histograma correspondiente
plt.hist(x)
plt.show() 

# Genera un grafico pastel
plt.pie()