# TFG - Predicción Ordinal de Altura de Olas con Kedro

Este repositorio contiene el Trabajo de Fin de Grado (TFG) sobre predicción ordinal de la altura de olas mediante modelos de Machine Learning, implementado con el framework **Kedro**.

---

## 📦 Requisitos

- Python 3.9+
- pip
- git

---

## 📂 Estructura de datos

Este proyecto utiliza una estructura dinámica de carpetas según cada ejecución (`run_id` + `execution_folder`):

```
data/
├── 01_raw/                   # Datasets puros. Sin tratamiento.
├── 02_primary/               # Datasets tratados listos para entrenar/testear
├── 03_models/                # Modelos entrenados (PKL)
├── 04_model_output/          # Predicciones (JSON)
├── 05_model_metrics/         # Métricas (JSON)
├── 06_reporting/             # Gráficas generadas por visualization
│   └── <execution_folder>/
│       └── <dataset_id>/
│           ├── heatmap.png
│           ├── scatter_qwk_mae.png
│           ├── nominal/
│           │   ├── accuracy.png
│           │   └── f1_score.png
│           └── ordinal/
│               ├── qwk.png
│               ├── mae.png
│               └── amae.png
```

---

## 🛠️ Instalación

### 1. Clona el repositorio:

```bash
git clone https://github.com/Franjpeca/TFG.git
cd TFG/proyecto-ola
```

### 2. Crea un entorno virtual:

```bash
python3 -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows
```

### 3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

> ⚠️ El archivo ya incluye la instalación de PyTorch CPU desde el repositorio oficial.

---

### 4. Compilar modelos de ORCA-Python (Linux)

> ⚠️ Asegúrate de dar permisos de ejecución al script antes de lanzarlo (chmod +x compile_orca.sh si es necesario).

#### ✅ Opción automática (script)

```bash
bash compile_orca.sh
```

Este script automatiza la compilación de los módulos svorex y libsvmRank de ORCA-Python. Detecta el sistema operativo, instala los compiladores y cabeceras necesarios (como build-essential y python3-dev), actualiza las herramientas de Python (pip, setuptools, wheel) y tras ello ejecuta los comandos de compilación desde las rutas correspondientes. Al finalizar, deja el entorno listo para ejecutar kedro run.

---

##### Alternativa manual si el script falla

> Dependiendo del sistema operativo y paquetes instalados, puede ser necesarios pasos adicionales.

```bash
# Asegúrate de estar en la raíz del proyecto
cd ~/TFG/proyecto-ola

# 1. Compilar svorex
cd orca-python/orca_python/classifiers/svorex
python3 setup.py build_ext --inplace

# 2. Compilar libsvmRank.svm (usado por REDSVM y SVOREX)
cd ../../libsvmRank/python
python3 setup.py build_ext --inplace

# 3. Volver a la raíz del proyecto
cd ~/TFG/proyecto-ola
```

---

### 5. Prueba del programa

```bash
kedro run
```

---

## ▶️ Ejecución general

### Lanzar todos los modelos junto con sus parámetros (preprocessing + training + evaluation)
```bash
kedro run
```

### Lanzar kedro-viz
```bash
kedro viz
```

---

## 🔖 Lanzamientos con etiquetas

> Se pueden combinar con otros comandos

### Lanzar toda una ejecución para un dataset concreto
```bash
kedro run --tags dataset_46014
```

### Lanzar una ejecución para un modelo concreto (training + evaluation)
```bash
kedro run --tags model_LogisticAT
```

### Lanzar entrenamiento + evaluación de un modelo concreto con un grid concreto (training + evaluation)
```bash
kedro run --tags model_DecisionTreeRegressor_grid_001
```

### Lanzar entrenamiento + evaluación de un modelo concreto con un grid concreto y un dataset concreto (sin preprocessing)
```bash
kedro run --tags model_DecisionTreeRegressor_grid_001_dataset_46014
```

---

## 🧹 Pipeline de preprocesamiento

### Lanza únicamente preprocesamiento
```bash
kedro run --pipeline preprocessing
```

### Lanzar preprocessing para un dataset concreto
```bash
kedro run --pipeline preprocessing --tags dataset_46014
```

---

## 🏋️ Pipeline de entrenamiento

### Lanzar únicamente training (todos los modelos)
```bash
kedro run --pipeline training
```

### Lanzar únicamente training de un modelo concreto
```bash
kedro run --pipeline training -t model_LogisticIT
```

---

## 📈 Pipeline de evaluación

### Lanzar únicamente evaluation (toma la última ejecución)
```bash
kedro run --pipeline evaluation
```

### Lanzar evaluation de una ejecución concreta
> Si no se especifica carpeta (`execution_folder`), se toma la última por fecha
```bash
kedro run --pipeline evaluation --params="execution_folder=001_20250816_231254"
```

### Lanzar evaluation de un modelo únicamente
```bash
kedro run --pipeline evaluation -t model_LogisticIT
```

### Lanzar evaluation de todos los modelos de un único dataset
```bash
kedro run --pipeline evaluation --tags dataset_46014
```

### Lanzar evaluation de un modelo concreto con un grid concreto
```bash
kedro run --pipeline evaluation --tags model_DecisionTreeRegressor_grid_001
```

---

## 📊 Pipeline de visualización

### Lanzar únicamente visualization (última ejecución)
```bash
kedro run --pipeline visualization
```

### Lanzar visualization de una ejecución concreta
```bash
kedro run --pipeline visualization --params="execution_folder=001_20250816_231254"
```

### Lanzar visualization tomando varias ejecuciones y juntarlas
```bash
kedro run --pipeline visualization --params="execution_folders=001_20250823_112227;001_20250823_112506;001_20250818_050539"
```

### Lanzar visualization de un dataset únicamente
```bash
kedro run --pipeline visualization --tags dataset_46014
```

### Lanzar visualization de una ejecución concreta y un dataset concreto
```bash
kedro run --pipeline visualization --params="execution_folder=001_20250816_231254" --tags dataset_46014
```

### Lanzar visualization de una gráfica de un tipo concreto para todos los datasets
```bash
kedro run --pipeline visualization --tags node_visualization_ordinal
```

### Lanzar visualization de una gráfica de un tipo concreto para un dataset concreto
```bash
kedro run --pipeline visualization --tags ordinal_dataset_46014
kedro run --pipeline visualization --tags heatmap_dataset_46014
```

### Lanzar visualization de una métrica en concreto
```bash
kedro run --pipeline visualization --nodes VIS_ORDINAL_QWK_46014
```

---

## ⚙️ Otros flags útiles

### --from-inputs
Permite ejecutar únicamente los nodos que consumen un input concreto (y sus descendientes).  
Es muy útil si has cambiado un dataset intermedio y quieres re-lanzar solo lo que depende de él.

```bash
kedro run --from-inputs cleaned_46014_train_ordinal
```

### --nodes
Permite lanzar directamente un nodo específico del pipeline, sin ejecutar el resto de nodos dependientes.  
Es útil si quieres probar un nodo aislado o re-ejecutarlo sin pasar por todo el pipeline.

```bash
kedro run --nodes PREPROCESSING_clean_pair_46042
```

### --only-nodes
Ejecuta únicamente los nodos especificados, sin ejecutar ni los anteriores ni los posteriores.  
Es útil cuando quieres forzar la ejecución de un nodo concreto sin que se arrastre el grafo de dependencias.  
En kedro-viz se pueden ver ejemplos que usan este elemento.

---

## 🧠 Modelos incluidos

- **MORD**: LAD, LogisticAT, LogisticIT, OrdinalRidge
- **ORCA (dlordinal)**: NNPOM, NNOP, REDSVM, SVOREX, OrdinalDecomposition
- **Nominales**: DecisionTreeRegressor, KNeighborsClassifier, LinearRegression

---

## 📌 Detalles importantes

- La identidad de ejecución se controla con `run_id` y `execution_folder`.
- Todos los resultados se guardan en rutas únicas para facilitar comparaciones y evitar sobrescritura.
- El hook dinámico asegura que los modelos, métricas y gráficas se registran y guardan correctamente incluso si no están en `catalog.yml`.


## 🔗 Repositorios externos

- [ORCA-Python (Ayrna)](https://github.com/ayrna/orca-python): librería utilizada para implementar modelos ordinales implementados por el grupo de investiación AYRNA de la Universidad de Córdoba.
- [MORD (Fabian Pedregosa)](https://github.com/fabianp/mord): colección de algoritmos de regresión ordinal en Python con API compatible con scikit-learn.