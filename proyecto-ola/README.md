# TFG - Predicción Ordinal de Altura de Olas con Kedro

Este repositorio contiene el Trabajo de Fin de Grado (TFG) sobre predicción ordinal de la altura de olas mediante modelos de Machine Learning, implementado con el framework **Kedro**.

---

## 📦 Requisitos

- Python 3.9+
- pip

---

## 📂 Estructura de datos

Este proyecto utiliza una estructura dinámica de carpetas según cada ejecución (`run_id` + `execution_folder`):

```
data/
├── 04_models/                # Modelos entrenados (PKL)
├── 05_model_output/          # Predicciones (JSON)
├── 06_model_metrics/         # Métricas (JSON)
├── 07_reporting/             # Gráficas generadas por visualization
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

1. Clona el repositorio:

```bash
git clone https://github.com/Franjpeca/TFG.git
cd TFG/proyecto-ola
```

2. Crea un entorno virtual:

```bash
python -m venv venv
source venv/bin/activate  # o .\venv\Scripts\activate en Windows
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

> ⚠️ El archivo ya incluye la instalación de PyTorch CPU desde el repositorio oficial.

---

## ▶️ Ejecución general

### Lanzar todos los modelos junto con sus parámetros (preprocessing + training + evaluation)
```bash
kedro run
```

### Lanzar toda una ejecución para un dataset concreto
```bash
kedro run --tags dataset_46014
```

### Lanzar toda una ejecución para un modelo concreto (training + evaluation)
```bash
kedro run --tags model_LogisticAT
```

### Lanzar entrenamiento + evaluación de un modelo concreto y un dataset concreto (preprocessing)
```bash
kedro run --tags model_DecisionTreeRegressor --tags dataset_46014
```

### Lanzar entrenamiento + evaluación de un modelo concreto con un grid concreto (training + evaluation)
```bash
kedro run --tags model_DecisionTreeRegressor_grid_001
```

### Lanzar entrenamiento + evaluación de un modelo concreto con un grid concreto y un dataset concreto (sin preprocessing)
```bash
kedro run --tags model_DecisionTreeRegressor_grid_001_dataset_46014
```

### Lanzar entrenamiento + evaluación de un modelo concreto con un grid concreto y un dataset concreto (con preprocessing)
```bash
kedro run --tags model_DecisionTreeRegressor_grid_001 --tags dataset_46014
```

### Lanzar entrenamiento + evaluación de grids concretos
```bash
kedro run --tags grid_001
```

### Lanzar entrenamiento + evaluación de un grid concreto de un dataset (con preprocessing)
```bash
kedro run --tags grid_002 --tags dataset_46042
```

---

## 🔄 Ejecución de subpipelines

### Lanza únicamente preprocesamiento
```bash
kedro run --pipeline preprocessing
```

### Lanzar preprocessing para un dataset concreto
```bash
kedro run --pipeline preprocessing --tags dataset_46014
```

### Lanza únicamente training (todos los modelos)
```bash
kedro run --pipeline training
```

### Lanza únicamente training de un modelo concreto
```bash
kedro run --pipeline training -t model_LogisticIT
```

### Lanza únicamente evaluation (toma última ejecución)
```bash
kedro run --pipeline evaluation
```

### Lanzar evaluation de una ejecución concreta
```bash
kedro run --pipeline evaluation --params="execution_folder=001_20250816_231254"
```

### Lanzar evaluation de un modelo unicamente
```bash
kedro run --pipeline evaluation -t model_LogisticIT
```

### Lanzar evaluation de todos los modelos de un dataset
```bash
kedro run --pipeline evaluation --tags dataset_46014
```

### Lanzar evaluation de un modelo concreto con un grid concreto
```bash
kedro run --pipeline evaluation --tags model_DecisionTreeRegressor_grid_001
```

> Si no se especifica carpeta (`execution_folder`) se toma la última por fecha.

---

## 📊 Visualización de resultados

⚠️ El pipeline de visualización **no forma parte del `__default__`** para evitar errores cuando no hay métricas generadas.

### Lanzar visualization (última ejecución)
```bash
kedro run --pipeline visualization
```

### Lanzar visualization de una ejecución concreta
```bash
kedro run --pipeline visualization --params="execution_folder=001_20250816_231254"
```

### Lanzar visualization de un dataset concreto
```bash
kedro run --pipeline visualization --tags dataset_46014
```

### Lanzar visualization de una ejecución concreta y un dataset concreto
```bash
kedro run --pipeline visualization --params="execution_folder=001_20250816_231254" --tags dataset_46014
```

### Lanzar visualization de una gráfica concreta para todos los datasets
```bash
kedro run --pipeline visualization --tags node_visualization_ordinal
```

### Lanzar visualization de una gráfica concreta para un dataset concreto
```bash
kedro run --pipeline visualization --tags ordinal_dataset_46014
kedro run --pipeline visualization --tags heatmap_dataset_46014
```

### Lanzar visualization de una métrica en concreto
```bash
kedro run --pipeline visualization --nodes VIS_ORDINAL_QWK_46014
```

---

## 🧠 Modelos incluidos

- **MORD**: LAD, LogisticAT, LogisticIT, OrdinalRidge
- **ORCA (dlordinal)**: NNPOM, NNOP, REDSVM, SVOREX, OrdinalDecomposition

---

## 📌 Detalles importantes

- La identidad de ejecución se controla con `run_id` y `execution_folder`.
- Todos los resultados se guardan en rutas únicas para facilitar comparaciones y evitar sobrescritura.
- El hook dinámico asegura que los modelos, métricas y gráficas se registran y guardan correctamente incluso si no están en `catalog.yml`.
