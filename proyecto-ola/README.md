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

Para lanzar todos los pipelines (preprocesado, entrenamiento, evaluación):

```bash
kedro run
```
Esto generará todos los resultados de los modelos encontrados en conf/base/parameters.yml.

Para lanzar fases concretas:

```bash
kedro run -p training
kedro run -p evaluation
```

---

## 📊 Visualización de resultados

⚠️ El pipeline de visualización **no forma parte del `__default__`** para evitar errores cuando no hay métricas generadas. Está diseñado para ejecutarse manualmente y detectar automáticamente la última ejecución con resultados.

### 🔁 Ejecución completa del pipeline de visualización:

```bash
kedro run -p visualization
```

Este comando generará todas las gráficas posibles para los modelos evaluados en la última ejecución válida (por orden de modificación en disco).

---

### 🎯 Visualizar una métrica concreta (opcional):

```bash
kedro run -p visualization \
  --params execution_folder=001_20250815_184843   --to-outputs=visualization.001_20250815_184843.46053.qwk
```

> 🧠 **Nota importante:** Los comandos generados por **Kedro Viz** no añaden automáticamente el argumento `--pipeline=visualization`.  
> Por eso, si se desea lanzar visualización desde la CLI con `--to-outputs`, hay que especificar explícitamente el pipeline con `-p visualization`.

---

## 🧠 Modelos incluidos

- **MORD**: LAD, LogisticAT, LogisticIT, OrdinalRidge
- **ORCA (dlordinal)**: NNPOM, NNOP, REDSVM, SVOREX, OrdinalDecomposition

---

## 📌 Detalles importantes

- La identidad de ejecución se controla con `run_id` y `execution_folder`.
- Todos los resultados se guardan en rutas únicas para facilitar comparaciones y evitar sobrescritura.
- El hook dinámico asegura que los modelos, métricas y gráficas se registran y guardan correctamente incluso si no están en `catalog.yml`.