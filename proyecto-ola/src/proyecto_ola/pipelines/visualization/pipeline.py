import logging
from functools import update_wrapper
from typing import Optional, List
from pathlib import Path
from kedro.pipeline import Pipeline, node, pipeline

from proyecto_ola.utils.pipelines_utils import find_parameters_cli, find_latest_metrics_execution_folder

from .nodes import (
    Visualize_Nominal_Metric,
    Visualize_Ordinal_Metric,
    Visualize_Heatmap_Metrics,
    Visualize_Scatter_QWKvsMAE,
)
from proyecto_ola.utils.wrappers import (
    make_nominal_viz_wrapper,
    make_ordinal_viz_wrapper,
    make_heatmap_viz_wrapper,
    make_scatter_qwk_mae_viz_wrapper,
)

logger = logging.getLogger(__name__)

def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})

    # Leer las metricas a visualizar
    nominal_metrics = params.get("nominal_metrics", ["accuracy", "f1_score"])
    ordinal_metrics = params.get("ordinal_metrics", ["qwk", "mae", "amae"])
    heatmap_metrics = params.get("heatmap_metrics", ["qwk", "mae", "amae", "f1_score", "accuracy"])

    # Buscar la carpeta de ejecucion
    execution_folder = find_parameters_cli("execution_folder", params)
    if not execution_folder:
        execution_folder = find_latest_metrics_execution_folder(Path("data/06_model_metrics"))

    logger.info(f"[VISUALIZATION] Usando carpeta de ejecucion: {execution_folder}")

    # Salimos si no hay carpeta valida
    if not execution_folder:
        logger.warning("[VISUALIZATION] No se encontro ninguna carpeta de metricas en 06_model_metrics. Pipeline vacio.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    # Buscar ficheros de metricas en la carpeta seleccionada
    metrics_dir = Path("data/06_model_metrics") / execution_folder
    metric_files = sorted(metrics_dir.glob("Metrics_*.json"))

    # Salimos si no hay ficheros de metricas
    if not metric_files:
        logger.info(f"[VISUALIZATION] La carpeta {metrics_dir} no contiene metricas. Pipeline vacio.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    # Revisar si hay evaluated_keys pasados por params
    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    # Asociar cada dataset a las metricas encontradas
    datasets_map = {}
    for f in metric_files:
        full_key = f.stem.replace("Metrics_", "", 1)
        if evaluated_keys and full_key not in evaluated_keys:
            continue
        tokens = full_key.split("_")
        if len(tokens) < 6:
            logger.warning(f"[VISUALIZATION] full_key inesperada: {full_key}")
            continue
        dataset_id = tokens[-6]
        json_key = f"evaluation.{execution_folder}.{f.stem}"
        datasets_map.setdefault(dataset_id, []).append(json_key)

    # Salimos si tras filtrar no hay datasets validos
    if not datasets_map:
        logger.info(f"[VISUALIZATION] No hay metricas validas tras filtrar en {metrics_dir}. Pipeline vacio.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    # Construir todos los nodos por dataset y tipo de metrica
    subpipelines: List[Pipeline] = []

    for dataset_id, metric_inputs in datasets_map.items():
        # Nodos de metricas nominales
        for metric_name in nominal_metrics:
            wrapped = make_nominal_viz_wrapper(
                viz_func=Visualize_Nominal_Metric,
                metric=metric_name,
                dataset_id=dataset_id,
                execution_folder=execution_folder,
            )
            wrapped = update_wrapper(wrapped, Visualize_Nominal_Metric)
            subpipelines.append(
                pipeline([
                    node(
                        func=wrapped,
                        inputs=metric_inputs,
                        outputs=f"visualization.{execution_folder}.{dataset_id}.{metric_name}",
                        name=f"VIS_NOMINAL_{metric_name.upper()}_{dataset_id}",
                        tags=[
                            "pipeline_visualization",
                            f"dataset_{dataset_id}",
                            f"execution_{execution_folder}",
                            f"nominal_dataset_{dataset_id}",
                            "node_visualization",
                            "node_visualization_nominal",
                        ],
                    )
                ])
            )
        # Nodos de metricas ordinales
        for metric_name in ordinal_metrics:
            wrapped = make_ordinal_viz_wrapper(
                viz_func=Visualize_Ordinal_Metric,
                metric=metric_name,
                dataset_id=dataset_id,
                execution_folder=execution_folder,
            )
            wrapped = update_wrapper(wrapped, Visualize_Ordinal_Metric)
            subpipelines.append(
                pipeline([
                    node(
                        func=wrapped,
                        inputs=metric_inputs,
                        outputs=f"visualization.{execution_folder}.{dataset_id}.{metric_name}",
                        name=f"VIS_ORDINAL_{metric_name.upper()}_{dataset_id}",
                        tags=[
                            "pipeline_visualization",
                            f"dataset_{dataset_id}",
                            f"execution_{execution_folder}",
                            f"ordinal_dataset_{dataset_id}",
                            "node_visualization",
                            "node_visualization_ordinal",
                        ],
                    )
                ])
            )
        # Nodo heatmap por dataset
        wrapped_heatmap = make_heatmap_viz_wrapper(
            viz_func=Visualize_Heatmap_Metrics,
            metrics=heatmap_metrics,
            dataset_id=dataset_id,
            execution_folder=execution_folder,
        )
        wrapped_heatmap = update_wrapper(wrapped_heatmap, Visualize_Heatmap_Metrics)
        subpipelines.append(
            pipeline([
                node(
                    func=wrapped_heatmap,
                    inputs=metric_inputs,
                    outputs=f"visualization.{execution_folder}.{dataset_id}.heatmap",
                    name=f"VIS_HEATMAP_{dataset_id}",
                    tags=[
                        "pipeline_visualization",
                        f"dataset_{dataset_id}",
                        f"execution_{execution_folder}",
                        f"heatmap_dataset_{dataset_id}",
                        "node_visualization",
                        "node_visualization_heatmap",
                    ],
                )
            ])
        )
        # Nodo scatter QWK vs MAE por dataset
        wrapped_scatter = make_scatter_qwk_mae_viz_wrapper(
            viz_func=Visualize_Scatter_QWKvsMAE,
            dataset_id=dataset_id,
            execution_folder=execution_folder,
        )
        wrapped_scatter = update_wrapper(wrapped_scatter, Visualize_Scatter_QWKvsMAE)
        subpipelines.append(
            pipeline([
                node(
                    func=wrapped_scatter,
                    inputs=metric_inputs,
                    outputs=f"visualization.{execution_folder}.{dataset_id}.scatter_qwk_mae",
                    name=f"VIS_SCATTER_QWK_MAE_{dataset_id}",
                    tags=[
                        "pipeline_visualization",
                        f"dataset_{dataset_id}",
                        f"execution_{execution_folder}",
                        f"plot_dataset_{dataset_id}",
                        "node_visualization",
                        "node_visualization_scatter",
                    ],
                )
            ])
        )

    # Devolver todos los subpipelines juntos
    return sum(subpipelines, Pipeline([]))