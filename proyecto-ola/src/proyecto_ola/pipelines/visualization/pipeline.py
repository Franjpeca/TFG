import logging
from functools import update_wrapper
from typing import List, Dict
from pathlib import Path
from kedro.pipeline import Pipeline, node, pipeline

from proyecto_ola.utils.pipelines_utils import (
    find_parameters_cli,
    find_latest_metrics_execution_folder,
    parse_folders_param,
)

from .nodes import (
    Visualize_Nominal_Metric,
    Visualize_Ordinal_Metric,
    Visualize_Heatmap_Metrics,
    Visualize_Scatter_QWKvsAMAE,
)
from proyecto_ola.utils.wrappers import (
    make_nominal_viz_wrapper,
    make_ordinal_viz_wrapper,
    make_heatmap_viz_wrapper,
    make_scatter_qwk_amae_viz_wrapper,
)

logger = logging.getLogger(__name__)
BASE_DIR = Path("data/05_model_metrics")


def _choose_output_folder(exec_folders: List[str]) -> str:
    try:
        return max(exec_folders, key=lambda f: (BASE_DIR / f).stat().st_mtime)
    except Exception:
        return exec_folders[0]


def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})

    nominal_metrics = params.get("nominal_metrics", ["accuracy", "f1_score"])
    ordinal_metrics = params.get("ordinal_metrics", ["qwk", "mae", "amae"])
    heatmap_metrics = params.get("heatmap_metrics", ["qwk", "mae", "amae", "f1_score", "accuracy"])

    cli_multi = find_parameters_cli("execution_folders", params)
    exec_folders = parse_folders_param(cli_multi)

    if not exec_folders:
        cli_single = find_parameters_cli("execution_folder", params) or params.get("execution_folder")
        exec_folders = parse_folders_param(cli_single) if cli_single else []

    if not exec_folders:
        latest = find_latest_metrics_execution_folder(BASE_DIR)
        if latest:
            exec_folders = [latest]

    if not exec_folders:
        logger.warning("[VISUALIZATION] No se encontró ninguna carpeta de métricas en 05_model_metrics. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    output_execution_folder = _choose_output_folder(exec_folders)

    logger.info(f"[INFO_VISUALIZATION] Carpetas de ENTRADA: {exec_folders}")
    logger.info(f"[INFO_VISUALIZATION] Carpeta de SALIDA: {output_execution_folder}\n")

    all_metric_files = []
    for folder in exec_folders:
        metrics_dir = BASE_DIR / folder
        files = sorted(metrics_dir.glob("Metrics_*.json"))
        if not files:
            logger.info(f"[VISUALIZATION] La carpeta {metrics_dir} no contiene métricas. Se omite.")
        all_metric_files.extend(files)

    if not all_metric_files:
        logger.info("[VISUALIZATION] Ninguna carpeta contiene métricas válidas. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    datasets_map: Dict[str, List[str]] = {}
    for f in all_metric_files:
        full_key = f.stem.replace("Metrics_", "", 1)
        if evaluated_keys and full_key not in evaluated_keys:
            continue
        tokens = full_key.split("_")
        if len(tokens) < 6:
            logger.warning(f"[VISUALIZATION] full_key inesperada: {full_key}")
            continue
        dataset_id = tokens[-8] if "seed" in tokens else tokens[-6]
        src_folder = f.parent.name
        json_key = f"evaluation.{src_folder}.{f.stem}"
        datasets_map.setdefault(dataset_id, []).append(json_key)

    if not datasets_map:
        logger.info("[VISUALIZATION] No hay métricas válidas tras filtrar. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    subpipelines: List[Pipeline] = []

    for dataset_id, metric_inputs in datasets_map.items():
        for metric_name in nominal_metrics:
            wrapped = make_nominal_viz_wrapper(
                viz_func=Visualize_Nominal_Metric,
                metric=metric_name,
                dataset_id=dataset_id,
                execution_folder=output_execution_folder,
            )
            wrapped = update_wrapper(wrapped, Visualize_Nominal_Metric)
            subpipelines.append(
                pipeline([
                    node(
                        func=wrapped,
                        inputs=metric_inputs,
                        outputs=f"visualization.{output_execution_folder}.{dataset_id}.{metric_name}",
                        name=f"VIS_NOMINAL_{metric_name.upper()}_{dataset_id}",
                        tags=[
                            "pipeline_visualization",
                            f"dataset_{dataset_id}",
                            f"execution_{output_execution_folder}",
                            f"nominal_dataset_{dataset_id}",
                            "node_visualization",
                            "node_visualization_nominal",
                        ],
                    )
                ])
            )

        for metric_name in ordinal_metrics:
            wrapped = make_ordinal_viz_wrapper(
                viz_func=Visualize_Ordinal_Metric,
                metric=metric_name,
                dataset_id=dataset_id,
                execution_folder=output_execution_folder,
            )
            wrapped = update_wrapper(wrapped, Visualize_Ordinal_Metric)
            subpipelines.append(
                pipeline([
                    node(
                        func=wrapped,
                        inputs=metric_inputs,
                        outputs=f"visualization.{output_execution_folder}.{dataset_id}.{metric_name}",
                        name=f"VIS_ORDINAL_{metric_name.upper()}_{dataset_id}",
                        tags=[
                            "pipeline_visualization",
                            f"dataset_{dataset_id}",
                            f"execution_{output_execution_folder}",
                            f"ordinal_dataset_{dataset_id}",
                            "node_visualization",
                            "node_visualization_ordinal",
                        ],
                    )
                ])
            )

        wrapped_heatmap = make_heatmap_viz_wrapper(
            viz_func=Visualize_Heatmap_Metrics,
            metrics=heatmap_metrics,
            dataset_id=dataset_id,
            execution_folder=output_execution_folder,
        )
        wrapped_heatmap = update_wrapper(wrapped_heatmap, Visualize_Heatmap_Metrics)
        subpipelines.append(
            pipeline([
                node(
                    func=wrapped_heatmap,
                    inputs=metric_inputs,
                    outputs=f"visualization.{output_execution_folder}.{dataset_id}.heatmap",
                    name=f"VIS_HEATMAP_{dataset_id}",
                    tags=[
                        "pipeline_visualization",
                        f"dataset_{dataset_id}",
                        f"execution_{output_execution_folder}",
                        f"heatmap_dataset_{dataset_id}",
                        "node_visualization",
                        "node_visualization_heatmap",
                    ],
                )
            ])
        )

        wrapped_scatter = make_scatter_qwk_amae_viz_wrapper(
            viz_func=Visualize_Scatter_QWKvsAMAE,
            dataset_id=dataset_id,
            execution_folder=output_execution_folder,
        )
        wrapped_scatter = update_wrapper(wrapped_scatter, Visualize_Scatter_QWKvsAMAE)
        subpipelines.append(
            pipeline([
                node(
                    func=wrapped_scatter,
                    inputs=metric_inputs,
                    outputs=f"visualization.{output_execution_folder}.{dataset_id}.scatter_qwk_amae",
                    name=f"VIS_SCATTER_QWK_AMAE_{dataset_id}",
                    tags=[
                        "pipeline_visualization",
                        f"dataset_{dataset_id}",
                        f"execution_{output_execution_folder}",
                        f"plot_dataset_{dataset_id}",
                        "node_visualization",
                        "node_visualization_scatter",
                    ],
                )
            ])
        )

    return sum(subpipelines, Pipeline([]))
