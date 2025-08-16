import logging
from pathlib import Path
from functools import update_wrapper
from typing import Optional, List
import sys
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import Visualize_Nominal_Metric, Visualize_Ordinal_Metric, Visualize_Heatmap_Metrics, Visualize_Scatter_QWKvsMAE
from proyecto_ola.utils.wrappers import make_nominal_viz_wrapper, make_ordinal_viz_wrapper, make_heatmap_viz_wrapper, make_scatter_qwk_mae_viz_wrapper

logger = logging.getLogger(__name__)


def get_execution_folder(run_id: Optional[str] = None) -> Optional[str]:
    base_dir = Path("data/06_model_metrics")
    if not base_dir.exists():
        return None

    pattern = f"{run_id}_*" if run_id else "*_*"

    # Última carpeta por mtime del DIRECTORIO, pero solo si contiene Metrics_*.json
    best_dir, best_ts = None, -1.0
    for d in base_dir.glob(pattern):
        if not any(d.glob("Metrics_*.json")):
            continue
        ts = d.stat().st_mtime
        if ts > best_ts:
            best_ts, best_dir = ts, d

    return best_dir.name if best_dir else None


def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    nominal_metrics = params.get("nominal_metrics", ["accuracy", "f1_score"])
    ordinal_metrics = params.get("ordinal_metrics", ["qwk", "mae", "amae"])
    # lista de métricas para el heatmap (puedes sobreescribir por params)
    heatmap_metrics = params.get("heatmap_metrics", ["qwk", "mae", "amae", "f1_score", "accuracy"])

    # --- Leer execution_folder: 1) params; 2) CLI (--params ...); 3) autodetección
    execution_folder = params.get("execution_folder")

    if not execution_folder:
        # Soporta --params="execution_folder=...,otra=..." y --params execution_folder=...,otra=...
        for arg in sys.argv:
            if arg.startswith("--params="):
                raw = arg.split("=", 1)[1].strip().strip('"').strip("'")
                for kv in raw.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        if k.strip() == "execution_folder":
                            execution_folder = v.strip()
                            break
            if execution_folder:
                break
        if not execution_folder and "--params" in sys.argv:
            try:
                i = sys.argv.index("--params")
                if i + 1 < len(sys.argv):
                    raw = sys.argv[i + 1]
                    for kv in raw.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            if k.strip() == "execution_folder":
                                execution_folder = v.strip()
                                break
            except Exception:
                pass

    if not execution_folder:
        execution_folder = get_execution_folder(params.get("run_id"))

    # No-op si no hay carpeta de métricas válida
    if not execution_folder:
        logger.warning("[VISUALIZATION] No se encontró ninguna carpeta de métricas en 06_model_metrics. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    # Si el hook inyecta evaluated_keys, filtramos por ellas; si no, usamos lo que haya en disco
    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    subpipelines: List[Pipeline] = []
    datasets_map = {}

    # Descubrir métricas realmente guardadas en disco en la carpeta elegida
    metrics_dir = Path("data/06_model_metrics") / execution_folder
    metric_files = sorted(metrics_dir.glob("Metrics_*.json"))

    if not metric_files:
        logger.info(f"[VISUALIZATION] La carpeta {metrics_dir} no contiene métricas. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    for f in metric_files:
        full_key = f.stem.replace("Metrics_", "", 1)

        # Si vienen evaluated_keys, respetarlas
        if evaluated_keys and full_key not in evaluated_keys:
            continue

        # Estructura: <model_name>_<combo_id>_<dataset_id>_<hyper_str>_<cv_str>
        tokens = full_key.split("_")
        if len(tokens) < 6:
            logger.warning(f"[VISUALIZATION] full_key inesperada: {full_key}")
            continue
        dataset_id = tokens[-6]

        # Debe coincidir con lo que registra el hook:
        # evaluation.{execution_folder}.Metrics_<full_key>
        json_key = f"evaluation.{execution_folder}.{f.stem}"
        datasets_map.setdefault(dataset_id, []).append(json_key)

    if not datasets_map:
        logger.info(f"[VISUALIZATION] No hay métricas válidas tras filtrar en {metrics_dir}. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="VISUALIZATION_NOOP", tags=["pipeline_visualization"])
        ])

    # Construcción de nodos por dataset y métrica
    for dataset_id, metric_inputs in datasets_map.items():
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
                    )
                ])
            )

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
                    )
                ])
            )

        # HEATMAP (una por dataset)
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
                )
            ])
        )

        # SCATTER QWK vs MAE (una por dataset)
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
                )                                                                       
            ])                                                                          
        )                                                                               

    return sum(subpipelines, Pipeline([]))