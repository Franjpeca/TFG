import logging
from pathlib import Path
from functools import update_wrapper
from typing import Optional, List

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import Visualize_Nominal_Metric, Visualize_Ordinal_Metric
from proyecto_ola.utils.wrappers import make_nominal_viz_wrapper, make_ordinal_viz_wrapper

logger = logging.getLogger(__name__)


def get_execution_folder(run_id: Optional[str] = None) -> Optional[str]:
    base_dir = Path("data/08_model_metrics")
    if not base_dir.exists():
        return None

    pattern = f"{run_id}_*" if run_id else "*_*"
    candidates = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    if not candidates:
        return None

    return candidates[-1].name


def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    execution_folder = get_execution_folder(params.get("run_id"))

    if not execution_folder:
        logger.warning("[VISUALIZATION] No se encontró ninguna carpeta de métricas en 08_model_metrics. Pipeline vacío.")
        return Pipeline([])

    nominal_metrics = params.get("nominal_metrics", ["accuracy", "f1_score"])
    ordinal_metrics = params.get("ordinal_metrics", ["qwk", "mae", "amae"])

    model_params = params.get("model_parameters", {})
    train_datasets = params.get("training_datasets", [])
    default_cv = params.get("cv_settings", {"n_splits": 5, "random_state": 42})
    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    subpipelines = []
    datasets_map = {}

    for model_name, combos in model_params.items():
        for combo_id, cfg in combos.items():
            hyper_str = "gridsearch" if "param_grid" in cfg else "hyperparams"
            cv_cfg = cfg.get("cv_settings", default_cv)
            cv_str = f"cv_{cv_cfg['n_splits']}_rs_{cv_cfg['random_state']}"

            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"

                if evaluated_keys and full_key not in evaluated_keys:
                    continue

                json_key = f"evaluation.{execution_folder}.Metrics_{full_key}"
                datasets_map.setdefault(dataset_id, []).append(json_key)

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
                        name=f"VIS_NOMINAL_{metric_name.upper()}_{dataset_id}"
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
                        name=f"VIS_ORDINAL_{metric_name.upper()}_{dataset_id}"
                    )
                ])
            )

    if not subpipelines:
        logger.info("No se generó ningún subpipeline: no se hallaron métricas evaluadas.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))
