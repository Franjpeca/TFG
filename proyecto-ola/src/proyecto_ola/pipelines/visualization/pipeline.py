import logging
from pathlib import Path
from functools import partial, update_wrapper
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import Visualization_Overview, Visualization_Distributions, Visualization_Correlations

logger = logging.getLogger(__name__)

def create_pipeline(**kwargs) -> Pipeline:
    params           = kwargs.get("params", {})
    run_id           = params.get("run_id", "debug")
    visualize_only   = params.get("visualize_only", None)
    model_params     = params.get("model_parameters", {})
    train_datasets   = params.get("training_datasets", [])
    default_cv       = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

    if visualize_only and isinstance(visualize_only, str):
        visualize_only = [visualize_only]

    subpipelines = []

    for model_name, combos in model_params.items():
        for combo_id, cfg in combos.items():

            if "param_grid" in cfg:
                hyper_str = "gridsearch"
            elif "hyperparams" in cfg:
                hp = cfg["hyperparams"] or {}
                hyper_str = (
                    "_".join(f"{k}-{v}" for k, v in sorted(hp.items())) if hp else "default"
                )
            else:
                continue  # combinación mal definida

            cv     = cfg.get("cv_settings", default_cv)
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key   = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"

                if visualize_only and full_key not in visualize_only:
                    continue

                input_json = f"evaluation.{run_id}.Metrics_{full_key}"
                tag        = full_key

                subpipelines.append(
                    pipeline([
                        node(
                            func=Visualization_Overview,
                            inputs=[input_json],
                            outputs=f"visualization.{run_id}.{full_key}_overview",
                            name=f"VISUALIZATION_overview_{full_key}",
                            tags=[tag, "pipeline_visualization", "node_visualization_overview"],
                        ),
                        node(
                            func=Visualization_Distributions,
                            inputs=[input_json],
                            outputs=f"visualization.{run_id}.{full_key}_distributions",
                            name=f"VISUALIZATION_distributions_{full_key}",
                            tags=[tag, "pipeline_visualization", "node_visualization_distributions"],
                        ),
                        node(
                            func=Visualization_Correlations,
                            inputs=[input_json],
                            outputs=f"visualization.{run_id}.{full_key}_correlations",
                            name=f"VISUALIZATION_correlations_{full_key}",
                            tags=[tag, "pipeline_visualization", "node_visualization_correlations"],
                        ),
                    ])
                )

    if not subpipelines:
        logger.info("No se procesaron subpipelines de visualización: no se hallaron métricas esperadas.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))
