import logging
from pathlib import Path
from functools import partial, update_wrapper
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import Visualization_Overview, Visualization_Distributions, Visualization_Correlations

logger = logging.getLogger(__name__)

def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    run_id = params.get("run_id", "debug")

    # Detect evaluation JSON files
    model_data_files = params.get("json_datasets", []) or []
    if not model_data_files:
        eval_dir = Path("data/07_model_output") / run_id
        if eval_dir.exists():
            model_data_files = [
                f"evaluation.{run_id}.{p.stem}" for p in eval_dir.glob("Metrics_*.json")
            ]

    if not model_data_files:
        logger.warning(f"[Visualization] No JSON files found for run_id={run_id}")
        return Pipeline([])

    subpipelines = []
    for model_data in model_data_files:
        full_key = model_data.replace(f"evaluation.{run_id}.Metrics_", "")
        tag = f"{full_key}"

        subpipelines.append(
            pipeline([
                node(
                    func=Visualization_Overview,
                    inputs=[model_data],
                    outputs=f"visualization.{run_id}.{full_key}_overview",
                    name=f"VISUALIZATION_overview_{full_key}",
                    tags=[tag, 
                        "pipeline_visualization",
                        "node_visualization_overview"],
                ),
                node(
                    func=Visualization_Distributions,
                    inputs=[model_data],
                    outputs=f"visualization.{run_id}.{full_key}_distributions",
                    name=f"VISUALIZATION_distributions_{full_key}",
                    tags=[tag, 
                        "pipeline_visualization",
                        "node_visualization_distributions"],
                ),
                node(
                    func=Visualization_Correlations,
                    inputs=[model_data],
                    outputs=f"visualization.{run_id}.{full_key}_correlations",
                    name=f"VISUALIZATION_correlations_{full_key}",
                    tags=[tag, 
                        "pipeline_visualization",
                        "node_visualization_correlations"],
                ),
            ])
        )
    return sum(subpipelines, Pipeline([]))

