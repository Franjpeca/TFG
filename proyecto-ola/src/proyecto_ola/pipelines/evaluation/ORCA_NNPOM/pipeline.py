from kedro.pipeline import Pipeline, node
from proyecto_ola.utils.wrappers import make_predict_wrapper, make_evaluate_wrapper
from .nodes import Predict_ORCA_NNPOM, Evaluate_ORCA_NNPOM
import re

def create_pipeline(
    param_key: str,
    model_type: str,
    model_ds: str,
    dataset_name: str,
    prediction_ds: str,
    output_ds: str,
    dataset_id: str,
) -> Pipeline:

    match = re.search(r'grid_\d+', param_key)
    grid_id = match.group(0) if match else "grid_unknown"

    wrapped_predict = make_predict_wrapper(
        Predict_ORCA_NNPOM, param_key, dataset_id
    )
    wrapped_evaluate = make_evaluate_wrapper(
        Evaluate_ORCA_NNPOM, param_key, model_type, dataset_id
    )

    return Pipeline([
        node(
            func=wrapped_predict,
            inputs=[model_ds, dataset_name],
            outputs=[
                prediction_ds,
                f"expected_labels_{param_key}",
                f"model_params_{param_key}",
            ],
            name=f"PREDICT_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                f"grid_{grid_id}",
                f"model_{model_type}_{grid_id}",
                f"model_{model_type}_{grid_id}_dataset_{dataset_id}",
                "pipeline_evaluation",
                "node_predict_model",
            ],
        ),
        node(
            func=wrapped_evaluate,
            inputs=[
                f"expected_labels_{param_key}",
                prediction_ds,
                f"model_params_{param_key}",
                "params:execution_folder",
            ],
            outputs=output_ds,
            name=f"EVALUATE_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                f"grid_{grid_id}",
                f"model_{model_type}_{grid_id}",
                f"model_{model_type}_{grid_id}_dataset_{dataset_id}",
                "pipeline_evaluation",
                "node_evaluate_model",
            ],
        ),
    ])
