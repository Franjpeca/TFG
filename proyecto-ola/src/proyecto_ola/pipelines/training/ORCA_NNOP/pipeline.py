from kedro.pipeline import Pipeline, node
from proyecto_ola.utils.wrappers import make_train_wrapper
from .nodes import Train_ORCA_NNOP
import re

def create_pipeline(
                    param_key: str,
                    model_type: str,
                    param_ds: str,
                    output_ds: str,
                    dataset_name: str,
                    cv_settings: str,
                    dataset_id: str,
                ) -> Pipeline:

    match = re.search(r'grid_\d+', param_key)
    grid_id = match.group(0) if match else "grid_unknown"

    wrapped = make_train_wrapper(
        Train_ORCA_NNOP,
        model_id=param_key,
        dataset_id=dataset_id
    )

    return Pipeline([
        node(
            func=wrapped,
            inputs=[dataset_name, param_ds, cv_settings],
            outputs=output_ds,
            name=f"TRAIN_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                f"grid_{grid_id}",
                f"model_{model_type}_{grid_id}",
                f"model_{model_type}_{grid_id}_dataset_{dataset_id}",
                "pipeline_training",
                "node_train_model"
            ],
        )
    ])
