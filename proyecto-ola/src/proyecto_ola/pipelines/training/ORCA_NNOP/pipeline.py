from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import Train_ORCA_NNOP
from functools import partial, update_wrapper


def create_pipeline(
    param_key: str,
    model_type: str,
    param_ds: str,
    output_ds: str,
    dataset_name: str,
    param_type: str,
    cv_settings: str,
    dataset_id: str,
) -> Pipeline:

    wrapped = partial(Train_ORCA_NNOP, dataset_id=dataset_id)
    wrapped = update_wrapper(wrapped, Train_ORCA_NNOP)  # oculta kwargs en Kedro-Viz

    return Pipeline(
        [
            node(
                func=wrapped,
                inputs=[dataset_name, param_ds, param_type, cv_settings],
                outputs=output_ds,
                name=f"TRAINING_Node_{param_key}",
                tags=[
                    param_key,
                    f"dataset_{dataset_id}",
                    f"model_{model_type}",
                    "pipeline_training",
                    "node_train_model",
                ],
            )
        ]
    )