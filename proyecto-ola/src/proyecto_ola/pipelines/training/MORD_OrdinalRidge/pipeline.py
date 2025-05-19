from kedro.pipeline import Pipeline, node
from kedro.io import MemoryDataset
from .nodes import Train_MORD_OrdinalRidge
from functools import partial, update_wrapper

def create_pipeline(param_key: str,
                    model_type: str,
                    param_ds: str,
                    output_ds: str,
                    dataset_name: str,
                    param_type: str,
                    cv_settings: str,
                    dataset_id: str
                    ) -> Pipeline:

    # Parametros que no son inputs reales
    wrapped = partial(
        Train_MORD_OrdinalRidge,
        dataset_id=dataset_id
    )
    wrapped = update_wrapper(wrapped, Train_MORD_OrdinalRidge)

    return Pipeline([
        node(
            func=wrapped,
            inputs=[
                dataset_name,
                param_ds,
                param_type,
                cv_settings
            ],
            outputs=output_ds,
            name=f"TRAINING_Node_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                "pipeline_training",
                "node_train_model"
            ]
        )
    ])