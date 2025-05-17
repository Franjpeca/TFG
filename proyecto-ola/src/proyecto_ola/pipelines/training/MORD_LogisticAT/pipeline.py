from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import Train_MORD_LogisticAT
from functools import partial, update_wrapper

def create_pipeline(param_key: str,
                    model_type: str,
                    param_ds: str,
                    output_ds: str,
                    dataset_name: str,
                    param_type: str,
                    cv_settings: str,
                    dataset_id: str
                ) -> Pipeline:  # <- aquí faltaba el `:` al final

    # Parametros que no son realmente inputs
    wrapped = partial(
        Train_MORD_LogisticAT,
        dataset_id=dataset_id
    )

    # Actualizamos el wrapped para ocultarlo en kedro viz
    wrapped = update_wrapper(wrapped, Train_MORD_LogisticAT)

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