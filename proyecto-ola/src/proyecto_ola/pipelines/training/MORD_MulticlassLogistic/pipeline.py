from kedro.pipeline import Pipeline, node
from functools import partial, update_wrapper
from .nodes import Train_MORD_MulticlassLogistic

def create_pipeline(param_key: str,
                    model_type: str,
                    param_ds: str,
                    output_ds: str,
                    dataset_name: str,
                    cv_settings: str,
                    dataset_id: str
                    ) -> Pipeline:

    # Envolvemos la funcion con parametros que no son inputs reales
    wrapped = partial(
        Train_MORD_MulticlassLogistic,
        dataset_id=dataset_id
    )

    wrapped = update_wrapper(wrapped, Train_MORD_MulticlassLogistic)

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