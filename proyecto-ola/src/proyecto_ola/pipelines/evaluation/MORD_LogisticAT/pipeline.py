from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import Evaluate_MORD_LogisticAT
from functools import partial, update_wrapper


def create_pipeline(param_key: str,
                    model_type: str,
                    model_ds: str,
                    dataset_name: str,
                    output_ds: str,
                    dataset_id: str) -> Pipeline:

    # Parametros que no son realmente inputs
    wrapped = partial(
        Evaluate_MORD_LogisticAT,
        model_id=param_key,
        model_type=model_type,
        dataset_id=dataset_id,
    )

    # Actualizamos el wrapped para ocultarlo en kedro viz
    wrapped = update_wrapper(wrapped, Evaluate_MORD_LogisticAT)

    return Pipeline([
        node(
            func=wrapped,   # Elementos del wrapped
            inputs=[model_ds, dataset_name],  # Solo inputs reales
            outputs=output_ds,
            name=f"EVALUATE_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                "pipeline_evaluation",
                "node_evaluate_model"
            ]
        )
    ])