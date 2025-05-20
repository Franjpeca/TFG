from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import Evaluate_MORD_LogisticIT
from functools import partial, update_wrapper


def create_pipeline(param_key: str,
                    model_type: str,
                    model_ds: str,
                    dataset_name: str,
                    output_ds: str,
                    dataset_id: str) -> Pipeline:

    wrapped = partial(
        Evaluate_MORD_LogisticIT,
        model_id=param_key,
        model_type=model_type,
        dataset_id=dataset_id,
    )

    wrapped = update_wrapper(wrapped, Evaluate_MORD_LogisticIT)

    return Pipeline([
        node(
            func=wrapped,
            inputs=[model_ds, dataset_name],
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