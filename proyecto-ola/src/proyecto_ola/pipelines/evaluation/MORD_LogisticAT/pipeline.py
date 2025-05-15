from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import MORD_LogisticAT


def create_pipeline(param_key: str,
                    model_type: str,
                    model_ds: str,
                    dataset_name: str,
                    output_ds: str,
                    dataset_id: str) -> Pipeline:
    return Pipeline([
        node(
            func=MORD_LogisticAT,
            inputs=[
                model_ds,
                dataset_name,
                param_key,
                model_type,
                f"params:{param_key}_dataset_id"
            ],
            outputs=output_ds,
            name=f"evaluate_{param_key}",
            tags=[param_key],
        )
    ])