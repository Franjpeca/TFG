from kedro.pipeline import Pipeline, node
from kedro.io import DataCatalog, MemoryDataset
from .nodes import MORD_LogisticAT

def create_pipeline(param_key: str,
                    param_ds: str,
                    output_ds: str,
                    dataset_name: str,
                    param_type: str,
                    cv_settings: str) -> Pipeline:
    return Pipeline([
        node(
            func=MORD_LogisticAT,
            inputs={
                "dataset": dataset_name,
                "params": param_ds,
                "param_type": param_type,
                "cv_settings": cv_settings
            },
            outputs=output_ds,
            name=f"MORD_LogisticAT_node_{param_key}",
            tags=[param_key]
        )
    ])