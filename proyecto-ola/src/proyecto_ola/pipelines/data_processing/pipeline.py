from kedro.pipeline import Pipeline, node
from .nodes import clean_data_all

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=clean_data_all,
                inputs=[
                    "46014_train_ordinal", "46014_test_ordinal",
                    "46025_train_ordinal", "46025_test_ordinal",
                    "46042_train_ordinal", "46042_test_ordinal",
                    "46053_train_ordinal", "46053_test_ordinal",
                    "46069_train_ordinal", "46069_test_ordinal"
                ],
                outputs="cleaned_data",  # El nodo simplemente regresa los datasets cargados
                name="clean_data_node"
            )
        ]
    )