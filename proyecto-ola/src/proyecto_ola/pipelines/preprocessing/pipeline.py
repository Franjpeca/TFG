from kedro.pipeline import Pipeline, node
from .nodes import clean_data_all, merge_data

def create_pipeline(**kwargs) -> Pipeline:
    
    return Pipeline(
        [
            node(
                func=clean_data_all, #Funcion del nodo a la que llamo
                inputs=[
                    "46014_train_ordinal", "46014_test_ordinal",
                    "46025_train_ordinal", "46025_test_ordinal",
                    "46042_train_ordinal", "46042_test_ordinal",
                    "46053_train_ordinal", "46053_test_ordinal",
                    "46069_train_ordinal", "46069_test_ordinal"
                ],
                outputs=[
                    "cleaned_46014_train_ordinal", "cleaned_46014_test_ordinal",
                    "cleaned_46025_train_ordinal", "cleaned_46025_test_ordinal",
                    "cleaned_46042_train_ordinal", "cleaned_46042_test_ordinal",
                    "cleaned_46053_train_ordinal", "cleaned_46053_test_ordinal",
                    "cleaned_46069_train_ordinal", "cleaned_46069_test_ordinal"
                ],
                name="clean_data_node" # Nombre del nodo
            ),
             node(
                func=merge_data,
                inputs=[
                    "cleaned_46014_train_ordinal", "cleaned_46014_test_ordinal",
                    "cleaned_46025_train_ordinal", "cleaned_46025_test_ordinal",
                    "cleaned_46042_train_ordinal", "cleaned_46042_test_ordinal",
                    "cleaned_46053_train_ordinal", "cleaned_46053_test_ordinal",
                    "cleaned_46069_train_ordinal", "cleaned_46069_test_ordinal"
                ],
                outputs=[
                    "train_ordinal", "test_ordinal",
                ],
                name="merge_data_node"
            )
        ]
    )