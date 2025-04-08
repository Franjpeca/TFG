from kedro.pipeline import Pipeline, node
from .nodes import orca_functions_node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=orca_functions_node,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas"
                ],
                name="orca_node"
            )
        ]
    )