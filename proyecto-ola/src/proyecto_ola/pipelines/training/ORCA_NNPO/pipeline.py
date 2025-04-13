from kedro.pipeline import Pipeline, node
from .nodes import ORCA_NNPO

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ORCA_NNPO,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas5"
                ],
                name="ORCA_NNPO_node"
            )
        ]
    )