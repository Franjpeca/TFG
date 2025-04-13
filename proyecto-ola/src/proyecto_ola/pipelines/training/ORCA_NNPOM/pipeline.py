from kedro.pipeline import Pipeline, node
from .nodes import ORCA_NNPOM

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ORCA_NNPOM,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas6"
                ],
                name="ORCA_NNPOM_node"
            )
        ]
    )