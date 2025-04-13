from kedro.pipeline import Pipeline, node
from .nodes import MORD_MNLogit

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=MORD_MNLogit,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas2"
                ],
                name="MORD_MNLogit_node"
            )
        ]
    )