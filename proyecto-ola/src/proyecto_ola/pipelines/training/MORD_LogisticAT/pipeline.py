from kedro.pipeline import Pipeline, node
from .nodes import MORD_LogisticAT

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=MORD_LogisticAT,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas1"
                ],
                name="MORD_LogisticAT_node"
            )
        ]
    )