from kedro.pipeline import Pipeline, node
from .nodes import MORD_OrdinalRidge

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=MORD_OrdinalRidge,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas4"
                ],
                name="MMORD_OrdinalRidge_node"
            )
        ]
    )