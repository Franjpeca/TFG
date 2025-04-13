from kedro.pipeline import Pipeline, node
from .nodes import MORD_OrdinalLogisticRegression

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=MORD_OrdinalLogisticRegression,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas3"
                ],
                name="MORD_OrdinalLogisticRegression_node"
            )
        ]
    )