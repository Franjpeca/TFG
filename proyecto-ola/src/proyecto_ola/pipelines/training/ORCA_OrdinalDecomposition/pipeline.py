from kedro.pipeline import Pipeline, node
from .nodes import ORCA_OrdinalDecomposition

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ORCA_OrdinalDecomposition,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas7"
                ],
                name="ORCA_OrdinalDecomposition"
            )
        ]
    )