from kedro.pipeline import Pipeline, node
from .nodes import ORCA_REDSVM

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ORCA_REDSVM,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas8"
                ],
                name="ORCA_REDSVM_node"
            )
        ]
    )