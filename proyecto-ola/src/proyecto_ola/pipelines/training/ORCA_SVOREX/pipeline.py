from kedro.pipeline import Pipeline, node
from .nodes import ORCA_SVOREX

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=ORCA_SVOREX,
                inputs=[
                    "train_ordinal", "test_ordinal",
                ],
                outputs=[
                    "pruebas9"
                ],
                name="ORCA_SVOREX_node"
            )
        ]
    )