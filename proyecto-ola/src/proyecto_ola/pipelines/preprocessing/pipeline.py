from kedro.pipeline import Pipeline, node
from .nodes import clean_pair

def create_pipeline(dataset_ids=None, **kwargs):
    if dataset_ids is None:
        dataset_ids = [46014, 46025, 46042, 46053, 46069]

    nodes = []
    for ds_id in dataset_ids:
        nodes.append(
            node(
                func=clean_pair,
                inputs=[
                    f"{ds_id}_train_ordinal",
                    f"{ds_id}_test_ordinal",
                    f"params:dataset_names.{ds_id}",
                    "params:preprocessing.drop_penultimate"
                ],
                outputs=[
                    f"cleaned_{ds_id}_train_ordinal",
                    f"cleaned_{ds_id}_test_ordinal",
                ],
                name=f"PREPROCESSING_clean_pair_{ds_id}",
                tags=[
                    "pipeline_preprocessing",
                    "node_clean_data",
                    f"dataset_{ds_id}",
                ]
            )
        )
    return Pipeline(nodes)