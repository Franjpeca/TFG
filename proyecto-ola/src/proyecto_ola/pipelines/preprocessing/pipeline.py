from kedro.pipeline import Pipeline, node
from .nodes import clean_pair

def create_pipeline(dataset_ids=None, **kwargs):
    # Este pipeline espera que dataset_ids venga de register_pipelines()
    if dataset_ids is None:
        raise ValueError("dataset_ids no puede ser None. Asegúrate de pasarlos desde register_pipelines.")

    nodes = []
    for ds_id in dataset_ids:
        nodes.append(
            node(
                func=clean_pair,
                inputs=[
                    f"{ds_id}_train_ordinal",
                    f"{ds_id}_test_ordinal",
                    f"params:dataset_names.{ds_id}",
                    "params:preprocessing",
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
                ],
            )
        )
    return Pipeline(nodes)