from kedro.pipeline import Pipeline, node
from .nodes import Predict_ORCA_NNOP, Evaluate_ORCA_NNOP
from functools import partial, update_wrapper

def create_pipeline(param_key: str,
                    model_type: str,
                    model_ds: str,
                    dataset_name: str,
                    prediction_ds: str,
                    output_ds: str,
                    dataset_id: str,
                    execution_folder: str,
                    ) -> Pipeline:

    wrapped_predict = partial(
        Predict_ORCA_NNOP,
        model_id=param_key,
        dataset_id=dataset_id,
    )
    wrapped_predict = update_wrapper(wrapped_predict, Predict_ORCA_NNOP)

    wrapped_evaluate = partial(
        Evaluate_ORCA_NNOP,
        model_id=param_key,
        model_type=model_type,
        dataset_id=dataset_id,
        execution_folder=execution_folder,
    )
    wrapped_evaluate = update_wrapper(wrapped_evaluate, Evaluate_ORCA_NNOP)

    return Pipeline([
        node(
            func=wrapped_predict,
            inputs=[model_ds, dataset_name],
            outputs=prediction_ds,
            name=f"PREDICT_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                "pipeline_evaluation",
                "node_predict_model"
            ]
        ),
        node(
            func=wrapped_evaluate,
            inputs=[model_ds, dataset_name, prediction_ds],
            outputs=output_ds,
            name=f"EVALUATE_{param_key}",
            tags=[
                param_key,
                f"dataset_{dataset_id}",
                f"model_{model_type}",
                "pipeline_evaluation",
                "node_evaluate_model"
            ]
        )
    ])