import logging
from kedro.pipeline import Pipeline

from proyecto_ola.pipelines.evaluation.MORD_LAD.pipeline import create_pipeline as create_MORD_LAD_pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticIT.pipeline import create_pipeline as create_MORD_LogisticIT_pipeline
from proyecto_ola.pipelines.evaluation.MORD_OrdinalRidge.pipeline import create_pipeline as create_MORD_OrdinalRidge_pipeline
from proyecto_ola.pipelines.evaluation.ORCA_NNOP import create_pipeline as create_ORCA_NNOP_pipeline
from proyecto_ola.pipelines.evaluation.ORCA_NNPOM.pipeline import create_pipeline as create_ORCA_NNPOM_pipeline
from proyecto_ola.pipelines.evaluation.ORCA_OrdinalDecomposition.pipeline import create_pipeline as create_ORCA_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.evaluation.ORCA_REDSVM.pipeline import create_pipeline as create_ORCA_REDSVM_pipeline
from proyecto_ola.pipelines.evaluation.ORCA_SVOREX.pipeline import create_pipeline as create_ORCA_SVOREX_evaluation_pipeline

logger = logging.getLogger(__name__)

MODEL_PIPELINES = {
    "LAD": create_MORD_LAD_pipeline,
    "LogisticAT": create_MORD_LogisticAT_pipeline,
    "LogisticIT": create_MORD_LogisticIT_pipeline,
    "OrdinalRidge": create_MORD_OrdinalRidge_pipeline,
    "NNOP": create_ORCA_NNOP_pipeline,
    "NNPOM": create_ORCA_NNPOM_pipeline,
    "OrdinalDecomposition": create_ORCA_OrdinalDecomposition_pipeline,
    "REDSVM": create_ORCA_REDSVM_pipeline,
    "SVOREX": create_ORCA_SVOREX_evaluation_pipeline,
}


def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    run_id = params.get("run_id", "debug")
    evaluate_only = params.get("evaluate_only", None)
    test_datasets = params.get("test_datasets", [])
    model_params = params.get("model_parameters", {})
    train_datasets = params.get("training_datasets", [])
    cv_default = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

    if isinstance(evaluate_only, str):
        evaluate_only = [evaluate_only]

    subpipelines = []

    for model_name, combos in model_params.items():
        if model_name not in MODEL_PIPELINES:
            logger.warning(f"Modelo no reconocido: {model_name}")
            continue

        for combo_id, cfg in combos.items():
            if "param_grid" not in cfg:
                continue

            cv = cfg.get("cv_settings", cv_default)
            hyper_str = "gridsearch"
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"

                if evaluate_only and full_key not in evaluate_only:
                    continue

                test_ds_name = f"cleaned_{dataset_id}_test_ordinal"
                if test_datasets and test_ds_name not in test_datasets:
                    continue

                model_ds = f"training.{run_id}.Model_{full_key}"
                output_ds = f"evaluation.{run_id}.Metrics_{full_key}"
                prediction_ds = f"evaluation.{run_id}.Predictions_{full_key}"

                pipeline_fn = MODEL_PIPELINES[model_name]
                subpipeline = pipeline_fn(
                    param_key=full_key,
                    model_type=model_name,
                    model_ds=model_ds,
                    dataset_name=test_ds_name,
                    prediction_ds=prediction_ds,
                    output_ds=output_ds,
                    dataset_id=dataset_id,
                ).tag([
                    full_key,
                    f"dataset_{dataset_id}",
                    f"model_{model_name}",
                    "pipeline_evaluation",
                ])

                subpipelines.append(subpipeline)

    if not subpipelines:
        logger.info("No se procesaron subpipelines de evaluación: no se hallaron modelos válidos.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))
