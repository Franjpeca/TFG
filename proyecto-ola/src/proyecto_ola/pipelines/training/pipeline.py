from kedro.pipeline import Pipeline
import logging

# Mord
from proyecto_ola.pipelines.training.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline
from proyecto_ola.pipelines.training.MORD_LogisticIT.pipeline import create_pipeline as create_MORD_LogisticIT_pipeline
from proyecto_ola.pipelines.training.MORD_LAD.pipeline import create_pipeline as create_MORD_LAD_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalRidge.pipeline import create_pipeline as create_MORD_OrdinalRidge_pipeline

# ORCA
from proyecto_ola.pipelines.training.ORCA_NNOP.pipeline import create_pipeline as create_ORCA_NNOP_pipeline
from proyecto_ola.pipelines.training.ORCA_NNPOM.pipeline import create_pipeline as create_ORCA_NNPOM_pipeline
from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition.pipeline import create_pipeline as create_ORCA_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.training.ORCA_REDSVM.pipeline import create_pipeline as create_ORCA_REDSVM_pipeline
from proyecto_ola.pipelines.training.ORCA_SVOREX.pipeline import create_pipeline as create_ORCA_SVOREX_pipeline

from proyecto_ola.pipelines.training.CLASSIC_LinearRegression.pipeline import create_pipeline as create_CLASSIC_LinearRegression_pipeline
from proyecto_ola.pipelines.training.CLASSIC_DecisionTreeRegressor.pipeline import create_pipeline as create_CLASSIC_DecisionTreeRegressor_pipeline
from proyecto_ola.pipelines.training.CLASSIC_KNeighborsClassifier.pipeline import create_pipeline as create_CLASSIC_KNeighborsClassifier_pipeline

logger = logging.getLogger(__name__)

MODEL_PIPELINES = {
    "DecisionTreeRegressor": create_CLASSIC_DecisionTreeRegressor_pipeline,
    "KNeighborsClassifier": create_CLASSIC_KNeighborsClassifier_pipeline,
    "LinearRegression": create_CLASSIC_LinearRegression_pipeline,
    "LAD": create_MORD_LAD_pipeline,
    "LogisticAT": create_MORD_LogisticAT_pipeline,
    "LogisticIT": create_MORD_LogisticIT_pipeline,
    "OrdinalRidge": create_MORD_OrdinalRidge_pipeline,
    "NNOP": create_ORCA_NNOP_pipeline,
    "NNPOM": create_ORCA_NNPOM_pipeline,
    "OrdinalDecomposition": create_ORCA_OrdinalDecomposition_pipeline,
    "REDSVM": create_ORCA_REDSVM_pipeline,
    "SVOREX": create_ORCA_SVOREX_pipeline,
}

def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    run_id = params.get("run_id", "001")
    model_params = params.get("model_parameters", {})
    train_datasets = params.get("training_datasets", [])
    cv_default = params.get("cv_settings", {"n_splits": 5, "random_state": 42})
    training_settings = params.get("training_settings", {})
    seed_val = training_settings.get("seed", "unk")

    subpipelines = []

    for model_name, combos in model_params.items():
        if model_name not in MODEL_PIPELINES:
            logger.warning(f"Modelo no reconocido: {model_name}")
            continue

        for combo_id, cfg in combos.items():
            if "param_grid" not in cfg:
                logger.info(f"{model_name}/{combo_id} no tiene param_grid definido. Se omite.")
                continue

            param_ds = f"params:model_parameters.{model_name}.{combo_id}.param_grid"
            hyper_str = "gridsearch"
            cv = cfg.get("cv_settings", cv_default)
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key = f"{model_name}_{combo_id}_{dataset_id}_seed_{seed_val}_{hyper_str}_{cv_str}"
                output_ds = f"training.{run_id}.Model_{full_key}"

                pipeline_fn = MODEL_PIPELINES[model_name]

                pipeline = pipeline_fn(
                    param_key=full_key,
                    model_type=model_name,
                    param_ds=param_ds,
                    output_ds=output_ds,
                    dataset_name=train_ds,
                    cv_settings="params:cv_settings",
                    training_settings="params:training_settings",
                    dataset_id=dataset_id,
                ).tag([
                    full_key,
                    f"dataset_{dataset_id}",
                    f"model_{model_name}",
                    "pipeline_training",
                ])

                subpipelines.append(pipeline)

    return sum(subpipelines, Pipeline([]))
