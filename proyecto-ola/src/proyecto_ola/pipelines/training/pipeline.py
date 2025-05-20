from kedro.pipeline import Pipeline
from kedro.io import MemoryDataset

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


def create_pipeline(**kwargs) -> Pipeline:
    params = kwargs.get("params", {})
    run_id = params.get("run_id", "debug")
    model_params = params.get("model_parameters", {})
    train_datasets = params.get("training_datasets", [])

    subpipelines = []

    for model_name, combos in model_params.items():
        for combo_id, cfg in combos.items():
            if "param_grid" not in cfg:
                print(f"{model_name}/{combo_id} no tiene param_grid definido. Se omite.")
                continue

            param_ds = f"params:model_parameters.{model_name}.{combo_id}.param_grid"
            hyper_str = "gridsearch"

            cv = cfg.get("cv_settings", params.get("cv_settings", {"n_splits": 5, "random_state": 42}))
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            for train_ds in train_datasets:

                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"
                output_ds = f"training.{run_id}.Model_{full_key}"

                if model_name == "LAD":
                    subpipelines.append(
                        create_MORD_LAD_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "LogisticAT":
                    subpipelines.append(
                        create_MORD_LogisticAT_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "LogisticIT":
                    subpipelines.append(
                        create_MORD_LogisticIT_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )
                    
                if model_name == "OrdinalRidge":
                    subpipelines.append(
                        create_MORD_OrdinalRidge_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "NNOP":
                    subpipelines.append(
                        create_ORCA_NNOP_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "NNPOM":
                    subpipelines.append(
                        create_ORCA_NNPOM_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )


                if model_name == "OrdinalDecomposition":
                    subpipelines.append(
                        create_ORCA_OrdinalDecomposition_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "REDSVM":
                    subpipelines.append(
                        create_ORCA_REDSVM_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

                if model_name == "SVOREX":
                    subpipelines.append(
                        create_ORCA_SVOREX_pipeline(
                            param_key=full_key,
                            model_type=model_name,
                            param_ds=param_ds,
                            output_ds=output_ds,
                            dataset_name=train_ds,
                            param_type=f"params:model_parameters.{model_name}.{combo_id}.param_type",
                            cv_settings="params:cv_settings",
                            dataset_id=dataset_id,
                        ).tag([
                            full_key,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_training",
                        ])
                    )

    return sum(subpipelines, Pipeline([]))