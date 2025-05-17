import logging
from pathlib import Path
from kedro.pipeline import Pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline

logger = logging.getLogger(__name__)

def create_pipeline(**kwargs) -> Pipeline:
    params         = kwargs.get("params", {})
    run_id         = params.get("run_id", "debug")
    evaluate_only  = params.get("evaluate_only", None)
    test_datasets  = params.get("test_datasets", [])

    # --- 1. Tomamos la misma info que usa el pipeline de training -------------
    model_params   = params.get("model_parameters", {})
    train_datasets = params.get("training_datasets", [])

    # Si el usuario pasó evaluate_only, lo convertimos a lista
    if evaluate_only and isinstance(evaluate_only, str):
        evaluate_only = [evaluate_only]

    subpipelines = []

    # --- 2. Re-creamos los full_key EXACTAMENTE IGUAL que en training ---------
    for model_name, combos in model_params.items():
        for combo_id, cfg in combos.items():

            # Detectar tipo de entrenamiento → hyper_str
            if "param_grid" in cfg:
                hyper_str = "gridsearch"
            elif "hyperparams" in cfg:
                hp = cfg["hyperparams"] or {}
                hyper_str = (
                    "_".join(f"{k}-{v}" for k, v in sorted(hp.items())) if hp else "default"
                )
            else:
                continue  # combinación mal definida

            # Configuración de CV
            cv      = cfg.get("cv_settings", {"n_splits": 5, "random_state": 42})
            cv_str  = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            # Recorremos los datasets de training
            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key   = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"

                # Filtrado opcional por evaluate_only
                if evaluate_only and full_key not in evaluate_only:
                    continue

                test_ds_name = f"cleaned_{dataset_id}_test_ordinal"
                if test_datasets and test_ds_name not in test_datasets:
                    continue

                model_ds = f"training.{run_id}.Model_{full_key}"
                output_ds = f"evaluation.{run_id}.Metrics_{full_key}"
                tag       = full_key

                if model_name == "LogisticAT":
                    subpipelines.append(
                        create_MORD_LogisticAT_pipeline(
                            param_key    = tag,
                            model_type   = model_name,
                            model_ds     = model_ds,
                            dataset_name = test_ds_name,
                            output_ds    = output_ds,
                            dataset_id   = dataset_id,
                        ).tag([
                            tag,
                            f"dataset_{dataset_id}",
                            f"model_{model_name}",
                            "pipeline_evaluation",
                        ])
                    )

    # --- 3. Si no hay nada que evaluar, devolvemos pipeline vacío ------------
    if not subpipelines:
        logger.info("No se procesaron subpipelines de evaluación: no se hallaron modelos válidos.")
        return Pipeline([])

    # --- 4. Devolvemos la suma de subpipelines --------------------------------
    return sum(subpipelines, Pipeline([]))