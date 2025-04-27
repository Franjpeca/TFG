from kedro.pipeline import Pipeline
from kedro.io import MemoryDataset

# Importa todos los subpipelines
# Mord
from proyecto_ola.pipelines.training.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline
from proyecto_ola.pipelines.training.MORD_LogisticIT.pipeline import create_pipeline as create_MORD_LogisticIT_pipeline
from proyecto_ola.pipelines.training.MORD_LAD.pipeline import create_pipeline as create_MORD_LAD_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalRidge.pipeline import create_pipeline as create_MORD_OrdinalRidge_pipeline
from proyecto_ola.pipelines.training.MORD_MulticlassLogistic.pipeline import create_pipeline as create_MORD_MulticlassLogistic_pipeline

# ORCA
from proyecto_ola.pipelines.training.ORCA_NNOP.pipeline import create_pipeline as create_ORCA_NNOP_pipeline
from proyecto_ola.pipelines.training.ORCA_NNPOM.pipeline import create_pipeline as create_ORCA_NNPOM_pipeline
from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition.pipeline import create_pipeline as create_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.training.ORCA_REDSVM.pipeline import create_pipeline as create_ORCA_REDSVM_pipeline
from proyecto_ola.pipelines.training.ORCA_SVOREX.pipeline import create_pipeline as create_ORCA_SVOREX_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    print("Preparando los modelos")

    params       = kwargs.get("params", {})
    run_id       = params.get("run_id", "debug")
    model_params = params.get("model_parameters", {})

    subpipelines = []

    for model_name, combos in model_params.items():
        for combo_id, cfg in combos.items():
            # Datos de entrada
            dataset_name = cfg.get("dataset_name", "train_ordinal")
            param_ds     = f"params:model_parameters.{model_name}.{combo_id}.hyperparams"

            # Construir cadena de hiperparámetros
            hyperparams = cfg.get("hyperparams", {})
            hyper_str   = "_".join(f"{k}-{v}" for k, v in hyperparams.items()) if hyperparams else "default"

            # Nombre final que usarán tanto el hook como el pipeline
            full_key   = f"{model_name}_{combo_id}_{hyper_str}"
            output_ds  = f"models.{run_id}.{full_key}"
            tag        = full_key

            if model_name == "LogisticAT":
                subpipelines.append(
                    create_MORD_LogisticAT_pipeline(
                        param_key    = tag,
                        param_ds     = param_ds,
                        output_ds    = output_ds,
                        dataset_name = dataset_name,
                    ).tag(tag)
                )

            # elif model_name == "LogisticIT":
            #     all_pipelines.append(create_MORD_LogisticIT_pipeline(params=config))
            # elif model_name == "LAD":
            #     all_pipelines.append(create_MORD_LAD_pipeline(params=config))
            # elif model_name == "OrdinalRidge":
            #     all_pipelines.append(create_MORD_OrdinalRidge_pipeline(params=config))
            # elif model_name == "MulticlassLogistic":
            #     all_pipelines.append(create_MORD_MulticlassLogistic_pipeline(params=config))
            # elif model_name == "NNOP":
            #     all_pipelines.append(create_ORCA_NNOP_pipeline(params=config))
            # elif model_name == "NNPOM":
            #     all_pipelines.append(create_ORCA_NNPOM_pipeline(params=config))
            #elif model_name == "OrdinalDecomposition":
                #print("Contenido de OrdinalDecomposition desde el pipeline general:", config) 
                #all_pipelines.append(create_OrdinalDecomposition_pipeline(param_key=param_key, params=config))
            #elif model_name == "REDSVM":
            #    all_pipelines.append(create_ORCA_REDSVM_pipeline(params=config))
            #elif model_name == "SVOREX":
            #    all_pipelines.append(create_ORCA_SVOREX_pipeline(params=config))
            #else:
                #raise ValueError(f"Modelo desconocido: {model_name}")
                #print(f"Modelo desconocido: {model_name}")
            
    print(f"Total de subpipelines creados: {len(subpipelines)}")
        
    return sum(subpipelines, Pipeline([]))