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
#from proyecto_ola.pipelines.training.ORCA_NNOP.pipeline import create_pipeline as create_ORCA_NNOP_pipeline
#from proyecto_ola.pipelines.training.ORCA_NNPOM.pipeline import create_pipeline as create_ORCA_NNPOM_pipeline
#from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition.pipeline import create_pipeline as create_OrdinalDecomposition_pipeline
#from proyecto_ola.pipelines.training.ORCA_REDSVM.pipeline import create_pipeline as create_ORCA_REDSVM_pipeline
#from proyecto_ola.pipelines.training.ORCA_SVOREX.pipeline import create_pipeline as create_ORCA_SVOREX_pipeline


def create_pipeline(**kwargs) -> Pipeline:
    # Toma los parametros dados por el pipeline
    params = kwargs.get("params", {})
    
    # Identificador de la ejecucion, por defecto es debug
    run_id = params.get("run_id", "debug")
    # Obtiene los parametros del modelo, ya sea por parameters.yml o CLI
    model_params = params.get("model_parameters", {})
    # Lista de datasets de entrenamiento definidos en parameters.yml
    train_datasets = params.get("training_datasets", [])

    # Lista que acumula los pipelines que se vayan a querer ejecutar
    subpipelines = []

    # For para recorrer los modelos
    for model_name, combos in model_params.items():
        # For para recorrer las combinaciones de hiperparametros
        for combo_id, cfg in combos.items():

            # Detectar tipo de entrenamiento: manual o gridsearch
            # Miramos la variable cfg donde se ve reflejada esta "bandera"
            if "hyperparams" in cfg:
                # Si estamos en modo manual, establecemos todo como tal
                param_ds = f"params:model_parameters.{model_name}.{combo_id}.hyperparams"
                hyperparams = cfg["hyperparams"]
                hyper_str = "_".join(f"{k}-{v}" for k, v in hyperparams.items()) if hyperparams else "default"

            # Lo mismo pero para el caso de hacer gridsearch
            elif "param_grid" in cfg and "hyperparams" not in cfg:
                # Solo hacemos gridsearch si no hay parametros manuales
                param_ds = f"params:model_parameters.{model_name}.{combo_id}.param_grid"
                hyper_str = "gridsearch"
            
            # Manejo de errores
            else:
                print(f"{model_name}/{combo_id} no tiene ni hyperparams ni param_grid definidos. Se omite.")
                continue

            # Generar identificadores
            # Obtener configuracion de validacion cruzada (con valores por defecto si no existen)
            cv = cfg.get("cv_settings", {"n_splits": 5, "random_state": 42})
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

            # For para recorrer cada dataset de entrenamiento
            for train_ds in train_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"
                output_ds = f"models.{run_id}.{full_key}"
                tag = full_key
                param_type_ds = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                cv_settings_ds = "params:cv_settings"

                # Construir subpipeline especifico segun modelo
                if model_name == "LogisticAT":
                    subpipelines.append(
                        create_MORD_LogisticAT_pipeline(
                            param_key    = tag,
                            param_ds     = param_ds,
                            output_ds    = output_ds,
                            dataset_name = train_ds,
                            param_type   = param_type_ds,
                            cv_settings  = cv_settings_ds
                        ).tag(tag)
                    )

    return sum(subpipelines, Pipeline([]))
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
            
            #print(f"Total de subpipelines creados: {len(subpipelines)}")
        
            #return sum(subpipelines, Pipeline([]))