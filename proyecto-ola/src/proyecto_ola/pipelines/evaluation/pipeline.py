import logging
from kedro.pipeline import Pipeline, node
from proyecto_ola.utils.pipelines_utils import find_parameters_cli, only_evaluation
from pathlib import Path
from typing import Optional

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
    run_id = params.get("run_id", "001")
    evaluate_only = params.get("evaluate_only")
    model_parameters = params.get("model_parameters", {})
    training_datasets = params.get("training_datasets", [])
    default_cv = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

    # Convertir evaluate_only a lista si viene como string
    if isinstance(evaluate_only, str):
        evaluate_only = [evaluate_only]

    # Leer evaluated_keys para el hook, pasar a lista si hace falta
    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    # Buscar carpeta de ejecucion por param o CLI
    exec_folder = find_parameters_cli("execution_folder", params)

    # Buscar carpeta de modelos segun exec_folder o ultima ejecucion
    model_dir: Optional[Path] = None
    if exec_folder:
        possible_dir = Path("data") / "04_models" / exec_folder
        if possible_dir.exists() and any(possible_dir.glob("Model_*.pkl")):
            model_dir = possible_dir
            logger.info(f"[INFO] Usando la ejecucion indicada: {model_dir.name}")
        else:
            logger.warning(f"[INFO] La carpeta indicada no existe o no contiene modelos: {possible_dir}")
    else:
        # Buscar la ultima ejecucion con modelos en disco
        dirs = sorted((d for d in (Path("data") / "04_models").glob("*_*") if any(d.glob("Model_*.pkl"))), key=lambda d: d.stat().st_mtime, reverse=True)
        model_dir = dirs[0] if dirs else None
        if model_dir:
            logger.info(f"[INFO] Usando ultima ejecucion: {model_dir.name}")

    # Extraer keys de modelos en disco
    disk_keys = [p.stem.replace("Model_", "", 1) for p in sorted(model_dir.glob("Model_*.pkl"))] if model_dir else []

    # Generar combinaciones posibles desde parametros
    param_keys = []
    for model_name, combos in model_parameters.items():
        for combo_id, cfg in combos.items():
            if "param_grid" not in cfg:
                continue
            cv = cfg.get("cv_settings", default_cv)
            cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"
            for train_ds in training_datasets:
                dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                param_keys.append(f"{model_name}_{combo_id}_{dataset_id}_gridsearch_{cv_str}")

    # Seleccionar las keys finales segun reglas del pipeline
    if evaluated_keys:
        selected_keys = list(dict.fromkeys(evaluated_keys))
    elif only_evaluation():
        selected_keys = list(dict.fromkeys(disk_keys))
    else:
        selected_keys = list(dict.fromkeys(disk_keys + param_keys))

    # Salida temprana si no hay keys para evaluar
    if not selected_keys:
        logger.info("No se procesaron subpipelines de evaluacion: no se hallaron modelos ni combinaciones.")
        return Pipeline([node(lambda _x: None, inputs="params:run_id", outputs=None, name="EVALUATION_NOOP", tags=["pipeline_evaluation"])])

    # Construir subpipelines por cada combinacion a evaluar
    subpipelines = []
    for key in selected_keys:
        if evaluate_only and key not in evaluate_only:
            continue

        tokens = key.split("_")
        if len(tokens) < 6:
            logger.warning(f"[evaluation] full_key inesperada: {key}")
            continue

        model_name = tokens[0]
        dataset_id = tokens[-6]
        if model_name not in MODEL_PIPELINES:
            logger.warning(f"Modelo no reconocido (se omite): {model_name}")
            continue

        pipeline_fn = MODEL_PIPELINES[model_name]
        subpipeline = pipeline_fn(
            param_key=key,
            model_type=model_name,
            model_ds=f"training.{run_id}.Model_{key}",
            dataset_name=f"cleaned_{dataset_id}_test_ordinal",
            prediction_ds=f"evaluation.{run_id}.Predicted_Labels_{key}",
            output_ds=f"evaluation.{run_id}.Metrics_{key}",
            dataset_id=dataset_id,
        ).tag([
            key,
            f"dataset_{dataset_id}",
            f"model_{model_name}",
            "pipeline_evaluation",
        ])
        subpipelines.append(subpipeline)

    # Salida temprana si tras filtrar no quedan subpipelines validos
    if not subpipelines:
        logger.info("No se procesaron subpipelines de evaluacion tras filtrar. Pipeline vacio.")
        return Pipeline([node(lambda _x: None, inputs="params:run_id", outputs=None, name="EVALUATION_NOOP", tags=["pipeline_evaluation"])])

    # Devolver la suma de todos los subpipelines
    return sum(subpipelines, Pipeline([]))