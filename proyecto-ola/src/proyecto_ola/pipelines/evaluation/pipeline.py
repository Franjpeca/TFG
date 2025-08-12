import logging
from kedro.pipeline import Pipeline
from kedro.pipeline import Pipeline, node
import sys
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
    run_id = params.get("run_id", "debug")
    evaluate_only = params.get("evaluate_only", None)
    test_datasets = params.get("test_datasets", [])

    if isinstance(evaluate_only, str):
        evaluate_only = [evaluate_only]

    # 1) Si el hook inyectó evaluated_keys, respétalas
    evaluated_keys = params.get("evaluated_keys", [])
    if isinstance(evaluated_keys, str):
        evaluated_keys = [evaluated_keys]

    full_keys = []
    if evaluated_keys:
        full_keys = list(evaluated_keys)
    else:
        # 2) Determinar carpeta a usar (sin helper):
        exec_folder = params.get("execution_folder")

        # Leer execution_folder del CLI (--params) inline
        if not exec_folder:
            for arg in sys.argv:
                if arg.startswith("--params="):
                    raw = arg.split("=", 1)[1].strip().strip('"').strip("'")
                    for kv in raw.split(","):
                        if "=" in kv:
                            k, v = kv.split("=", 1)
                            if k.strip() == "execution_folder":
                                exec_folder = v.strip()
                                break
                if exec_folder:
                    break
            if not exec_folder and "--params" in sys.argv:
                try:
                    i = sys.argv.index("--params")
                    if i + 1 < len(sys.argv):
                        raw = sys.argv[i + 1]
                        for kv in raw.split(","):
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                if k.strip() == "execution_folder":
                                    exec_folder = v.strip()
                                    break
                except Exception:
                    pass

        models_dir: Optional[Path] = None

        if exec_folder:
            d = Path("data") / "04_models" / exec_folder
            if not d.exists():
                logger.error(f"[evaluation] La carpeta forzada no existe: {d}")
                return Pipeline([
                    node(lambda _x: None, inputs="params:run_id", outputs=None,
                         name="EVALUATION_NOOP", tags=["pipeline_evaluation"])
                ])
            if not any(d.glob("Model_*.pkl")):
                logger.error(f"[evaluation] La carpeta forzada no contiene modelos: {d}")
                return Pipeline([
                    node(lambda _x: None, inputs="params:run_id", outputs=None,
                         name="EVALUATION_NOOP", tags=["pipeline_evaluation"])
                ])
            models_dir = d
            logger.info(f"[evaluation] usando carpeta forzada: {models_dir.name}")
        else:
            # ❗ Mismo criterio que el hook: última carpeta por mtime del DIRECTORIO
            base = Path("data") / "04_models"
            best_dir: Optional[Path] = None
            best_ts = -1.0
            for d in base.glob("*_*"):
                if not any(d.glob("Model_*.pkl")):
                    continue
                ts = d.stat().st_mtime  # ← igual que el hook
                if ts > best_ts:
                    best_ts, best_dir = ts, d
            models_dir = best_dir
            if not models_dir:
                logger.info("[evaluation] No se encontraron modelos en data/04_models. Pipeline vacío.")
                return Pipeline([
                    node(lambda _x: None, inputs="params:run_id", outputs=None,
                         name="EVALUATION_NOOP", tags=["pipeline_evaluation"])
                ])
            logger.info(f"[evaluation] usando carpeta detectada: {models_dir.name}")

        # 3) Tomar SOLO los modelos presentes en la carpeta elegida
        full_keys = [p.stem.replace("Model_", "", 1) for p in sorted(models_dir.glob("Model_*.pkl"))]

    if not full_keys:
        logger.info("No se procesaron subpipelines de evaluación: no se hallaron modelos válidos.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="EVALUATION_NOOP", tags=["pipeline_evaluation"])
        ])

    subpipelines = []

    # 4) Construcción de subpipelines SOLO para las keys detectadas en disco
    for full_key in full_keys:
        if evaluate_only and full_key not in evaluate_only:
            continue

        # Estructura: <model_name>_<combo_id>_<dataset_id>_<hyper_str>_<cv_str>
        tokens = full_key.split("_")
        if len(tokens) < 6:
            logger.warning(f"[evaluation] full_key inesperada: {full_key}")
            continue

        model_name = tokens[0]
        dataset_id = tokens[-6]  # robusto aunque combo_id tenga '_'

        if model_name not in MODEL_PIPELINES:
            logger.warning(f"Modelo no reconocido (omitido): {model_name}")
            continue

        model_ds = f"training.{run_id}.Model_{full_key}"
        output_ds = f"evaluation.{run_id}.Metrics_{full_key}"
        prediction_ds = f"evaluation.{run_id}.Predicted_Labels_{full_key}"

        pipeline_fn = MODEL_PIPELINES[model_name]
        subpipeline = pipeline_fn(
            param_key=full_key,
            model_type=model_name,
            model_ds=model_ds,
            dataset_name=f"cleaned_{dataset_id}_test_ordinal",
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
        logger.info("No se procesaron subpipelines de evaluación tras filtrar. Pipeline vacío.")
        return Pipeline([
            node(lambda _x: None, inputs="params:run_id", outputs=None,
                 name="EVALUATION_NOOP", tags=["pipeline_evaluation"])
        ])

    return sum(subpipelines, Pipeline([]))