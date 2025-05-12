from pathlib import Path
from kedro.pipeline import Pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline

def create_pipeline(**kwargs) -> Pipeline:

    params = kwargs.get("params", {})
    run_id = params.get("run_id", "debug")
    evaluate_only = params.get("evaluate_only", None)
    eval_ds_name = params.get("evaluation_dataset", "test_ordinal")
    models_dir = Path(f"data/06_models/{run_id}")

    if not models_dir.exists():
        print(f"No se encontró la carpeta de modelos: {models_dir}. No se realizará ninguna evaluación.")
        return Pipeline([])

    model_files = sorted(models_dir.glob("*.pkl"))
    model_keys = [f.stem for f in model_files]

    # Filtrar si se especifica evaluate_only
    if evaluate_only:
        if isinstance(evaluate_only, str):
            evaluate_only = [evaluate_only]
        model_keys = [k for k in model_keys if k in evaluate_only]

    if not model_keys:
        print(f"No se encontraron modelos para evaluar en {models_dir}. No se realizará ninguna evaluación.")
        return Pipeline([])

    subpipelines = []

    for key in model_keys:
        if "_" not in key:
            print(f"Formato inválido en el nombre del modelo: {key}. Se omite.")
            continue

        model_type, combo_id = key.split("_", 1)
        model_ds = f"models.{run_id}.{key}"
        output_ds = f"evaluation.{run_id}.{key}_output"
        tag = key

        # Construir nombre de archivo de salida para guardar resultados
        output_file_name = f"{model_type}_{combo_id}_{run_id}_output.pkl"

        # Guardar en el catálogo con nombre dinámico
        output_path = Path(f"data/06_models/{run_id}/{output_file_name}")

        if model_type == "LogisticAT":
            subpipelines.append(
                create_MORD_LogisticAT_pipeline(
                    param_key    = tag,
                    model_type   = model_type,
                    model_ds     = model_ds,
                    dataset_name = eval_ds_name,
                    output_ds    = output_ds,
                ).tag(tag)
            )

    if not subpipelines:
        print("No se procesaron subpipelines de evaluación debido a la falta de modelos válidos.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))