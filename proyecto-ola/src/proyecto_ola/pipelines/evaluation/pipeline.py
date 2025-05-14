from pathlib import Path
from kedro.pipeline import Pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    from pathlib import Path
    from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline
    from kedro.pipeline import Pipeline

    # Variables "de entrada"
    params         = kwargs.get("params", {})
    run_id         = params.get("run_id", "debug")
    evaluate_only  = params.get("evaluate_only", None)
    test_datasets  = params.get("test_datasets", [])
    models_dir     = Path(f"data/06_models/{run_id}")

    model_files = sorted(models_dir.glob("*.pkl"))
    model_keys  = [f.stem for f in model_files]

    if evaluate_only:
        if isinstance(evaluate_only, str):
            evaluate_only = [evaluate_only]
        model_keys = [k for k in model_keys if k in evaluate_only]

    subpipelines = []

    for key in model_keys:
        parts = key.split("_")

        if len(parts) < 6:
            print(f"Formato inválido en el nombre del modelo: {key}. Se omite.")
            continue

        model_name  = parts[0]
        combo_id    = parts[1]
        dataset_id  = parts[2]
        # El resto no es necesario extraerlo

        test_ds_name = f"cleaned_{dataset_id}_test_ordinal"
        if test_datasets and test_ds_name not in test_datasets:
            continue

        model_ds = f"models.{run_id}.{key}"
        output_ds = f"evaluation.{run_id}.{key}_output"
        tag = key

        if model_name == "LogisticAT":
            subpipelines.append(
                create_MORD_LogisticAT_pipeline(
                    param_key    = tag,
                    model_type   = model_name,
                    model_ds     = model_ds,
                    dataset_name = test_ds_name,
                    output_ds    = output_ds,
                ).tag(tag)
            )

    if not subpipelines:
        print("No se procesaron subpipelines de evaluación debido a la falta de modelos válidos.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))