from pathlib import Path
from kedro.pipeline import Pipeline
from proyecto_ola.pipelines.evaluation.MORD_LogisticAT.pipeline import create_pipeline as create_MORD_LogisticAT_pipeline

def create_pipeline(**kwargs) -> Pipeline:
    params         = kwargs.get("params", {})
    run_id         = params.get("run_id", "debug")
    evaluate_only  = params.get("evaluate_only", None)
    test_datasets  = params.get("test_datasets", [])
    catalog        = kwargs.get("catalog", None)

    models_dir     = Path(f"data/06_models/{run_id}")
    model_files    = sorted(models_dir.glob("*.pkl"))
    model_keys     = [f.stem for f in model_files]

    if not model_keys:
        logger.warning("No se encontraron modelos en: %s", models_dir)

    if evaluate_only:
        if isinstance(evaluate_only, str):
            evaluate_only = [evaluate_only]
        model_keys = [k for k in model_keys if k in evaluate_only]

    subpipelines = []

    for key in model_keys:
        try:
            # Separar desde el final para encontrar los últimos 4 elementos
            prefix, cv_tag, cv_n, rs_tag, rs_n = key.rsplit("_", 4)
            # Ahora separo el resto: model_name, combo_id, combo_num, dataset_id, hyperparam_str...
            parts = prefix.split("_", 4)

            if len(parts) < 4:
                logger.warning("Nombre inválido: %s. Esperado al menos 4 elementos antes de _cv_...", key)
                continue

            model_name  = parts[0]
            combo_id    = parts[1]
            combo_num   = parts[2]
            dataset_id  = parts[3]

        except ValueError:
            logger.warning("No se pudo parsear el nombre del modelo: %s", key)
            continue

        test_ds_name = f"cleaned_{dataset_id}_test_ordinal"
        if test_datasets and test_ds_name not in test_datasets:
            continue

        if catalog and test_ds_name not in catalog.list():
            logger.warning("Dataset %s no está en el catálogo. Se omite.", test_ds_name)
            continue

        model_ds  = f"models.{run_id}.{key}"
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
                    dataset_id   = dataset_id
                ).tag(tag)
            )

    if not subpipelines:
        logger.info("No se procesaron subpipelines de evaluación por falta de modelos válidos.")
        return Pipeline([])

    return sum(subpipelines, Pipeline([]))