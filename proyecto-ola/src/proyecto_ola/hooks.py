import os
import datetime
from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from kedro_datasets.matplotlib import MatplotlibWriter

class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        config_loader = OmegaConfigLoader(conf_source="conf")
        params = config_loader.get("parameters", {})

        extra = run_params.get("extra_params", {}) or {}
        if "run_id" in extra:
            params["run_id"] = extra["run_id"]

        for key, value in extra.items():
            if key.startswith("model_parameters."):
                d = params
                for part in key.split(".")[1:-1]:
                    d = d.setdefault(part, {})
                d[key.split(".")[-1]] = value

        run_id = params.get("run_id", "default")
        model_params = params.get("model_parameters", {})
        train_datasets = params.get("training_datasets", [])
        default_cv = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

        # Generamos un timestamp para identificar la ejecución
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_folder = f"{run_id}_{timestamp}"
        models_dir = os.path.join("data", "06_models", execution_folder)
        output_dir = os.path.join("data", "07_model_output", execution_folder)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        evaluated_keys = []

        for model_name, combos in model_params.items():
            for combo_id, cfg in combos.items():
                if "param_grid" not in cfg:
                    continue

                tipo = "param_grid"
                hyper_str = "gridsearch"
                cv = cfg.get("cv_settings", default_cv)
                cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"

                for train_ds in train_datasets:
                    dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                    full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"
                    evaluated_keys.append(full_key)

                    # Registro del modelo
                    model_ds_key = f"training.{run_id}.Model_{full_key}"
                    model_path = os.path.join(models_dir, f"Model_{full_key}.pkl")
                    if model_ds_key not in catalog.list():
                        catalog.add(model_ds_key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))
                    if full_key not in catalog.list():
                        catalog.add(full_key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))

                    # Registro de metricas
                    output_name = f"Metrics_{full_key}"
                    output_ds_key = f"evaluation.{run_id}.{output_name}"
                    output_path = os.path.join(output_dir, f"{output_name}.json")
                    if output_ds_key not in catalog.list():
                        catalog.add(output_ds_key, JSONDataset(filepath=output_path))

                    # Param type
                    param_type_key = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                    if param_type_key not in catalog.list():
                        catalog.add(param_type_key, MemoryDataset(data=tipo, copy_mode="assign"))

                    # Nombre del modelo
                    if model_name not in catalog.list():
                        catalog.add(model_name, MemoryDataset(data=model_name, copy_mode="assign"))

                    # Parametros dinamicos por clave
                    train_dataset_id_param = f"params:{full_key}_train_dataset_id"
                    dataset_id_param = f"params:{full_key}_dataset_id"
                    if train_dataset_id_param not in catalog.list():
                        catalog.add(train_dataset_id_param, MemoryDataset(data=dataset_id))
                    if dataset_id_param not in catalog.list():
                        catalog.add(dataset_id_param, MemoryDataset(data=dataset_id))

                    # Visualizaciones
                    for stage in ["overview", "distributions", "correlations"]:
                        output_viz_key = f"visualization.{run_id}.{full_key}_{stage}"
                        output_viz_path = os.path.join("data", "08_reporting", run_id, dataset_id, stage, f"{full_key}.png")
                        if output_viz_key not in catalog.list():
                            catalog.add(output_viz_key, MatplotlibWriter(filepath=output_viz_path))

        # Añadir y rellenar el dataset de evaluated_keys (Por testear)
        if "evaluated_keys" not in catalog.list():
            catalog.add("evaluated_keys", MemoryDataset(copy_mode="assign"))

        catalog._datasets["evaluated_keys"].data = evaluated_keys