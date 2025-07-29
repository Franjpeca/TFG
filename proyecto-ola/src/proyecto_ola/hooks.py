import os
import datetime
from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from kedro_datasets.matplotlib import MatplotlibWriter

from kedro_datasets.json import JSONDataset
from pathlib import Path

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

        catalog.add_feed_dict(
            {"params:execution_folder": execution_folder},
            replace=True,
)

        models_dir = os.path.join("data", "06_models", execution_folder)
        output_dir = os.path.join("data", "07_model_output", execution_folder)
        metrics_dir = os.path.join("data", "08_model_metrics", execution_folder)
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

                    # Registro de predicciones
                    pred_name = f"Predicted_Labels_{full_key}"
                    pred_ds_key = f"evaluation.{run_id}.{pred_name}"
                    pred_path = os.path.join(output_dir, f"{pred_name}.json")
                    if pred_ds_key not in catalog.list():
                        catalog.add(pred_ds_key, JSONDataset(filepath=pred_path))

                    # Registro de métricas
                    metrics_name = f"Metrics_{full_key}"
                    metrics_ds_key = f"evaluation.{run_id}.{metrics_name}"
                    metrics_path = os.path.join(metrics_dir, f"{metrics_name}.json")
                    if metrics_ds_key not in catalog.list():
                        catalog.add(metrics_ds_key, JSONDataset(filepath=metrics_path))

                    # Param type
                    param_type_key = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                    if param_type_key not in catalog.list():
                        catalog.add(param_type_key, MemoryDataset(data=tipo, copy_mode="assign"))

                    # Nombre del modelo
                    if model_name not in catalog.list():
                        catalog.add(model_name, MemoryDataset(data=model_name, copy_mode="assign"))

                    # Parámetros dinámicos por clave
                    train_dataset_id_param = f"params:{full_key}_train_dataset_id"
                    dataset_id_param = f"params:{full_key}_dataset_id"
                    if train_dataset_id_param not in catalog.list():
                        catalog.add(train_dataset_id_param, MemoryDataset(data=dataset_id))
                    if dataset_id_param not in catalog.list():
                        catalog.add(dataset_id_param, MemoryDataset(data=dataset_id))


        # Añadir y rellenar el dataset de evaluated_keys (por testear)
        if "evaluated_keys" not in catalog.list():
            catalog.add("evaluated_keys", MemoryDataset(copy_mode="assign"))
        catalog._datasets["evaluated_keys"].data = evaluated_keys

        # Parte de visualization
        if "visualization" in str(run_params.get("pipeline_name", "")):

            base_dir = Path("data/08_model_metrics")

            # Se busca la carpeta a usar co nlas metricas
            if run_id and (base_dir / run_id).is_dir():
                # Si se pasa el nombre mas reciente
                vis_folder = run_id
            elif run_id and any(base_dir.glob(f"{run_id}_*")):
                # Si pasa solo el prefijo se usa la mas reciente con ese
                vis_folder = sorted(
                    base_dir.glob(f"{run_id}_*"),
                    key=lambda p: p.stat().st_mtime
                )[-1].name
            else:
                # Si no se pasa nada usa la mas reciente
                folders = sorted(
                    base_dir.glob("*_*"),
                    key=lambda p: p.stat().st_mtime
                )
                if not folders:
                    raise FileNotFoundError(f"No hay métricas en {base_dir}. Ejecuta training/evaluation primero.")
                vis_folder = folders[-1].name

            # Registrar cada Metrics_*.json en el DataCatalog
            metrics_dir = base_dir / vis_folder
            for f in metrics_dir.glob("Metrics_*.json"):
                ds_key = f"evaluation.{vis_folder}.{f.stem}"
                if ds_key not in catalog.list():
                    catalog.add(ds_key, JSONDataset(filepath=str(f)))

            # Se lo permite pasar al pipeline (params:execution_folder)
            catalog.add_feed_dict({"params:execution_folder": vis_folder}, replace=True)

            print(f"[HOOK] Visualization ► usando carpeta '{vis_folder}' "
                f"y registrados {len(list(metrics_dir.glob('Metrics_*.json')))} datasets.")