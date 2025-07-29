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
from matplotlib.figure import Figure

class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        pipeline_name = run_params.get("pipeline_name", "")
        if "visualization" in str(pipeline_name):
            base_dir = Path("data/08_model_metrics")
            if base_dir.exists() and any(base_dir.glob("*_*")):
                folders = sorted(base_dir.glob("*_*"), key=lambda p: p.stat().st_mtime)
                vis_folder = folders[-1].name
                catalog.add_feed_dict({"params:execution_folder": vis_folder}, replace=True)
                for f in (base_dir / vis_folder).glob("Metrics_*.json"):
                    ds_key = f"evaluation.{vis_folder}.{f.stem}"
                    if ds_key not in catalog.list():
                        catalog.add(ds_key, JSONDataset(filepath=str(f)))
            return

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_folder = f"{run_id}_{timestamp}"
        catalog.add_feed_dict({"params:execution_folder": execution_folder}, replace=True)

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
                hyper_str = "gridsearch"
                cv = cfg.get("cv_settings", default_cv)
                cv_str = f"cv_{cv['n_splits']}_rs_{cv['random_state']}"
                for train_ds in train_datasets:
                    dataset_id = train_ds.replace("cleaned_", "").replace("_train_ordinal", "")
                    full_key = f"{model_name}_{combo_id}_{dataset_id}_{hyper_str}_{cv_str}"
                    evaluated_keys.append(full_key)
                    model_ds_key = f"training.{run_id}.Model_{full_key}"
                    model_path = os.path.join(models_dir, f"Model_{full_key}.pkl")
                    if model_ds_key not in catalog.list():
                        catalog.add(model_ds_key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))
                    if full_key not in catalog.list():
                        catalog.add(full_key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))
                    pred_name = f"Predicted_Labels_{full_key}"
                    pred_ds_key = f"evaluation.{run_id}.{pred_name}"
                    pred_path = os.path.join(output_dir, f"{pred_name}.json")
                    if pred_ds_key not in catalog.list():
                        catalog.add(pred_ds_key, JSONDataset(filepath=pred_path))
                    metrics_name = f"Metrics_{full_key}"
                    metrics_ds_key = f"evaluation.{run_id}.{metrics_name}"
                    metrics_path = os.path.join(metrics_dir, f"{metrics_name}.json")
                    if metrics_ds_key not in catalog.list():
                        catalog.add(metrics_ds_key, JSONDataset(filepath=metrics_path))
                    param_type_key = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                    if param_type_key not in catalog.list():
                        catalog.add(param_type_key, MemoryDataset(data="param_grid", copy_mode="assign"))
                    if model_name not in catalog.list():
                        catalog.add(model_name, MemoryDataset(data=model_name, copy_mode="assign"))
                    td_param = f"params:{full_key}_train_dataset_id"
                    id_param = f"params:{full_key}_dataset_id"
                    if td_param not in catalog.list():
                        catalog.add(td_param, MemoryDataset(data=dataset_id))
                    if id_param not in catalog.list():
                        catalog.add(id_param, MemoryDataset(data=dataset_id))

        if "evaluated_keys" not in catalog.list():
            catalog.add("evaluated_keys", MemoryDataset(copy_mode="assign"))
        catalog._datasets["evaluated_keys"].data = evaluated_keys

    @hook_impl
    def after_node_run(self, node, inputs, outputs, catalog):
        from matplotlib.figure import Figure
        for output_name, value in outputs.items():
            if not isinstance(value, Figure):
                continue
            if not output_name.startswith("visualization."):
                continue
            parts = output_name.split(".")
            _, run_id, dataset_id, metric = parts
            metric_type = "ordinal" if metric in ["qwk", "mae", "amae"] else "nominal"
            d = os.path.join("data", "09_reporting", run_id, dataset_id, metric_type)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{metric}.png")
            if output_name not in catalog.list():
                catalog.add(output_name, MatplotlibWriter(filepath=p))
            catalog.save(output_name, value)