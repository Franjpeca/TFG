import os
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List

from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from kedro_datasets.matplotlib import MatplotlibWriter
from matplotlib.figure import Figure
from typing import Optional
logger = logging.getLogger(__name__)

MODELS_BASE = Path("data") / "04_models"
OUTPUT_BASE = Path("data") / "05_model_output"
METRICS_BASE = Path("data") / "06_model_metrics"
REPORT_BASE = Path("data") / "07_reporting"


class DynamicModelCatalogHook:
    def _is_eval(self, pipeline_name: str) -> bool:
        return "evaluation" in str(pipeline_name)

    def _is_viz(self, pipeline_name: str) -> bool:
        return "visualization" in str(pipeline_name)

    def _load_params(self) -> Dict[str, Any]:
        cfg = OmegaConfigLoader(conf_source="conf")
        return cfg.get("parameters", {}) or {}

    def _choose_exec_folder_last_with_models(self, run_id: str, is_eval: bool) -> str:
        if is_eval:
            latest = self._latest_folder_by_inner_files(MODELS_BASE, "Model_*.pkl")
            if latest:
                logger.info(f"[Evaluating] Usando carpeta (última por ficheros): {latest.name}")
                return latest.name
            logger.warning(f"[Evaluating] No se encontraron modelos para run_id={run_id}.")
            return f"{run_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            return f"{run_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _ensure_dirs(self, is_eval: bool, models_dir: Path, output_dir: Path, metrics_dir: Path) -> None:
        if not is_eval:
            models_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

    def _register_if_missing(self, catalog: DataCatalog, key: str, ds) -> None:
        if key not in catalog.list():
            catalog.add(key, ds)

    def _latest_folder_by_inner_files(self, base: Path, pattern: str) -> Optional[Path]:
        if not base.exists():
            return None
        best_dir, best_ts = None, -1.0
        for d in base.glob("*_*"):
            files = list(d.glob(pattern))
            if not files:
                continue
            ts = max(f.stat().st_mtime for f in files)
            if ts > best_ts:
                best_ts, best_dir = ts, d
        return best_dir

    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        pipeline_name = run_params.get("pipeline_name", "")

        params_file = self._load_params()
        params_cli = run_params.get("extra_params", {})
        params = {**params_file, **params_cli}

        run_id = params.get("run_id", "001")
        forced_folder = params.get("execution_folder")
        is_eval = self._is_eval(pipeline_name)

        # VISUALIZATION
        if self._is_viz(pipeline_name):
            base_dir = METRICS_BASE
            vis_folder = forced_folder or (self._latest_folder_by_inner_files(base_dir, "Metrics_*.json") or Path(""))
            if vis_folder:
                vis_folder = Path(vis_folder).name
                if "params:execution_folder" not in catalog.list():
                    catalog.add_feed_dict({"params:execution_folder": vis_folder}, replace=True)
                for f in (METRICS_BASE / vis_folder).glob("Metrics_*.json"):
                    ds_key = f"evaluation.{vis_folder}.{f.stem}"
                    self._register_if_missing(catalog, ds_key, JSONDataset(filepath=str(f)))
            else:
                logger.warning("[VISUALIZATION] No se encontró ninguna carpeta de métricas válida.")
            return

        # EVALUATION o TRAINING
        execution_folder = forced_folder or self._choose_exec_folder_last_with_models(run_id, is_eval)
        if "params:execution_folder" not in catalog.list():
            catalog.add_feed_dict({"params:execution_folder": execution_folder}, replace=True)

        model_params = params.get("model_parameters", {})
        train_datasets = params.get("training_datasets", [])
        default_cv = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

        models_dir = MODELS_BASE / execution_folder
        output_dir = OUTPUT_BASE / execution_folder
        metrics_dir = METRICS_BASE / execution_folder
        self._ensure_dirs(is_eval, models_dir, output_dir, metrics_dir)

        evaluated_keys: List[str] = []

        if is_eval:
            for pkl in models_dir.glob("Model_*.pkl"):
                full_key = pkl.stem.replace("Model_", "", 1)
                evaluated_keys.append(full_key)

                ds_model = PickleDataset(filepath=str(pkl), backend="pickle", save_args={"protocol": 5})
                model_ds_key = f"training.{run_id}.Model_{full_key}"
                self._register_if_missing(catalog, model_ds_key, ds_model)
                self._register_if_missing(catalog, full_key, ds_model)

                pred_name = f"Predicted_Labels_{full_key}"
                metr_name = f"Metrics_{full_key}"
                self._register_if_missing(
                    catalog,
                    f"evaluation.{run_id}.{pred_name}",
                    JSONDataset(filepath=str(output_dir / f"{pred_name}.json")),
                )
                self._register_if_missing(
                    catalog,
                    f"evaluation.{run_id}.{metr_name}",
                    JSONDataset(filepath=str(metrics_dir / f"{metr_name}.json")),
                )

                try:
                    _, _, dataset_id, _, _ = full_key.split("_", 4)
                except ValueError:
                    dataset_id = "unknown"
                    logger.warning(f"[evaluation] No se pudo extraer dataset_id de {full_key}")

                self._register_if_missing(catalog, f"params:{full_key}_train_dataset_id", MemoryDataset(data=dataset_id))
                self._register_if_missing(catalog, f"params:{full_key}_dataset_id", MemoryDataset(data=dataset_id))

            if "evaluated_keys" not in catalog.list():
                catalog.add("evaluated_keys", MemoryDataset(copy_mode="assign"))
            catalog.save("evaluated_keys", evaluated_keys)
            catalog.add_feed_dict({"params:evaluated_keys": evaluated_keys}, replace=True)
            return

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
                    model_path = models_dir / f"Model_{full_key}.pkl"
                    evaluated_keys.append(full_key)

                    ds_model = PickleDataset(filepath=str(model_path), backend="pickle", save_args={"protocol": 5})
                    model_ds_key = f"training.{run_id}.Model_{full_key}"
                    self._register_if_missing(catalog, model_ds_key, ds_model)
                    self._register_if_missing(catalog, full_key, ds_model)

                    pred_name = f"Predicted_Labels_{full_key}"
                    pred_ds_key = f"evaluation.{run_id}.{pred_name}"
                    pred_path = output_dir / f"{pred_name}.json"
                    self._register_if_missing(catalog, pred_ds_key, JSONDataset(filepath=str(pred_path)))

                    metrics_name = f"Metrics_{full_key}"
                    metrics_ds_key = f"evaluation.{run_id}.{metrics_name}"
                    metrics_path = metrics_dir / f"{metrics_name}.json"
                    self._register_if_missing(catalog, metrics_ds_key, JSONDataset(filepath=str(metrics_path)))

                    param_type_key = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                    self._register_if_missing(catalog, param_type_key, MemoryDataset(data="param_grid", copy_mode="assign"))
                    self._register_if_missing(catalog, model_name, MemoryDataset(data=model_name, copy_mode="assign"))

                    td_param = f"params:{full_key}_train_dataset_id"
                    id_param = f"params:{full_key}_dataset_id"
                    self._register_if_missing(catalog, td_param, MemoryDataset(data=dataset_id))
                    self._register_if_missing(catalog, id_param, MemoryDataset(data=dataset_id))

        if "evaluated_keys" not in catalog.list():
            catalog.add("evaluated_keys", MemoryDataset(copy_mode="assign"))
        catalog.save("evaluated_keys", evaluated_keys)
        catalog.add_feed_dict({"params:evaluated_keys": evaluated_keys}, replace=True)

    @hook_impl
    def after_node_run(self, node, inputs, outputs, catalog: DataCatalog):
        for output_name, value in outputs.items():
            if not isinstance(value, Figure):
                continue
            if not output_name.startswith("visualization."):
                continue

            try:
                _, run_id, dataset_id, metric = output_name.split(".")
            except ValueError:
                logger.warning(f"[viz] Nombre de output inesperado: {output_name}")
                continue

            metric_type = "ordinal" if metric in {"qwk", "mae", "amae"} else "nominal"
            out_dir = REPORT_BASE / run_id / dataset_id / metric_type
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / f"{metric}.png"
            if output_name not in catalog.list():
                catalog.add(output_name, MatplotlibWriter(filepath=str(out_path)))
            catalog.save(output_name, value)
