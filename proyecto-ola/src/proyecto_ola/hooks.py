import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from kedro_datasets.matplotlib import MatplotlibWriter
from matplotlib.figure import Figure

from proyecto_ola.utils.pipelines_utils import find_latest_metrics_execution_folder
logger = logging.getLogger(__name__)

# Carpetas base (no cambiar)
MODELS_BASE = Path("data") / "04_models"
OUTPUT_BASE = Path("data") / "05_model_output"
METRICS_BASE = Path("data") / "06_model_metrics"
REPORT_BASE = Path("data") / "07_reporting"


# Comprueba si es pipeline de evaluacion
def is_evaluation(name: str) -> bool:
    return "evaluation" in str(name)


# Comprueba si es pipeline de visualizacion
def is_visualization(name: str) -> bool:
    return "visualization" in str(name)


# Carga parametros desde conf/
def load_parameters() -> Dict[str, Any]:
    return OmegaConfigLoader(conf_source="conf").get("parameters", {}) or {}


# Devuelve la ultima carpeta con ficheros que casen con el patron
def find_latest_folder_with(base: Path, pattern: str) -> Optional[Path]:
    if not base.exists():
        return None
    best, ts = None, -1.0
    for folder in base.glob("*_*"):
        files = list(folder.glob(pattern))
        if not files:
            continue
        latest_mtime = max(f.stat().st_mtime for f in files)
        if latest_mtime > ts:
            ts, best = latest_mtime, folder
    return best


# Registra dataset si falta
def register_if_missing(catalog: DataCatalog, key: str, dataset_obj) -> None:
    if key not in catalog.list():
        catalog.add(key, dataset_obj)


# Setea params:* solo si cambia
def set_param_if_changed(catalog: DataCatalog, param_key: str, value: Any) -> None:
    try:
        current = catalog.load(param_key)
    except Exception:
        current = object()
    if current != value:
        catalog.add_feed_dict({param_key: value}, replace=True)


# Asegura carpetas de salida
def ensure_dirs(is_eval: bool, models_dir: Path, outputs_dir: Path, metrics_dir: Path) -> None:
    if not is_eval:
        models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


# Resuelve carpeta de ejecucion
def resolve_execution_folder(pipeline_name: str, run_id: str, forced: Optional[str]) -> str:
    if forced:
        return forced
    if is_evaluation(pipeline_name) and pipeline_name == "evaluation":
        latest = find_latest_folder_with(MODELS_BASE, "Model_*.pkl")
        if latest:
            logger.info(f"[Evaluating] Using folder: {latest.name}")
            return latest.name
        logger.warning(f"[Evaluating] No models found for run_id={run_id}.")
    return f"{run_id}_{datetime.datetime.now():%Y%m%d_%H%M%S}"


# Extrae dataset_id desde la clave completa
def dataset_id_from_fullkey(full_key: str) -> str:
    try:
        return full_key.split("_", 4)[2]
    except Exception:
        logger.warning(f"[evaluation] Could not extract dataset_id from {full_key}")
        return "unknown"


# Extrae dataset_id desde el nombre del dataset de training
def dataset_id_from_train_name(name: str) -> str:
    return name.replace("cleaned_", "").replace("_train_ordinal", "")


# Guarda evaluated_keys sin warnings
def save_evaluated_keys(catalog: DataCatalog, keys: List[str]) -> None:
    register_if_missing(catalog, "evaluated_keys", MemoryDataset(copy_mode="assign"))
    catalog.save("evaluated_keys", keys)
    set_param_if_changed(catalog, "params:evaluated_keys", keys)


# Registra el modelo con clave oficial y alias
def register_model(catalog: DataCatalog, run_id: str, full_key: str, model_path: Path) -> None:
    ds = PickleDataset(filepath=str(model_path), backend="pickle", save_args={"protocol": 5})
    register_if_missing(catalog, f"training.{run_id}.Model_{full_key}", ds)
    register_if_missing(catalog, full_key, ds)


# Registra JSON de predicciones y metricas
def register_evaluation_outputs(
    catalog: DataCatalog,
    run_id: str,
    execution_folder: str,
    full_key: str,
    outputs_dir: Path,
    metrics_dir: Path,
) -> None:
    pred_name = f"Predicted_Labels_{full_key}"
    metric_name = f"Metrics_{full_key}"
    pred_path = outputs_dir / f"{pred_name}.json"
    metric_path = metrics_dir / f"{metric_name}.json"
    pairs: Tuple[Tuple[str, Path], ...] = (
        (f"evaluation.{run_id}.{pred_name}", pred_path),
        (f"evaluation.{run_id}.{metric_name}", metric_path),
        (f"evaluation.{execution_folder}.{pred_name}", pred_path),
        (f"evaluation.{execution_folder}.{metric_name}", metric_path),
    )
    for key, path in pairs:
        register_if_missing(catalog, key, JSONDataset(filepath=str(path)))


# Calcula la ruta de salida para figuras de visualizacion
def visualization_output_path(output_name: str) -> Optional[Path]:
    parts = output_name.split(".")
    if len(parts) != 4 or parts[0] != "visualization":
        return None
    _, run_folder, dataset_id, metric = parts
    metric_l = metric.lower()
    base = REPORT_BASE / run_folder / dataset_id
    if metric_l == "heatmap":
        return base / "heatmap.png"
    if metric_l == "scatter_qwk_mae":
        return base / "scatter_qwk_mae.png"
    sub = "ordinal" if metric_l in {"qwk", "mae", "amae"} else "nominal"
    return base / sub / f"{metric}.png"


class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        # Resuelve parametros base
        pipeline_name = run_params.get("pipeline_name", "")
        params = {**load_parameters(), **run_params.get("extra_params", {})}
        run_id = params.get("run_id", "001")
        forced_execution_folder = params.get("execution_folder")

        # Modo visualizacion: registra entradas JSON
        if is_visualization(pipeline_name):
            visualization_folder = forced_execution_folder or find_latest_metrics_execution_folder(METRICS_BASE)
            #logger.info(f"[HOOK] Usando carpeta de ejecucion: {visualization_folder}")
            if not visualization_folder:
                logger.warning("[VISUALIZATION] No valid metrics folder found.")
                return
            set_param_if_changed(catalog, "params:execution_folder", Path(visualization_folder).name)
            metrics_root = METRICS_BASE / Path(visualization_folder).name
            for file in metrics_root.glob("Metrics_*.json"):
                key = f"evaluation.{Path(visualization_folder).name}.{file.stem}"
                register_if_missing(catalog, key, JSONDataset(filepath=str(file)))
            return

        # Modo training/evaluation: resuelve carpeta de ejecucion
        execution_folder = resolve_execution_folder(pipeline_name, run_id, forced_execution_folder)
        set_param_if_changed(catalog, "params:execution_folder", execution_folder)

        # Prepara directorios
        models_dir = MODELS_BASE / execution_folder
        outputs_dir = OUTPUT_BASE / execution_folder
        metrics_dir = METRICS_BASE / execution_folder
        ensure_dirs(is_evaluation(pipeline_name), models_dir, outputs_dir, metrics_dir)

        # Lee parametros de modelos/datasets/cv
        model_parameters: Dict[str, Dict[str, Any]] = params.get("model_parameters", {})
        training_datasets: List[str] = params.get("training_datasets", [])
        default_cv: Dict[str, Any] = params.get("cv_settings", {"n_splits": 5, "random_state": 42})

        # Construye items a registrar
        if is_evaluation(pipeline_name):
            items: List[Tuple[str, Path]] = [
                (p.stem.replace("Model_", "", 1), p) for p in models_dir.glob("Model_*.pkl")
            ]
        else:
            items = []
            for model_name, combos in model_parameters.items():
                for combo_id, cfg in combos.items():
                    if "param_grid" not in cfg:
                        continue
                    cv_cfg = cfg.get("cv_settings", default_cv)
                    cv_str = f"cv_{cv_cfg['n_splits']}_rs_{cv_cfg['random_state']}"
                    for train_ds in training_datasets:
                        dsid = dataset_id_from_train_name(train_ds)
                        full_key = f"{model_name}_{combo_id}_{dsid}_gridsearch_{cv_str}"
                        items.append((full_key, models_dir / f"Model_{full_key}.pkl"))

        # Registra modelos, salidas y params por modelo
        evaluated_keys: List[str] = []
        for full_key, model_path in items:
            evaluated_keys.append(full_key)
            register_model(catalog, run_id, full_key, model_path)
            register_evaluation_outputs(
                catalog=catalog,
                run_id=run_id,
                execution_folder=execution_folder,
                full_key=full_key,
                outputs_dir=outputs_dir,
                metrics_dir=metrics_dir,
            )
            dsid = dataset_id_from_fullkey(full_key)
            register_if_missing(catalog, f"params:{full_key}_train_dataset_id", MemoryDataset(copy_mode="assign"))
            register_if_missing(catalog, f"params:{full_key}_dataset_id", MemoryDataset(copy_mode="assign"))
            catalog.save(f"params:{full_key}_train_dataset_id", dsid)
            catalog.save(f"params:{full_key}_dataset_id", dsid)

        # Guarda evaluated_keys
        save_evaluated_keys(catalog, evaluated_keys)

    @hook_impl
    def after_node_run(self, node, inputs, outputs, catalog: DataCatalog):
        # Persiste figuras de visualizacion
        for output_name, value in outputs.items():
            if not (isinstance(value, Figure) and output_name.startswith("visualization.")):
                continue
            path = visualization_output_path(output_name)
            if not path:
                logger.warning(f"[visualization] Unexpected output name: {output_name}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if output_name not in catalog.list():
                register_if_missing(catalog, output_name, MatplotlibWriter(filepath=str(path)))
            catalog.save(output_name, value)