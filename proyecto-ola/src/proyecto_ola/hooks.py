import datetime
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from kedro_datasets.matplotlib import MatplotlibWriter
from matplotlib.figure import Figure

from proyecto_ola.utils.pipelines_utils import (
    find_latest_metrics_execution_folder,
    find_parameters_cli,
    parse_folders_param,
)

logger = logging.getLogger(__name__)

# Carpetas base (no cambiar)
MODELS_BASE = Path("data") / "03_models"
OUTPUT_BASE = Path("data") / "04_model_output"
METRICS_BASE = Path("data") / "05_model_metrics"
REPORT_BASE = Path("data") / "06_reporting"

# Soporta nombres con o sin segmento seed_<valor>
KEY_RE = re.compile(
    r'^(?P<model>[^_]+)_(?P<combo>grid_\d+)_(?P<dataset>\d+)'
    r'(?:_seed_(?P<seed>[^_]+))?_'
    r'(?P<train>gridsearch)_(?P<cv>cv_\d+_rs_\d+)$'
)

def is_evaluation(name: str) -> bool:
    return "evaluation" in str(name)

def is_visualization(name: str) -> bool:
    return "visualization" in str(name)

def load_parameters() -> Dict[str, Any]:
    return OmegaConfigLoader(conf_source="conf").get("parameters", {}) or {}

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

def register_if_missing(catalog: DataCatalog, key: str, dataset_obj) -> None:
    if key not in catalog.list():
        catalog.add(key, dataset_obj)

def set_param_if_changed(catalog: DataCatalog, param_key: str, value: Any) -> None:
    try:
        current = catalog.load(param_key)
    except Exception:
        current = object()
    if current != value:
        catalog.add_feed_dict({param_key: value}, replace=True)

def ensure_dirs(is_eval: bool, models_dir: Path, outputs_dir: Path, metrics_dir: Path) -> None:
    if not is_eval:
        models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

def resolve_execution_folder(pipeline_name: str, run_id: str, forced: Optional[str], cv_cfg: Dict[str, Any]) -> str:
    if forced:
        return forced
    if is_evaluation(pipeline_name):
        sig = f"cv_{cv_cfg.get('n_splits', 5)}_rs_{cv_cfg.get('random_state', 42)}"
        latest = find_latest_folder_with(MODELS_BASE, f"Model_*_{sig}.pkl")
        if latest:
            logger.info(f"[Evaluating] Using folder: {latest.name}")
            return latest.name
        latest_any = find_latest_folder_with(MODELS_BASE, "Model_*.pkl")
        if latest_any:
            logger.info(f"[Evaluating] Using folder: {latest_any.name}")
            return latest_any.name
        logger.warning(f"[Evaluating] No models found for run_id={run_id}.")
    return f"{run_id}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

def dataset_id_from_fullkey(full_key: str) -> str:
    m = KEY_RE.match(full_key)
    if m:
        return m.group("dataset")
    logger.warning(f"[evaluation] Could not extract dataset_id from {full_key}")
    return "unknown"

def dataset_id_from_train_name(name: str) -> str:
    return name.replace("cleaned_", "").replace("_train_ordinal", "")

def save_evaluated_keys(catalog: DataCatalog, keys: List[str]) -> None:
    register_if_missing(catalog, "evaluated_keys", MemoryDataset(copy_mode="assign"))
    catalog.save("evaluated_keys", keys)
    set_param_if_changed(catalog, "params:evaluated_keys", keys)

def register_model(catalog: DataCatalog, run_id: str, full_key: str, model_path: Path) -> None:
    ds = PickleDataset(filepath=str(model_path), backend="pickle", save_args={"protocol": 5})
    register_if_missing(catalog, f"training.{run_id}.Model_{full_key}", ds)
    register_if_missing(catalog, full_key, ds)

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

def visualization_output_path(output_name: str) -> Optional[Path]:
    parts = output_name.split(".")
    if len(parts) != 4 or parts[0] != "visualization":
        return None
    _, run_folder, dataset_id, metric = parts
    metric_l = metric.lower()
    base = REPORT_BASE / run_folder / dataset_id

    if metric_l == "heatmap":
        return base / "heatmap.png"

    if metric_l.startswith("scatter_"):
        return base / f"{metric_l}.png"

    sub = "ordinal" if metric_l in {"qwk", "mae", "amae"} else "nominal"
    return base / sub / f"{metric}.png"

def _choose_recent_folder(folders: List[str], base: Path) -> str:
    try:
        return max(folders, key=lambda f: (base / f).stat().st_mtime)
    except Exception:
        return folders[0]

class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        pipeline_name = run_params.get("pipeline_name", "")
        params = {**load_parameters(), **run_params.get("extra_params", {})}
        run_id = params.get("run_id", "001")
        forced_execution_folder = params.get("execution_folder")
        default_cv: Dict[str, Any] = params.get("cv_settings", {"n_splits": 5, "random_state": 42})
        training_settings: Dict[str, Any] = params.get("training_settings", {})
        seed_val = training_settings.get("seed", "unk")

        # --- Cabecera visible por ejecución ---
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("=" * 89)
        logger.info(f"NUEVA EJECUCIÓN | pipeline='{pipeline_name or 'default'}' | run_id='{run_id}' | timestamp={ts}")
        if forced_execution_folder:
            logger.info(f"[RunContext] execution_folder (forced)='{forced_execution_folder}'\n")
            logger.info("")

        # -------- VISUALIZATION MODE --------
        if is_visualization(pipeline_name):
            cli_multi = find_parameters_cli("execution_folders", params)
            exec_folders = parse_folders_param(cli_multi)

            if not exec_folders:
                cli_single = find_parameters_cli("execution_folder", params) or params.get("execution_folder")
                exec_folders = parse_folders_param(cli_single) if cli_single else []

            if not exec_folders:
                latest = find_latest_metrics_execution_folder(METRICS_BASE)
                if latest:
                    exec_folders = [latest]

            if not exec_folders:
                logger.warning("[VISUALIZATION] No valid metrics folder found.")
                return

            output_folder = _choose_recent_folder(exec_folders, METRICS_BASE)
            set_param_if_changed(catalog, "params:execution_folder", output_folder)

            all_metric_files = []
            for folder in exec_folders:
                metrics_root = METRICS_BASE / folder
                files = list(metrics_root.glob("Metrics_*.json"))
                for file in files:
                    key = f"evaluation.{folder}.{file.stem}"
                    register_if_missing(catalog, key, JSONDataset(filepath=str(file)))
                all_metric_files.extend(files)

            if not all_metric_files:
                logger.warning("[VISUALIZATION] No metric files found in provided folders.")
                return

            dataset_ids = set()
            for file in all_metric_files:
                full_key = file.stem.replace("Metrics_", "", 1)
                dsid = dataset_id_from_fullkey(full_key)
                if dsid != "unknown":
                    dataset_ids.add(dsid)

            nominal_metrics = params.get("nominal_metrics", ["accuracy", "f1_score"])
            ordinal_metrics = params.get("ordinal_metrics", ["qwk", "mae", "amae"])
            heatmap_metrics = params.get("heatmap_metrics", ["qwk", "mae", "amae", "f1_score", "accuracy"])

            def _ensure_writer(key: str):
                out_path = visualization_output_path(key)
                if out_path:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    register_if_missing(catalog, key, MatplotlibWriter(filepath=str(out_path)))

            for dsid in dataset_ids:
                for m in nominal_metrics:
                    _ensure_writer(f"visualization.{output_folder}.{dsid}.{m}")
                for m in ordinal_metrics:
                    _ensure_writer(f"visualization.{output_folder}.{dsid}.{m}")
                _ensure_writer(f"visualization.{output_folder}.{dsid}.heatmap")
                # Registrar ambos por compatibilidad: MAE y AMAE
                _ensure_writer(f"visualization.{output_folder}.{dsid}.scatter_qwk_mae")
                _ensure_writer(f"visualization.{output_folder}.{dsid}.scatter_qwk_amae")
            return

        # -------- TRAINING / EVALUATION --------
        execution_folder = resolve_execution_folder(pipeline_name, run_id, forced_execution_folder, default_cv)
        set_param_if_changed(catalog, "params:execution_folder", execution_folder)

        # Log del contexto de ejecución ya resuelto
        logger.info(f"[RunContext] execution_folder='{execution_folder}'")

        logger.info("=" * 89)
        logger.info("")

        models_dir = MODELS_BASE / execution_folder
        outputs_dir = OUTPUT_BASE / execution_folder
        metrics_dir = METRICS_BASE / execution_folder
        ensure_dirs(is_evaluation(pipeline_name), models_dir, outputs_dir, metrics_dir)

        model_parameters: Dict[str, Dict[str, Any]] = params.get("model_parameters", {})
        training_datasets: List[str] = params.get("training_datasets", [])

        if is_evaluation(pipeline_name):
            # Firma CV activa (sea 2, 3, 5... lo que venga en params)
            sig = f"cv_{default_cv.get('n_splits', 5)}_rs_{default_cv.get('random_state', 42)}"

            if forced_execution_folder:
                # Si fuerzas carpeta: no filtramos por CV, pero avisamos si la firma activa no está presente.
                items: List[Tuple[str, Path]] = [
                    (p.stem.replace("Model_", "", 1), p) for p in models_dir.glob("Model_*.pkl")
                ]
                if not items:
                    logger.warning("[evaluation] No se encontraron modelos en %s.", models_dir)
                    # --- añadido: propaga vacío y sal ---
                    save_evaluated_keys(catalog, [])
                    return
                else:
                    # Detecta qué firmas CV hay en la carpeta forzada y avisa si falta la activa.
                    present_sigs = set()
                    for _, p in items:
                        name = p.stem  # Model_<...>_cv_X_rs_Y
                        parts = name.split("_")
                        if len(parts) >= 4:
                            present_sigs.add("_".join(parts[-4:]))  # e.g. "cv_3_rs_32"
                    if sig not in present_sigs:
                        logger.warning(
                            "[evaluation] La carpeta forzada '%s' no contiene modelos con la firma activa %s. "
                            "Firmas encontradas: %s",
                            models_dir, sig, sorted(present_sigs) if present_sigs else "ninguna"
                        )
            else:
                # Si no fuerzas carpeta: filtra por la firma CV activa para evitar mismatches.
                items = [
                    (p.stem.replace("Model_", "", 1), p) for p in models_dir.glob(f"Model_*_{sig}.pkl")
                ]
                if not items:
                    logger.warning(
                        "[evaluation] En la ultima ejecucion no hay modelos con firma %s en %s. "
                        "Pasa --params \"execution_folder=<carpeta_con_modelos>\" o ajusta cv_settings.",
                        sig, models_dir
                    )
                    save_evaluated_keys(catalog, [])
                    return
        else:
            items: List[Tuple[str, Path]] = []
            # Claves esperadas del training actual (se guardarán en models_dir cuando termine el fit)
            for model_name, combos in model_parameters.items():
                for combo_id, cfg in combos.items():
                    if "param_grid" not in cfg:
                        continue
                    cv_cfg = cfg.get("cv_settings", default_cv)
                    cv_str = f"cv_{cv_cfg['n_splits']}_rs_{cv_cfg['random_state']}"
                    for train_ds in training_datasets:
                        dsid = dataset_id_from_train_name(train_ds)
                        full_key = f"{model_name}_{combo_id}_{dsid}_seed_{seed_val}_gridsearch_{cv_str}"
                        items.append((full_key, models_dir / f"Model_{full_key}.pkl"))

        evaluated_keys: List[str] = []
        for full_key, model_path in items:
            evaluated_keys.append(full_key)
            register_model(catalog, run_id, full_key, model_path)
            register_evaluation_outputs(
                catalog=catalog,
                run_id=run_id,
                execution_folder=execution_folder,
                full_key=full_key,
                outputs_dir=OUTPUT_BASE / execution_folder,
                metrics_dir=METRICS_BASE / execution_folder,
            )
            dsid = dataset_id_from_fullkey(full_key)
            register_if_missing(catalog, f"params:{full_key}_train_dataset_id", MemoryDataset(copy_mode="assign"))
            register_if_missing(catalog, f"params:{full_key}_dataset_id", MemoryDataset(copy_mode="assign"))
            catalog.save(f"params:{full_key}_train_dataset_id", dsid)
            catalog.save(f"params:{full_key}_dataset_id", dsid)

        save_evaluated_keys(catalog, evaluated_keys)

    @hook_impl
    def after_node_run(self, node, inputs, outputs, catalog: DataCatalog):
        for output_name, value in outputs.items():
            if not (isinstance(value, Figure) and output_name.startswith("visualization.")):
                continue
            try:
                catalog.save(output_name, value)
            except KeyError:
                path = visualization_output_path(output_name)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    value.savefig(str(path))
                else:
                    logger.warning(f"[visualization] Unexpected output name: {output_name}")
