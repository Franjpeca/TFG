import os
from kedro.framework.hooks import hook_impl
from kedro.config import OmegaConfigLoader
from kedro.io import DataCatalog, MemoryDataset
from kedro_datasets.pickle import PickleDataset
from kedro_datasets.json import JSONDataset
from pyspark import SparkConf
from pyspark.sql import SparkSession


class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:

        os.environ["SPARK_CONF_DIR"] = str(context.project_path / "conf" / "base")
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        spark = (
            SparkSession.builder
            .appName(context.project_path.name)
            .enableHiveSupport()
            .config(conf=spark_conf)
            .config(
                "spark.driver.extraJavaOptions",
                "-Dlog4j.configuration=file:"
                + str(context.project_path / "conf" / "base" / "log4j.properties"),
            )
            .config(
                "spark.executor.extraJavaOptions",
                "-Dlog4j.configuration=file:"
                + str(context.project_path / "conf" / "base" / "log4j.properties"),
            )
            .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")
        context.spark = spark


class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, catalog: DataCatalog) -> None:
        # --- 1 Cargar parametros base -----------------------------
        config_loader = OmegaConfigLoader(conf_source="conf")
        params = config_loader.get("parameters", {})

        # Merge con posibles extra_params (soporte CLI en futuro)
        extra = run_params.get("extra_params", {}) or {}
        if "run_id" in extra:
            params["run_id"] = extra["run_id"]

        for key, value in extra.items():
            if key.startswith("model_parameters."):
                d = params
                for part in key.split(".")[1:-1]:
                    d = d.setdefault(part, {})
                d[key.split(".")[-1]] = value
        # ------------------------------------------------------------

        run_id = params.get("run_id", "default")
        model_params = params.get("model_parameters", {})

        # --- 2 Crear carpetas si no existen -------------------------
        models_dir = os.path.join("data", "06_models", run_id)
        output_dir = os.path.join("data", "07_model_output", run_id)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # --- 3 Registrar datasets dinamicos --------------------------
        for model_name, combos in model_params.items():
            for combo_id, cfg in combos.items():
                # Detectar tipo de entrenamiento y generar sufijo
                if "param_grid" in cfg:
                    tipo = "param_grid"
                    hyper_str = "gridsearch"
                elif "hyperparams" in cfg:
                    tipo = "hyperparams"
                    hp = cfg.get("hyperparams", {}) or {}
                    hyper_str = "_".join(f"{k}-{v}" for k, v in sorted(hp.items())) if hp else "default"
                else:
                    continue

                # Configuracion de validacion cruzada (global o especifica)
                cv = cfg.get("cv_settings", params.get("cv_settings", {"n_splits": 5, "random_state": 42}))
                cv_str = f"cv{cv['n_splits']}_rs{cv['random_state']}"

                key = f"{model_name}_{combo_id}_{hyper_str}_{cv_str}"

                # --- Modelo entrenado (.pkl)
                model_ds_key = f"models.{run_id}.{key}"
                model_path = os.path.join(models_dir, f"{key}.pkl")

                if model_ds_key not in catalog.list():
                    catalog.add(model_ds_key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))
                if key not in catalog.list():
                    catalog.add(key, PickleDataset(filepath=model_path, save_args={"protocol": 4}))

                # --- Output de evaluacion (.json)
                output_name = f"{key}_output"
                output_ds_key = f"evaluation.{run_id}.{output_name}"
                output_path = os.path.join(output_dir, f"{output_name}.json")

                if output_ds_key not in catalog.list():
                    catalog.add(output_ds_key, JSONDataset(filepath=output_path))

                # --- Param type para el nodo
                param_type_key = f"params:model_parameters.{model_name}.{combo_id}.param_type"
                if param_type_key not in catalog.list():
                    catalog.add(param_type_key, MemoryDataset(data=tipo))

                # --- Alias por nombre del modelo
                if model_name not in catalog.list():
                    catalog.add(model_name, MemoryDataset(data=model_name))
