from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
from kedro.config import OmegaConfigLoader 
from kedro_datasets.pickle import PickleDataset
from kedro.io import MemoryDataset
from kedro_datasets.json import JSONDataset
import os
import json

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
                    "-Dlog4j.configuration=file:" + str(context.project_path / "conf" / "base" / "log4j.properties")
                )
                .config(
                    "spark.executor.extraJavaOptions",
                    "-Dlog4j.configuration=file:" + str(context.project_path / "conf" / "base" / "log4j.properties")
                )
                .getOrCreate()
        )

        spark.sparkContext.setLogLevel("ERROR")
        context.spark = spark

class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, pipeline, catalog) -> None:
        print("🔧 Hook: Cargando modelos detectados en disco")

        config_loader = OmegaConfigLoader(conf_source="conf")
        params = config_loader.get("parameters", {})
        extra = run_params.get("extra_params", {}) or {}

        if "run_id" in extra:
            params["run_id"] = extra["run_id"]

        run_id = params.get("run_id", "default")

        models_dir = os.path.join("data", "06_models", run_id)
        os.makedirs(models_dir, exist_ok=True)

        output_dir = os.path.join("data", "07_model_output", run_id)
        os.makedirs(output_dir, exist_ok=True)

        # 1) Registrar los modelos .pkl ya existentes (para cargar en evaluación)
        for file in os.listdir(models_dir):
            if not file.endswith(".pkl"):
                continue

            param_key = os.path.splitext(file)[0]          # Ej: LogisticAT_logAT_001_alpha-...
            ds_key    = f"models.{run_id}.{param_key}"
            fp        = os.path.join(models_dir, file)

            if ds_key not in catalog.list():
                catalog.add(
                    ds_key,
                    PickleDataset(filepath=fp, save_args={"protocol": 4})
                )
                print(f"✅ PickleDataset registrado → {ds_key}")

        # 2) Registrar datasets de salida para guardar nuevos modelos
        model_params = params.get("model_parameters", {})
        for model_name, combos in model_params.items():
            for combo_id, cfg in combos.items():
                hyperparams = cfg.get("hyperparams", {})
                hyper_str   = "_".join(f"{k}-{v}" for k, v in hyperparams.items()) if hyperparams else "default"

                key       = f"{model_name}_{combo_id}_{hyper_str}"
                ds_key    = f"models.{run_id}.{key}"
                filepath  = os.path.join(models_dir, f"{key}.pkl")

                if ds_key not in catalog.list():
                    catalog.add(
                        ds_key,
                        PickleDataset(filepath=filepath, save_args={"protocol": 4})
                    )
                    print(f"📝 Registrado output → {ds_key} → {filepath}")

                # Alias adicional por si el nodo usa solo 'key'
                if key not in catalog.list():
                    catalog.add(
                        key,
                        PickleDataset(filepath=filepath, save_args={"protocol": 4})
                    )
                    print(f"🔗 Alias registrado → {key} → {filepath}")

                # Registrar nombre del modelo como MemoryDataset (ej: 'LogisticAT')
                if model_name not in catalog.list():
                    catalog.add(model_name, MemoryDataset(data=model_name))
                    print(f"🧠 ModelType registrado → {model_name}")

        # 3) Registrar resultados de evaluación con sufijo _output
        for model_name, combos in model_params.items():
            for combo_id, cfg in combos.items():
                hyperparams = cfg.get("hyperparams", {})
                hyper_str = "_".join([f"{k}-{v}" for k, v in sorted(hyperparams.items())])  # ← aquí

                key = f"{model_name}_{combo_id}_{hyper_str}"
                output_name     = f"{key}_output"
                output_ds_key   = f"evaluation.{run_id}.{output_name}"
                output_filepath = os.path.join(output_dir, f"{output_name}.json")

                if output_ds_key not in catalog.list():
                    catalog.add(
                        output_ds_key,
                        JSONDataset(filepath=output_filepath)
                    )
                    print(f"✅ Output registrado como → {output_ds_key} → {output_filepath}")