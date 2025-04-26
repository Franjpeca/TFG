from kedro.framework.hooks import hook_impl
from pyspark import SparkConf
from pyspark.sql import SparkSession
from kedro.config import OmegaConfigLoader 
from kedro_datasets.pickle import PickleDataset
import os

class SparkHooks:
    @hook_impl
    def after_context_created(self, context) -> None:
        # 1) Indica a Spark dónde está log4j.properties
        os.environ["SPARK_CONF_DIR"] = str(context.project_path / "conf" / "base")

        # 2) Carga tu configuración de spark.yml
        parameters = context.config_loader["spark"]
        spark_conf = SparkConf().setAll(parameters.items())

        # 3) Añade las opciones JVM para driver y executor
        spark = (
            SparkSession.builder
                .appName(context.project_path.name)
                .enableHiveSupport()
                .config(conf=spark_conf)
                .config(
                    "spark.driver.extraJavaOptions",
                    "-Dlog4j.configuration=file:"
                    + str(context.project_path / "conf" / "base" / "log4j.properties")
                )
                .config(
                    "spark.executor.extraJavaOptions",
                    "-Dlog4j.configuration=file:"
                    + str(context.project_path / "conf" / "base" / "log4j.properties")
                )
                .getOrCreate()
        )

        # 4) Opcionalmente controla el nivel Python-side
        spark.sparkContext.setLogLevel("ERROR")
        context.spark = spark

class DynamicModelCatalogHook:
    @hook_impl
    def before_pipeline_run(self, run_params: dict, pipeline, catalog) -> None:
        """
        Inyecta en el catálogo un PickleDataset para cada modelo/combo
        justo antes de ejecutar el pipeline.
        """
        # 1) Carga la configuración completa de parameters.yml
        config_loader = OmegaConfigLoader(conf_source="conf")
        params = config_loader.get("parameters", {})

        # 2) Sobrescribe con los extra_params pasados por CLI
        extra = run_params.get("extra_params", {}) or {}
        # run_id es top-level, así que update directo vale
        if "run_id" in extra:
            params["run_id"] = extra["run_id"]
        # no necesitamos sobrescribir model_parameters, usar los del YAML

        run_id = params.get("run_id", "default")
        model_params = params.get("model_parameters", {})

        # 3) Para cada modelo y combo, añade un dataset dinámico si no existe
        for model_name, combos in model_params.items():
            for combo_id in combos:
                ds_key = f"models.{run_id}.{model_name}_{combo_id}"
                fp     = os.path.join("data", "06_models", run_id, f"{model_name}_{combo_id}.pkl")

                if ds_key not in catalog.list():
                    catalog.add(
                        ds_key,
                        PickleDataset(filepath=fp, save_args={"protocol": 4})
                    )
                    print(f"📦 Dataset dinámico añadido → {ds_key} → {fp}")