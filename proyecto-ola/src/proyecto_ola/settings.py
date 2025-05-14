from kedro.config import OmegaConfigLoader
from proyecto_ola.hooks import DynamicModelCatalogHook#, SparkHooks 

LOGGING_CONFIG = "conf/base/logging.yml"

# Configuración del loader
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    },
}

# Registro de hooks
HOOKS = (
    #SparkHooks(),
    DynamicModelCatalogHook(),
)