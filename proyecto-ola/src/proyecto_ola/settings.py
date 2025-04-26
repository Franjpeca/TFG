from kedro.config import OmegaConfigLoader
from proyecto_ola.hooks import SparkHooks, DynamicModelCatalogHook

LOGGING_CONFIG = "conf/base/logging.yml"

# Configuración del loader (ya la tenías correcta)
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark": ["spark*", "spark*/**"],
    },
}

# Aquí registramos **ambos** hooks, SIN duplicados
HOOKS = (
    SparkHooks(),
    DynamicModelCatalogHook(),
)