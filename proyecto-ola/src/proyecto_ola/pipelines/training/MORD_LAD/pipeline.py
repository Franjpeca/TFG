from kedro.pipeline import Pipeline, node
from .nodes import MORD_LAD

def create_pipeline(**kwargs) -> Pipeline:
# Recuperamos los parámetros de kwargs
    params = kwargs.get("parameters", {})  # Obtenemos los parámetros pasados desde el orquestador
    run_id = params.get("run_id", "default")

    # Aseguramos que el parámetro 'dataset_name' esté disponible
    dataset_name = params.get("dataset_name", "default_dataset_name")

    # Establecemos los valores de los parámetros para el modelo 'LAD'
    alpha = params.get("alpha", 1.0)  # Valor por defecto para alpha
    max_iter = params.get("max_iter", 1000)  # Valor por defecto para max_iter

    # Crear el nombre dinámico basado en los parámetros
    dataset_name = f"LAD_alpha_{alpha}_max_iter_{max_iter}_run_{run_id}".replace(".", "")

    # Ahora construimos el pipeline con el nodo correspondiente
    return Pipeline(
        [
            node(
                func=MORD_LAD,  # Función LAD que devolverá el modelo vacío
                inputs=["train_ordinal", "parameters"],  # Pasamos los datos y los parámetros al nodo
                outputs=[dataset_name],  # Usar el parámetro dinámico para la salida
                name=f"MORD_LAD_node_{alpha}_{max_iter}_{run_id}"  # Nombre único para el nodo
            )
        ]
    )