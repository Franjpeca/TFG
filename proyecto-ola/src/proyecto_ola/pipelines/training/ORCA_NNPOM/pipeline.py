from kedro.pipeline import Pipeline, node
from .nodes import ORCA_NNPOM

def create_pipeline(**kwargs) -> Pipeline:
    # Recuperamos los parámetros de kwargs
    params = kwargs.get("parameters", {})  # Obtenemos los parámetros pasados desde el orquestador

    # Aseguramos que el parámetro 'dataset_name' está disponible
    dataset_name = params.get("dataset_name", "default_dataset_name")

    # Crear el nombre dinámico basado en los parámetros
    run_id = params.get("run_id", "default")
    learning_rate = params.get("learning_rate", 0.01)
    hidden_layer_size = params.get("hidden_layer_size", 10)
    max_iter = params.get("max_iter", 1000)
    alpha = params.get("alpha", 0.1)
    
    # Nombre dinámico para el dataset
    dataset_name = f"ORCA_NNPOM_lr_{learning_rate}_hidden_size_{hidden_layer_size}_max_iter_{max_iter}_alpha_{alpha}_run_{run_id}".replace(".", "")

    return Pipeline(
        [
            node(
                func=ORCA_NNPOM,  # La función ORCA_NNPOM que devolverá el modelo vacío
                inputs=["train_ordinal", "parameters"],  # Pasamos los datos de entrenamiento y los parámetros al nodo
                outputs=dataset_name,  # Usamos el parámetro dinámico para la salida
                name=f"ORCA_NNPOM_node_lr_{learning_rate}_hidden_size_{hidden_layer_size}_max_iter_{max_iter}_alpha_{alpha}_run_{run_id}"
            )
        ]
    )