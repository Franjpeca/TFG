from kedro.pipeline import Pipeline, node
from .nodes import MORD_MulticlassLogistic

def create_pipeline(**kwargs) -> Pipeline:
    # Recuperamos los parámetros de kwargs
    params = kwargs.get("parameters", {})  # Obtenemos los parámetros pasados desde el orquestador

    # Aseguramos que el parámetro 'dataset_name' está disponible
    dataset_name = params.get("dataset_name", "default_dataset_name")

    # Crear el nombre dinámico basado en los parámetros
    run_id = params.get("run_id", "default")
    alpha = params.get("alpha", 0.1)
    max_iter = params.get("max_iter", 1000)
    tol = params.get("tol", 1e-4)
    
    # Nombre dinámico para el dataset
    dataset_name = f"MORD_MulticlassLogistic_alpha_{alpha}_max_iter_{max_iter}_tol_{tol}_run_{run_id}".replace(".", "")

    return Pipeline(
        [
            node(
                func=MORD_MulticlassLogistic,  # La función MORD_MulticlassLogistic que devolverá el modelo vacío
                inputs=["train_ordinal", "parameters"],  # Pasamos los datos de entrenamiento y los parámetros al nodo
                outputs=dataset_name,  # Usamos el parámetro dinámico para la salida
                name=f"MORD_MulticlassLogistic_node_{run_id}_alpha_{alpha}_max_iter_{max_iter}_tol_{tol}_run_{run_id}"
            )
        ]
    )