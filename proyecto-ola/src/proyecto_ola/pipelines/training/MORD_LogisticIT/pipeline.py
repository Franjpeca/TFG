from kedro.pipeline import Pipeline, node
from .nodes import LogisticIT

def create_pipeline(**kwargs) -> Pipeline:
    # Recuperamos los parámetros de kwargs
    params = kwargs.get("parameters", {})  # Obtenemos los parámetros pasados desde el orquestador

    # Aseguramos que el parámetro 'dataset_name' está disponible
    dataset_name = params.get("dataset_name", "default_dataset_name")
    print("Contenido de params:", params)
    # Crear el nombre dinámico basado en los parámetros
    run_id = params.get("run_id", "default")
    alpha = params.get("alpha", 0.1)
    max_iter = params.get("max_iter", 1000)
    tol = params.get("tol", 1e-4)
    
    # Nombre dinámico para el dataset
    dataset_name = f"LogisticIT_alpha_{alpha}_max_iter_{max_iter}_tol_{tol}_run_{run_id}".replace(".", "")

    return Pipeline(
        [
            node(
                func=LogisticIT,  # La función LogisticIT que devolverá el modelo vacío
                inputs=["train_ordinal", "parameters"],  # Pasamos los datos de entrenamiento y los parámetros al nodo
                outputs=dataset_name,  # Usamos el parámetro dinámico para la salida
                name=f"LogisticIT_node_alpha_{alpha}_max_iter_{max_iter}_tol_{tol}_run_{run_id}"
            )
        ]
    )