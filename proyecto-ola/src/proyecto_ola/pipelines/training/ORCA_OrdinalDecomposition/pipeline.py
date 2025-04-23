from kedro.pipeline import Pipeline, node
from .nodes import ORCA_OrdinalDecomposition

def create_pipeline(**kwargs) -> Pipeline:
    # Recuperamos los parámetros de kwargs
    params = kwargs.get("parameters", {})  # Obtenemos los parámetros pasados desde el orquestador

    # Aseguramos que el parámetro 'dataset_name' está disponible
    dataset_name = params.get("dataset_name", "default_dataset_name")

    # Crear el nombre dinámico basado en los parámetros
    run_id = params.get("run_id", "default")
    C = params.get("C", 1.0)
    epsilon = params.get("epsilon", 0.1)
    max_iter = params.get("max_iter", 1000)
    
    # Nombre dinámico para el dataset
    dataset_name = f"ORCA_OrdinalDecomposition_C_{C}_epsilon_{epsilon}_max_iter_{max_iter}_run_{run_id}".replace(".", "")

    return Pipeline(
        [
            node(
                func=ORCA_OrdinalDecomposition,  # La función ORCA_OrdinalDecomposition que devolverá el modelo vacío
                inputs=["train_ordinal", "parameters"],  # Pasamos los datos de entrenamiento y los parámetros al nodo
                outputs=dataset_name,  # Usamos el parámetro dinámico para la salida
                name=f"ORCA_OrdinalDecomposition_node_C_{C}_epsilon_{epsilon}_max_iter_{max_iter}_run_{run_id}"
            )
        ]
    )