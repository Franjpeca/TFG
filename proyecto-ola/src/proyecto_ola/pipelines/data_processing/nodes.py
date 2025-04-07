import pandas as pd

def clean_data_all(
    # Aceptamos cada dataset como parámetro explícito
    dataset_46014_train_ordinal,
    dataset_46014_test_ordinal,
    dataset_46025_train_ordinal,
    dataset_46025_test_ordinal,
    dataset_46042_train_ordinal,
    dataset_46042_test_ordinal,
    dataset_46053_train_ordinal,
    dataset_46053_test_ordinal,
    dataset_46069_train_ordinal,
    dataset_46069_test_ordinal
):
    """
    Nodo que recibe varios datasets, los imprime para verificar que se cargan correctamente
    y los retorna tal cual se reciben sin realizar ninguna operación sobre ellos.
    """

    # Imprimir las primeras filas de cada dataset para verificar que se han cargado correctamente
    print("Este es el paso inicial de carga de datos.")

    # Mostrar los primeros registros de cada dataset
    for dataset_name, dataset in locals().items():
        if "dataset_" in dataset_name:  # Asegurarnos de que solo iteramos sobre los datasets
            print(f"Dataset {dataset_name}:")
            print(dataset.head())  # Muestra las primeras filas de cada dataset

    # Regresar los datasets tal cual, sin ninguna transformación
    cleaned_data = pd.DataFrame()
    return cleaned_data


def clean_data():
    print("Este es el paso inicial de limpieza de datos.")
    
    # Aquí deberías realizar la limpieza de los datos
    cleaned_data = "algo"  # Esto es solo un ejemplo, reemplázalo con el procesamiento real de los datos
    
    return cleaned_data

def merge_data(cleaned_data):

    print("Este es otro paso en el pipeline: combinación de datos.")
    
    # Aquí deberías realizar la fusión de datos (o el procesamiento necesario)
    merged_data = f"Datos procesados a partir de {cleaned_data}"  # Este es solo un ejemplo
    
    return merged_data