import pandas as pd

def clean_data_all(*datasets):
    cleaned_list = []
    print("Limpiando los datasets...")
    for dataset in datasets:
        cleaned_data = clean_data(dataset)  # Funcion dentro del nodo
        cleaned_list.append(cleaned_data)  # Guardar el dataset limpio en la lista
    return tuple(cleaned_list)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Verificar que el DataFrame no este vacio
    if data.empty:
        print("Warning: DataFrame vacio!")

    # Eliminacion de valores nulos
    data_cleaned = data.dropna()

    # Eliminar la columna de regresion
    data = data.drop(columns=[data.columns[-2]])
    return data


def merge_data(*cleaned_datasets):
    train_datasets = cleaned_datasets[::2]  # Los datasets de entrenamiento están en las posiciones pares
    test_datasets = cleaned_datasets[1::2]  # Los datasets de test están en las posiciones impares
    
    # Juntar todos los datasets de train en uno solo
    merged_train = pd.concat(train_datasets, axis=0, ignore_index=True)
    
    # Juntar todos los datasets de test en uno solo
    merged_test = pd.concat(test_datasets, axis=0, ignore_index=True)
    
    return merged_train, merged_test