import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import re


def clean_label(model_str):
    """Genera una etiqueta legible para el eje Y basada en el model_id."""
    # Extrae nombre del modelo
    model_name_match = re.match(r"(\w+)", model_str)
    model_name = model_name_match.group(1) if model_name_match else "Model"

    # Hiperparámetros clave a mostrar
    keys_of_interest = [
        "C", "alpha", "k", "d", "g", "p", "t", "epsilonInit",
        "hiddenN", "tol", "lambdaValue"
    ]
    params = []
    for key in keys_of_interest:
        match = re.search(rf"{key}=([^,)\s]+)", model_str)
        if match:
            params.append(f"{key}={match.group(1)}")

    return f"{model_name}({', '.join(params)})" if params else model_name


def plot_metric(data, metric, dataset_id):
    models, values = zip(*sorted(data, key=lambda x: x[1], reverse=True))
    labels = [clean_label(m) for m in models]

    fig_height = max(4, 0.6 * len(models))  # más espacio entre modelos
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ax.barh(range(len(models)), values)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} – Dataset {dataset_id}")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=False))  # sin redondeos feos
    fig.subplots_adjust(left=0.35)  # margen para nombres
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig


def Visualize_Ordinal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="ordinal"):
    data = [
        (j["model_id"], j.get("ordinal_metrics", {}).get(metric))
        for j in metrics_jsons
        if j.get("ordinal_metrics", {}).get(metric) is not None
    ]
    return plot_metric(data, metric, dataset_id) if data else None


def Visualize_Nominal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="nominal"):
    data = [
        (j["model_id"], j.get("nominal_metrics", {}).get(metric))
        for j in metrics_jsons
        if j.get("nominal_metrics", {}).get(metric) is not None
    ]
    return plot_metric(data, metric, dataset_id) if data else None