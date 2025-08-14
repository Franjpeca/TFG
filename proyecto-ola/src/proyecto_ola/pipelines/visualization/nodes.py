import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import re
import numpy as np


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


def Visualize_Heatmap_Metrics(metrics_jsons, metrics, dataset_id, execution_folder, metric_type="heatmap"):
    if not metrics_jsons:
        return None

    metrics = list(metrics)  # asegurar orden estable
    invert_metrics = {"mae", "amae"}  # métricas a minimizar

    # 1) Recopilar valores por modelo y métrica
    #    Estructura esperada en cada json:
    #    - j["model_id"]
    #    - j["nominal_metrics"]{accuracy, f1_score}
    #    - j["ordinal_metrics"]{qwk, mae, amae}
    models = []
    values_matrix = []  # matriz con valores reales (no normalizados)

    for j in metrics_jsons:
        model_id = j.get("model_id")
        if model_id is None:
            # si no hay model_id, saltamos
            continue

        row_vals = []
        for m in metrics:
            if m in {"qwk", "mae", "amae"}:
                v = j.get("ordinal_metrics", {}).get(m)
            else:
                v = j.get("nominal_metrics", {}).get(m)
            row_vals.append(v if v is not None else np.nan)
        # si la fila está totalmente vacía, ignoramos ese modelo
        if not np.all(np.isnan(row_vals)):
            models.append(model_id)
            values_matrix.append(row_vals)

    if not values_matrix:
        return None

    values = np.array(values_matrix, dtype=float)  # shape: (n_models, n_metrics)

    # 2) Normalización por columna a [0,1], invirtiendo 'mae' y 'amae'
    norm = np.empty_like(values)
    norm[:] = np.nan
    for j_col, m in enumerate(metrics):
        col = values[:, j_col]
        mask = ~np.isnan(col)
        if not np.any(mask):
            continue

        vmin = np.nanmin(col)
        vmax = np.nanmax(col)
        if np.isclose(vmax, vmin):
            # todos iguales -> poner 1.0 para no “apagar” la columna
            norm[mask, j_col] = 1.0
            continue

        if m in invert_metrics:
            # mayor es mejor tras invertir
            norm[mask, j_col] = (vmax - col[mask]) / (vmax - vmin)
        else:
            norm[mask, j_col] = (col[mask] - vmin) / (vmax - vmin)

    # 3) Plot
    labels = [clean_label(m) for m in models]
    n_models, n_metrics = norm.shape

    fig_width = max(8, 1.6 * n_metrics + 3)
    fig_height = max(4, 0.5 * n_models + 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # imshow sobre matriz normalizada (0–1), colormap viridis
    im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    # ticks y etiquetas
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=10)

    # anotar valores reales dentro de las celdas
    # elegir color del texto para contraste (negro en celdas claras, blanco en oscuras)
    for i in range(n_models):
        for j_col in range(n_metrics):
            val = values[i, j_col]
            if np.isnan(val):
                txt = "-"
                # Si la celda es NaN, píntalo tenue (texto gris)
                ax.text(j_col, i, txt, ha="center", va="center", color="0.7", fontsize=9)
            else:
                txt = f"{val:.3f}"
                shade = norm[i, j_col]
                color = "black" if (not np.isnan(shade) and shade > 0.6) else "white"
                ax.text(j_col, i, txt, ha="center", va="center", color=color, fontsize=9)

    ax.set_title(f"Comparativa de métricas – Dataset {dataset_id}")
    fig.tight_layout()

    return fig