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
    import numpy as np
    import matplotlib.pyplot as plt

    if not metrics_jsons:
        return None

    metrics = list(metrics)  # asegurar orden estable
    invert_metrics = {"mae", "amae"}  # métricas a minimizar

    # 1) Recopilar valores por modelo y métrica
    models = []
    values_matrix = []

    for j in metrics_jsons:
        model_id = j.get("model_id")
        if model_id is None:
            continue
        row_vals = []
        for m in metrics:
            if m in {"qwk", "mae", "amae"}:
                v = j.get("ordinal_metrics", {}).get(m)
            else:
                v = j.get("nominal_metrics", {}).get(m)
            row_vals.append(v if v is not None else np.nan)
        if not np.all(np.isnan(row_vals)):
            models.append(model_id)
            values_matrix.append(row_vals)

    if not values_matrix:
        return None

    values = np.array(values_matrix, dtype=float)
    norm = np.full_like(values, np.nan)

    for j_col, m in enumerate(metrics):
        col = values[:, j_col]
        mask = ~np.isnan(col)
        if not np.any(mask):
            continue
        vmin = np.nanmin(col)
        vmax = np.nanmax(col)
        if np.isclose(vmax, vmin):
            norm[mask, j_col] = 1.0
        elif m in invert_metrics:
            norm[mask, j_col] = (vmax - col[mask]) / (vmax - vmin)
        else:
            norm[mask, j_col] = (col[mask] - vmin) / (vmax - vmin)

    labels = [clean_label(m) for m in models]
    n_models, n_metrics = norm.shape

    fig_width = max(8, 1.6 * n_metrics + 3)
    fig_height = max(4, 0.5 * n_models + 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(norm, aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Valor normalizado (0 peor → 1 mejor)", fontsize=10)

    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n_models):
        for j_col in range(n_metrics):
            val = values[i, j_col]
            shade = norm[i, j_col]
            if np.isnan(val):
                ax.text(j_col, i, "-", ha="center", va="center", color="0.7", fontsize=9)
            else:
                color = "black" if shade > 0.6 else "white"
                txt = f"{val:.2f}\n({shade:.1f})"
                ax.text(j_col, i, txt, ha="center", va="center", color=color, fontsize=8)

    ax.set_title(f"Comparativa de métricas – Dataset {dataset_id}", fontsize=13)
    fig.suptitle("(Normalizado por columna; MAE/AMAE invertidas)", fontsize=10, y=0.95)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def Visualize_Scatter_QWKvsMAE(
    metrics_jsons,
    dataset_id,
    execution_folder,
    x_metric="mae",
    y_metric="qwk",
    metric_type="scatter",
):
    """
    Genera un scatter con MAE en X y QWK en Y.
    - Un punto por modelo (si tiene ambas métricas).
    - Etiqueta con nombre corto del modelo.
    - Cuadrícula ligera.
    - Título: “QWK vs MAE – Dataset XXXX”.
    Devuelve un matplotlib.figure.Figure.
    """
    if not metrics_jsons:
        return None

    xs, ys, labels = [], [], []
    for j in metrics_jsons:
        ordm = j.get("ordinal_metrics", {}) or {}
        x = ordm.get(x_metric)
        y = ordm.get(y_metric)
        if x is None or y is None:
            continue

        # Etiqueta corta = nombre de modelo (sin hiperparámetros)
        model_id = j.get("model_id", "Model")
        m = re.match(r"(\w+)", model_id)
        short = m.group(1) if m else "Model"

        xs.append(x)
        ys.append(y)
        labels.append(short)

    if not xs:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys)

    # Etiquetas cercanas al punto con pequeño desplazamiento
    for x, y, text in zip(xs, ys, labels):
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xlabel("MAE (↓ mejor)")
    ax.set_ylabel("QWK (↑ mejor)")
    ax.set_title(f"QWK vs MAE – Dataset {dataset_id}")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.margins(0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    return fig