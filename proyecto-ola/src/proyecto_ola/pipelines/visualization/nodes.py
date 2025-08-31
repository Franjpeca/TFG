import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

def clean_label(model_str, grid_id=None):
    """Devuelve 'NombreModelo grid_XXX' si hay grid_id, si no solo el nombre."""
    m = re.match(r"(\w+)", model_str or "")
    model_name = m.group(1) if m else "Model"
    return f"{model_name} {grid_id}" if grid_id else model_name

def plot_metric(data, metric, dataset_id):
    ORDINAL_SET = {
        "LAD", "LogisticAT", "LogisticIT", "OrdinalRidge",
        "OrdinalDecomposition", "REDSVM", "SVOREX", "NNOP", "NNPOM"
    }

    def get_family(model_id):
        m = re.match(r"(\w+)", model_id or "")
        name = m.group(1) if m else ""
        return "Ordinal" if name in ORDINAL_SET else "Clásico"

    # normaliza a triples (model_id, valor, grid_id)
    normed = []
    for e in data:
        if isinstance(e, tuple) and len(e) == 3:
            m_id, val, g_id = e
        else:
            m_id, val = e
            g_id = None
        normed.append((m_id, val, g_id))

    minimize = {"mae", "amae"}
    reverse = False if metric in minimize else True
    normed.sort(key=lambda x: x[1], reverse=reverse)

    models   = [m for m, _, _ in normed]
    values   = [v for _, v, _ in normed]
    grids    = [g for _, _, g in normed]
    labels   = [clean_label(m, g) for m, g in zip(models, grids)]
    families = [get_family(m) for m in models]
    colors   = ["#1f77b4" if fam == "Ordinal" else "#aec7e8" for fam in families]

    fig_height = max(4, 0.6 * len(models))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    ax.barh(range(len(models)), values, color=colors)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    arrow = "↓ mejor" if metric in minimize else "↑ mejor"
    ax.set_xlabel(metric)
    ax.set_title(f"{metric} ({arrow}) - Dataset {dataset_id}")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=False))
    fig.subplots_adjust(left=0.35)

    handles = [
        plt.Line2D([0], [0], marker='s', linestyle='None', color='#1f77b4', label='Ordinal'),
        plt.Line2D([0], [0], marker='s', linestyle='None', color='#aec7e8', label='Clásico'),
    ]
    ax.legend(handles=handles, loc="lower right")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    return fig

def Visualize_Ordinal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="ordinal"):
    data = [
        (j.get("model_id"), j.get("ordinal_metrics", {}).get(metric), j.get("grid_id"))
        for j in metrics_jsons
        if j.get("ordinal_metrics", {}).get(metric) is not None
    ]
    logger.info(f"[VISUALIZATION] Gráfica ORDINAL generada: metric={metric}, dataset={dataset_id}, ejecución={execution_folder}, n_modelos={len(data)}")
    return plot_metric(data, metric, dataset_id) if data else None

def Visualize_Nominal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="nominal"):
    data = [
        (j.get("model_id"), j.get("nominal_metrics", {}).get(metric), j.get("grid_id"))
        for j in metrics_jsons
        if j.get("nominal_metrics", {}).get(metric) is not None
    ]
    logger.info(f"[VISUALIZATION] Gráfica NOMINAL generada: metric={metric}, dataset={dataset_id}, ejecución={execution_folder}, n_modelos={len(data)}")
    return plot_metric(data, metric, dataset_id) if data else None

def Visualize_Heatmap_Metrics(metrics_jsons, metrics, dataset_id, execution_folder, metric_type="heatmap"):
    if not metrics_jsons:
        logger.info(f"[VISUALIZATION] Heatmap NO generado (sin datos): dataset={dataset_id}, ejecución={execution_folder}")
        return None

    ORDINAL_LABEL_COLOR = "#1f77b4"
    CLASSIC_LABEL_COLOR = "#5da5da"

    metrics = list(metrics)
    invert_metrics = {"mae", "amae"}

    models = []
    grids = []
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
            grids.append(j.get("grid_id"))
            values_matrix.append(row_vals)

    if not values_matrix:
        logger.info(f"[VISUALIZATION] Heatmap NO generado (sin datos): dataset={dataset_id}, ejecución={execution_folder}")
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

    labels = [clean_label(m, g) for m, g in zip(models, grids)]
    n_models, n_metrics = norm.shape

    fig_width = max(8, 1.6 * n_metrics + 3)
    fig_height = max(4, 0.5 * n_models + 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(norm, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
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

    ORDINAL_SET = {
        "LAD", "LogisticAT", "LogisticIT", "OrdinalRidge",
        "OrdinalDecomposition", "REDSVM", "SVOREX", "NNOP", "NNPOM"
    }
    for tick in ax.get_yticklabels():
        m = re.match(r"(\w+)", tick.get_text())
        name = m.group(1) if m else ""
        tick.set_color(ORDINAL_LABEL_COLOR if name in ORDINAL_SET else CLASSIC_LABEL_COLOR)

    ax.set_title(f"Comparativa de métricas - Dataset {dataset_id}", fontsize=13)
    fig.suptitle("(Normalizado por columna; MAE/AMAE invertidas)", fontsize=10, y=0.95)

    handles = [
        plt.Line2D([0], [0], marker='s', linestyle='None', color=ORDINAL_LABEL_COLOR, label='Ordinal'),
        plt.Line2D([0], [0], marker='s', linestyle='None', color=CLASSIC_LABEL_COLOR, label='Clásico'),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    logger.info(f"[VISUALIZATION] Heatmap generado: métricas={metrics}, dataset={dataset_id}, ejecución={execution_folder}, n_modelos={len(models)}")
    return fig

def Visualize_Scatter_QWKvsAMAE(
    metrics_jsons,
    dataset_id,
    execution_folder,
    x_metric="amae",   # ← por defecto AMAE
    y_metric="qwk",
    metric_type="scatter",
):
    if not metrics_jsons:
        logger.info(f"[VISUALIZATION] Scatter QWK vs {x_metric.upper()} NO generado (sin datos): dataset={dataset_id}, ejecución={execution_folder}")
        return None

    xs, ys, labels, colors = [], [], [], []

    ORDINAL_SET = {
        "LAD", "LogisticAT", "LogisticIT", "OrdinalRidge",
        "OrdinalDecomposition", "REDSVM", "SVOREX", "NNOP", "NNPOM"
    }

    for j in metrics_jsons:
        ordm = j.get("ordinal_metrics", {}) or {}
        x = ordm.get(x_metric)
        y = ordm.get(y_metric)
        if x is None or y is None:
            continue

        model_id = j.get("model_id", "Model")
        grid_id = j.get("grid_id")
        m = re.match(r"(\w+)", model_id or "")
        short = m.group(1) if m else "Model"

        xs.append(x)
        ys.append(y)
        labels.append(f"{short} {grid_id}" if grid_id else short)
        colors.append("#1f77b4" if short in ORDINAL_SET else "#aec7e8")

    if not xs:
        logger.info(f"[VISUALIZATION] Scatter QWK vs {x_metric.upper()} NO generado (sin datos): dataset={dataset_id}, ejecución={execution_folder}")
        return None

    # Etiquetas dinámicas y flechas segun métrica
    def axis_label(metric_name):
        metric_name_up = metric_name.upper()
        minimize = {"mae", "amae"}
        arrow = "↓ mejor" if metric_name.lower() in minimize else "↑ mejor"
        return f"{metric_name_up} ({arrow})"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys, c=colors)

    for x, y, text in zip(xs, ys, labels):
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=9)

    ax.set_xlabel(axis_label(x_metric))
    ax.set_ylabel(axis_label(y_metric))
    ax.set_title(f"{y_metric.upper()} vs {x_metric.upper()} - Dataset {dataset_id}")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.margins(0.05)

    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='None', color='#1f77b4', label='Ordinal'),
        plt.Line2D([0], [0], marker='o', linestyle='None', color='#aec7e8', label='Clásico'),
    ]
    ax.legend(handles=handles, loc="lower left")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    logger.info(f"[VISUALIZATION] Scatter {y_metric.upper()} vs {x_metric.upper()} generado: dataset={dataset_id}, ejecución={execution_folder}, n_modelos={len(xs)}")
    return fig