import matplotlib.pyplot as plt
import warnings
import textwrap

def Visualize_Nominal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="nominal"):
    data = [(j["model_id"], j["nominal_metrics"].get(metric))
            for j in metrics_jsons
            if j.get("nominal_metrics", {}).get(metric) is not None]
    if not data:
        return None

    models, values = zip(*sorted(data, key=lambda x: x[1], reverse=True))
    labels = [textwrap.shorten(m, width=25, placeholder="…") for m in models]

    fig_height = max(4, 0.5 * len(models))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(range(len(models)), values)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel("Modelos", fontsize=10)
    ax.set_title(f"{metric} – Dataset {dataset_id}", fontsize=12)

    fig.subplots_adjust(left=0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    return fig


def Visualize_Ordinal_Metric(metrics_jsons, metric, dataset_id, execution_folder, metric_type="ordinal"):
    data = [(j["model_id"], j["ordinal_metrics"].get(metric))
            for j in metrics_jsons
            if j.get("ordinal_metrics", {}).get(metric) is not None]
    if not data:
        return None

    models, values = zip(*sorted(data, key=lambda x: x[1], reverse=True))
    labels = [textwrap.shorten(m, width=25, placeholder="…") for m in models]

    fig_height = max(4, 0.5 * len(models))
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.barh(range(len(models)), values)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric, fontsize=10)
    ax.set_ylabel("Modelos", fontsize=10)
    ax.set_title(f"{metric} – Dataset {dataset_id}", fontsize=12)

    fig.subplots_adjust(left=0.35)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()

    return fig