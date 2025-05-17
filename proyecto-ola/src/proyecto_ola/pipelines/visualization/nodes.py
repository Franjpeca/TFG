import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def Visualization_Overview(json_data):
    logger.info(f"[Visualization] Mostrando overview para {json_data['dataset_id']}")

    df = pd.DataFrame({
        "Métrica de Evaluación": list(json_data["nominal_metrics"]) + list(json_data["ordinal_metrics"]),
        "Valor": list(json_data["nominal_metrics"].values()) + list(json_data["ordinal_metrics"].values())
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(df["Métrica de Evaluación"], df["Valor"], color="lightgreen")
    ax.bar_label(bars, fmt="%.3f")  # ✅ Etiquetas con valores numéricos
    ax.set_title(f"Resumen de métricas de evaluación\nDataset: {json_data['dataset_id']}\nModelo: {json_data['model_id']}")
    ax.set_ylabel("Valor de la métrica (rango 0-1)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig

def Visualization_Distributions(json_data):
    logger.info(f"[Visualization] Mostrando distribuciones para {json_data['dataset_id']}")
    fig, ax = plt.subplots(figsize=(6, 4))
    nominal = json_data["nominal_metrics"]
    bars = ax.bar(nominal.keys(), nominal.values(), color="skyblue")
    ax.bar_label(bars, fmt="%.3f")  # ✅ Etiquetas con valores numéricos
    ax.set_title(f"Métricas de evaluación para variables Nominales\nDataset: {json_data['dataset_id']}\nModelo: {json_data['model_id']}")
    ax.set_ylabel("Valor de la métrica (rango 0-1)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig

def Visualization_Correlations(json_data):
    logger.info(f"[Visualization] Mostrando correlaciones para {json_data['dataset_id']}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ordinal = json_data["ordinal_metrics"]
    bars = ax.bar(ordinal.keys(), ordinal.values(), color="salmon")
    ax.bar_label(bars, fmt="%.3f")  # ✅ Etiquetas con valores numéricos
    ax.set_title(f"Métricas de evaluación para variables Ordinales\nDataset: {json_data['dataset_id']}\nModelo: {json_data['model_id']}")
    ax.set_ylabel("Valor de la métrica (rango 0-1)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig