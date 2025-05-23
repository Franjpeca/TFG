from kedro.pipeline import Pipeline

from kedro.config import OmegaConfigLoader

from proyecto_ola.pipelines.preprocessing import pipeline as preprocessing_pipeline
from proyecto_ola.pipelines.training import pipeline as training_pipeline
from proyecto_ola.pipelines.evaluation import pipeline as evaluation_pipeline
from proyecto_ola.pipelines.visualization import pipeline as visualization_pipeline

from proyecto_ola.pipelines.training.MORD_LogisticAT import pipeline as MORD_LogisticAT_pipeline
from proyecto_ola.pipelines.training.MORD_LogisticIT import pipeline as MORD_LogisticIT_pipeline
from proyecto_ola.pipelines.training.MORD_LAD import pipeline as MORD_LAD_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalRidge import pipeline as MORD_OrdinalRidge_pipeline
from proyecto_ola.pipelines.training.MORD_MulticlassLogistic import pipeline as MORD_MulticlassLogistic_pipeline

from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition import pipeline as ORCA_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.training.ORCA_NNOP import pipeline as ORCA_NNOP_pipeline
from proyecto_ola.pipelines.training.ORCA_NNPOM import pipeline as ORCA_NNPOM_pipeline
from proyecto_ola.pipelines.training.ORCA_REDSVM import pipeline as ORCA_REDSVM_pipeline
from proyecto_ola.pipelines.training.ORCA_SVOREX import pipeline as ORCA_SVOREX_pipeline

from kedro.pipeline import pipeline as pipeline_factory

def register_pipelines():
    # Carga de parametros
    config_loader = OmegaConfigLoader(conf_source="conf")
    params = config_loader.get("parameters")

    # Subpipelines
    preprocessing = preprocessing_pipeline.create_pipeline()
    training = training_pipeline.create_pipeline(params=params)
    evaluation = evaluation_pipeline.create_pipeline(params=params)
    visualization = visualization_pipeline.create_pipeline(params=params)

    return {
        # Pipeline completo en el orden correcto
        "__default__": preprocessing + training + evaluation,
        "preprocessing": preprocessing,
        "training": training,
        "evaluation": evaluation,
        "visualization": visualization,
    }
