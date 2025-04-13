from kedro.pipeline import Pipeline

from proyecto_ola.pipelines.preprocessing import pipeline as preprocessing_pipeline
from proyecto_ola.pipelines.training import pipeline as training_pipeline

from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition import pipeline as ORCA_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.training.MORD_LogisticAT import pipeline as MORD_LogisticAT_pipeline
from proyecto_ola.pipelines.training.MORD_MNLogit import pipeline as MORD_MNLogit_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalLogisticRegression import pipeline as MORD_OrdinalLogisticRegression_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalRidge import pipeline as MORD_OrdinalRidge_pipeline
from proyecto_ola.pipelines.training.ORCA_NNPO import pipeline as ORCA_NNPO_pipeline
from proyecto_ola.pipelines.training.ORCA_NNPOM import pipeline as ORCA_NNPOM_pipeline
from proyecto_ola.pipelines.training.ORCA_REDSVM import pipeline as ORCA_REDSVM_pipeline
from proyecto_ola.pipelines.training.ORCA_SVOREX import pipeline as ORCA_SVOREX_pipeline


def register_pipelines():
    return {
        "preprocessing": preprocessing_pipeline.create_pipeline(),
        # Pipelines de training
        "training": training_pipeline.create_pipeline(),
        # Pipelines de metodos (subpipelines)
        "ORCA_OrdinalDecomposition": ORCA_OrdinalDecomposition_pipeline.create_pipeline(),
        "MORD_LogisticAT": MORD_LogisticAT_pipeline.create_pipeline(),
        "MORD_MNLogi": MORD_MNLogit_pipeline.create_pipeline(),
        "MORD_OrdinalLogisticRegression": MORD_OrdinalLogisticRegression_pipeline.create_pipeline(),
        "MORD_OrdinalRidge": MORD_OrdinalRidge_pipeline.create_pipeline(),
        "ORCA_NNPO": ORCA_NNPO_pipeline.create_pipeline(),
        "ORCA_NNPOM": ORCA_NNPOM_pipeline.create_pipeline(),
        "ORCA_REDSVM": ORCA_REDSVM_pipeline.create_pipeline(),
        "ORCA_SVOREX": ORCA_SVOREX_pipeline.create_pipeline(),

        "__default__": preprocessing_pipeline.create_pipeline()
    }