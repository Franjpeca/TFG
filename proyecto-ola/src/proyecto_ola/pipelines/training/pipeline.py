from kedro.pipeline import Pipeline

from proyecto_ola.pipelines.training.MORD_LogisticAT.pipeline import create_pipeline as create_ORCA_LogisticAT_pipeline
from proyecto_ola.pipelines.training.MORD_MNLogit.pipeline import create_pipeline as create_MORD_MNLogit_pipeline
from proyecto_ola.pipelines.training.MORD_OrdinalLogisticRegression.pipeline import create_pipeline as create_MORD_OrdinalLogisticRegression
from proyecto_ola.pipelines.training.MORD_OrdinalRidge.pipeline import create_pipeline as create_MORD_OrdinalRidge
from proyecto_ola.pipelines.training.ORCA_NNPO.pipeline import create_pipeline as create_ORCA_NNPO
from proyecto_ola.pipelines.training.ORCA_NNPOM.pipeline import create_pipeline as create_ORCA_NNPOM
from proyecto_ola.pipelines.training.ORCA_OrdinalDecomposition.pipeline import create_pipeline as create_OrdinalDecomposition_pipeline
from proyecto_ola.pipelines.training.ORCA_REDSVM.pipeline import create_pipeline as create_ORCA_REDSVM
from proyecto_ola.pipelines.training.ORCA_SVOREX.pipeline import create_pipeline as create_ORCA_SVOREX

def create_pipeline(**kwargs) -> Pipeline:
    return (
        create_OrdinalDecomposition_pipeline() 
        + create_ORCA_LogisticAT_pipeline() 
        + create_MORD_MNLogit_pipeline()
        + create_MORD_OrdinalLogisticRegression()
        + create_MORD_OrdinalRidge()
        + create_ORCA_NNPO()
        + create_ORCA_NNPOM()
        + create_ORCA_REDSVM()
        + create_ORCA_SVOREX()
    )