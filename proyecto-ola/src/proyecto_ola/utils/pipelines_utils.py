import sys
from typing import Optional
from pathlib import Path

def find_parameters_cli(param_name: str, params: dict) -> Optional[str]:
    # 1) Prioriza lo que ya viene en params (parameters.yml + extra_params merged)
    value = params.get(param_name)
    if value is not None:
        return value

    # Hay dos formas de pasar los parametros en Kedro por CLI. Se manejan ambas formas
    # 2) Busca --params=... o --params ... en un SOLO bucle
    raw = None
    for i, arg in enumerate(sys.argv):
        if arg == "--params" and i + 1 < len(sys.argv):
            raw = sys.argv[i + 1]
        elif arg.startswith("--params="):
            raw = arg.split("=", 1)[1].strip("\"'")
        # Si ya lo tienes, no sigas
        if raw:
            break

    if not raw:
        return None

    # 3) Parsea "k=v,k2=v2,..." y devuelve el solicitado
    for kv in raw.split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k.strip() == param_name:
                return v.strip()

    return None


def only_evaluation() -> bool:
    args = sys.argv
    for i, a in enumerate(args):
        if a == "-p" and i + 1 < len(args) and args[i + 1] == "evaluation":
            return True
        if a == "--pipeline" and i + 1 < len(args) and args[i + 1] == "evaluation":
            return True
        if a.startswith("--pipeline=") and a.split("=", 1)[1] == "evaluation":
            return True
    return False


def find_latest_metrics_execution_folder(metrics_base: Path) -> Optional[str]:
    latest_file = None
    latest_time = -1.0
    for folder in metrics_base.glob("*_*"):
        for file in folder.glob("Metrics_*.json"):
            mtime = file.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file
    if latest_file:
        return latest_file.parent.name
    return None