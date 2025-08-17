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


def get_execution_folder(run_id: Optional[str] = None) -> Optional[str]:
    base_dir = Path("data/06_model_metrics")
    if not base_dir.exists():
        return None

    pattern = f"{run_id}_*" if run_id else "*_*"
    best_dir, best_ts = None, -1.0


    for d in base_dir.glob(pattern):
        if not any(d.glob("Metrics_*.json")):
            continue
        ts = d.stat().st_mtime
        if ts > best_ts:
            best_ts, best_dir = ts, d

    return best_dir.name if best_dir else None