from dataclasses import asdict, is_dataclass
from pathlib import Path
from enum import Enum

def to_serializable(obj):
    if is_dataclass(obj):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [to_serializable(v) for v in obj]
    else:
        return obj
