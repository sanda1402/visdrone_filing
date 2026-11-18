"""Utilitaires : environnement, graine, lecture/ecriture de resultats."""
from __future__ import annotations

import json
import platform
import random
import sys
from pathlib import Path

import numpy as np


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_info() -> dict:
    """Versions des bibliotheques et materiel (a joindre aux resultats pour la reproductibilite)."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["torch", "torchvision", "ultralytics", "numpy", "cv2", "pycocotools", "pandas", "matplotlib"]:
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except ImportError:
            info[mod] = None
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def upsert_csv(path: str | Path, row: dict, key: str = "name") -> None:
    """Ajoute (ou remplace, si meme `key`) une ligne dans un CSV de resultats."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old[key] != row[key]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
