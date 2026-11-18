"""Constantes et lecture des annotations VisDrone2019-DET.

Format d'une ligne d'annotation VisDrone :
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
 - category : 0 = ignored regions, 1..10 = classes utiles, 11 = others
 - score    : 0 = boite ignoree pour l'evaluation, 1 = prise en compte
 - truncation : 0 aucune, 1 partielle ; occlusion : 0 aucune, 1 partielle, 2 forte
"""
from __future__ import annotations

from pathlib import Path

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

SPLIT_DIRS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test-dev": "VisDrone2019-DET-test-dev",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Convention COCO pour la taille des objets (aire en pixels^2)
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2


def size_category(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def parse_annotation_file(path: str | Path) -> list[dict]:
    """Lit un fichier d'annotations VisDrone et renvoie une liste de dictionnaires."""
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p for p in line.strip().strip(",").split(",") if p != ""]
            if len(parts) < 8:
                continue
            x, y, w, h = (float(v) for v in parts[:4])
            score, cat, trunc, occ = (int(float(v)) for v in parts[4:8])
            rows.append(dict(x=x, y=y, w=w, h=h, score=score, cat=cat, trunc=trunc, occ=occ))
    return rows


def list_images(images_dir: str | Path) -> list[Path]:
    images_dir = Path(images_dir)
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def split_paths(root: str | Path, split: str) -> tuple[Path, Path]:
    """Renvoie (dossier images, dossier annotations) d'un split."""
    base = Path(root) / SPLIT_DIRS[split]
    return base / "images", base / "annotations"
