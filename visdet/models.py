"""Chargement des detecteurs (Ultralytics YOLO / RT-DETR) et construction des fonctions de prediction."""
from __future__ import annotations

import numpy as np

from .tiling import _result_to_np, predict_tiled


def load_model(kind: str, weights: str):
    """kind : 'yolo' ou 'rtdetr'. `weights` : chemin .pt ou nom de poids pre-entraines (telecharge auto)."""
    from ultralytics import RTDETR, YOLO

    kind = kind.lower()
    if kind == "yolo":
        return YOLO(weights)
    if kind in ("rtdetr", "rt-detr"):
        return RTDETR(weights)
    raise ValueError(f"kind inconnu : {kind} (attendu : yolo | rtdetr)")


def make_predictor(model, mode: str = "full", imgsz: int = 640, conf: float = 0.001, iou: float = 0.6,
                   max_det: int = 500, device=None, half: bool = False,
                   tile: int = 640, overlap: float = 0.2, merge_iou: float = 0.5,
                   include_full: bool = False, full_imgsz: int = 1280, border_margin: int = 0,
                   batch: int = 16):
    """Renvoie predict_fn(img_bgr) -> (boxes xyxy, scores, classes 0-based).

    mode='full'  : l'image entiere est redimensionnee a `imgsz` (baseline).
    mode='tiled' : strategie proposee (tuiles `tile` avec chevauchement `overlap`).
    """
    extra = {}
    if device is not None:
        extra["device"] = device
    if half:
        extra["half"] = True

    if mode == "full":
        def predict_fn(img: np.ndarray):
            r = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False, **extra)[0]
            return _result_to_np(r)
    elif mode == "tiled":
        def predict_fn(img: np.ndarray):
            return predict_tiled(model, img, tile=tile, overlap=overlap, conf=conf, tile_iou=iou,
                                 merge_iou=merge_iou, imgsz=tile, include_full=include_full,
                                 full_imgsz=full_imgsz, border_margin=border_margin,
                                 max_det=max_det, batch=batch, **extra)
    else:
        raise ValueError("mode doit etre 'full' ou 'tiled'")
    return predict_fn
