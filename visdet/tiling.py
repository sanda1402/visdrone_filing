"""Strategie multi-echelle : decoupage en tuiles avec chevauchement + fusion des predictions.

Principe
--------
1. L'image est decoupee en tuiles de taille `tile` avec un chevauchement `overlap`
   (fraction de la taille de la tuile). Les dernieres tuiles sont recalees sur le bord
   de l'image pour que toute l'image soit couverte.
2. Le detecteur est lance sur chaque tuile (traitement par lots), eventuellement aussi
   sur l'image entiere (`include_full`) pour conserver les grands objets.
3. Les boites sont replacees dans le repere de l'image d'origine.
4. Optionnel (`border_margin`) : les boites collees a un bord *interne* de tuile sont
   supprimees (elles sont probablement tronquees ; la tuile voisine, grace au
   chevauchement, voit l'objet entier).
5. Fusion : NMS par classe (seuil `merge_iou`) pour supprimer les doublons des zones
   de chevauchement.

Le module n'utilise que numpy ; le modele doit seulement exposer `predict(images, ...)`
au format Ultralytics (resultats avec `.boxes.xyxy`, `.boxes.conf`, `.boxes.cls`).
"""
from __future__ import annotations

import numpy as np


def _np(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def tile_coords(W: int, H: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int]]:
    """Coins haut-gauche (x, y) des tuiles couvrant une image W x H."""
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap doit etre dans [0, 1[")
    if tile <= 0:
        raise ValueError("tile doit etre > 0")
    step = max(1, int(round(tile * (1.0 - overlap))))

    def axis(length: int) -> list[int]:
        if length <= tile:
            return [0]
        pos = list(range(0, length - tile + 1, step))
        if pos[-1] + tile < length:
            pos.append(length - tile)  # derniere tuile calee sur le bord
        return pos

    return [(x, y) for y in axis(H) for x in axis(W)]


def box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a1 + a2 - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """NMS classique. Renvoie les indices conserves (tries par score decroissant)."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou_one(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou_thr]
    return np.asarray(keep, dtype=int)


def batched_nms(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, iou_thr: float) -> np.ndarray:
    """NMS applique separement a chaque classe (astuce du decalage de coordonnees)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    offset = cls.astype(np.float64) * (float(boxes.max()) + 1.0)
    return nms(boxes + offset[:, None], scores, iou_thr)


def _result_to_np(r) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = r.boxes
    return (
        _np(b.xyxy).reshape(-1, 4).astype(np.float64),
        _np(b.conf).reshape(-1).astype(np.float64),
        _np(b.cls).reshape(-1).astype(int),
    )


def predict_tiled(
    model,
    img: np.ndarray,
    tile: int = 640,
    overlap: float = 0.2,
    conf: float = 0.001,
    tile_iou: float = 0.6,
    merge_iou: float = 0.5,
    imgsz: int | None = None,
    include_full: bool = False,
    full_imgsz: int = 1280,
    border_margin: int = 0,
    max_det: int = 500,
    batch: int = 16,
    **predict_kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detection par tuiles. Renvoie (boxes xyxy, scores, classes) dans le repere de `img` (BGR, HxWx3)."""
    H, W = img.shape[:2]
    coords = tile_coords(W, H, tile, overlap)
    all_b, all_s, all_c = [], [], []

    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        crops = [np.ascontiguousarray(img[y:y + tile, x:x + tile]) for x, y in chunk]
        results = model.predict(crops, imgsz=imgsz or tile, conf=conf, iou=tile_iou,
                                max_det=max_det, verbose=False, **predict_kwargs)
        for (x, y), crop, r in zip(chunk, crops, results):
            b, s, c = _result_to_np(r)
            if len(b) == 0:
                continue
            if border_margin > 0:
                th, tw = crop.shape[:2]
                keep = np.ones(len(b), dtype=bool)
                if x > 0:
                    keep &= b[:, 0] > border_margin
                if y > 0:
                    keep &= b[:, 1] > border_margin
                if x + tw < W:
                    keep &= b[:, 2] < tw - border_margin
                if y + th < H:
                    keep &= b[:, 3] < th - border_margin
                b, s, c = b[keep], s[keep], c[keep]
            b = b + np.array([x, y, x, y], dtype=np.float64)
            all_b.append(b); all_s.append(s); all_c.append(c)

    if include_full:
        r = model.predict(img, imgsz=full_imgsz, conf=conf, iou=tile_iou,
                          max_det=max_det, verbose=False, **predict_kwargs)[0]
        b, s, c = _result_to_np(r)
        if len(b):
            all_b.append(b); all_s.append(s); all_c.append(c)

    if not all_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)

    boxes = np.concatenate(all_b); scores = np.concatenate(all_s); cls = np.concatenate(all_c)
    # rogner aux limites de l'image
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
    keep = batched_nms(boxes, scores, cls, merge_iou)[:max_det]
    return boxes[keep], scores[keep], cls[keep]
