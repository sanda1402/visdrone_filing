"""Visualisation des detections (OpenCV)."""
from __future__ import annotations

import cv2
import numpy as np

from .data import CLASS_NAMES

_PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
            (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0)]


def draw_boxes(img: np.ndarray, boxes, scores=None, cls=None, conf_thr: float = 0.25,
               thickness: int = 1, with_labels: bool = False) -> np.ndarray:
    out = img.copy()
    for i, b in enumerate(boxes):
        s = float(scores[i]) if scores is not None else 1.0
        if s < conf_thr:
            continue
        c = int(cls[i]) if cls is not None else 0
        col = _PALETTE[c % len(_PALETTE)]
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thickness)
        if with_labels:
            cv2.putText(out, f"{CLASS_NAMES[c % len(CLASS_NAMES)]} {s:.2f}", (x1, max(y1 - 2, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    return out


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 * len(text) + 20), 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out
