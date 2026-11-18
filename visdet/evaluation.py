"""Evaluation unifiee d'une fonction de prediction (meme protocole pour YOLO, RT-DETR et YOLO+tiling).

Metriques : Precision, Recall, F1 (IoU 0.5, seuil de confiance `pr_conf`), mAP@0.5, mAP@0.5:0.95,
AP des petits / moyens / grands objets (COCO, pycocotools) et temps d'inference par image
(batch = 1, decodage de l'image exclu, tiling + fusion inclus).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from .metrics import best_f1, coco_evaluate, precision_recall


def _sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def load_gt(gt_json: str):
    with open(gt_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    images = {im["id"]: im for im in d["images"]}
    gt = {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        b, c = gt.setdefault(a["image_id"], ([], []))
        b.append([x, y, x + w, y + h])
        c.append(a["category_id"] - 1)
    gt = {k: (np.asarray(b, dtype=np.float64).reshape(-1, 4), np.asarray(c, dtype=int)) for k, (b, c) in gt.items()}
    return images, gt


def _in_ignored(box, regions) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(rx <= cx <= rx + rw and ry <= cy <= ry + rh for rx, ry, rw, rh in regions)


def run_eval(predict_fn, images_dir: str, gt_json: str, limit: int | None = None,
             pr_conf: float = 0.25, warmup: int = 3, progress: bool = True) -> dict:
    """Evalue `predict_fn(img_bgr) -> (boxes xyxy, scores, cls)` sur les images du fichier COCO `gt_json`."""
    images, gt_all = load_gt(gt_json)
    ids = sorted(images)
    if limit:
        ids = ids[:limit]
    images_dir = Path(images_dir)

    # echauffement (GPU, chargement des noyaux) : non chronometre
    for k in ids[:warmup]:
        img = cv2.imread(str(images_dir / images[k]["file_name"]))
        predict_fn(img)

    times, preds_by_img, coco_preds = [], {}, []
    it = ids
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(ids, desc="inference")
        except ImportError:
            pass
    for k in it:
        im = images[k]
        img = cv2.imread(str(images_dir / im["file_name"]))
        if img is None:
            raise FileNotFoundError(images_dir / im["file_name"])
        _sync()
        t0 = time.perf_counter()
        boxes, scores, cls = predict_fn(img)
        _sync()
        times.append((time.perf_counter() - t0) * 1000.0)

        # les predictions dont le centre est dans une zone "ignored region" ne comptent pas
        regions = im.get("ignore", [])
        if regions and len(boxes):
            m = np.array([not _in_ignored(b, regions) for b in boxes])
            boxes, scores, cls = boxes[m], scores[m], cls[m]
        preds_by_img[k] = (boxes, scores, cls)
        for b, s, c in zip(boxes, scores, cls):
            coco_preds.append({"image_id": int(k), "category_id": int(c) + 1,
                               "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                               "score": float(s)})

    gt = {k: gt_all[k] for k in ids if k in gt_all}
    res = coco_evaluate(gt_json, coco_preds, ids)
    pr = precision_recall(gt, preds_by_img, pr_conf)
    bf = best_f1(gt, preds_by_img)
    t = np.asarray(times)
    res.update({
        "precision": pr["precision"], "recall": pr["recall"], "f1": pr["f1"], "pr_conf": pr_conf,
        "best_f1": bf["f1"], "best_f1_conf": bf["conf"],
        "best_f1_precision": bf["precision"], "best_f1_recall": bf["recall"],
        "ms_per_img_mean": float(t.mean()), "ms_per_img_median": float(np.median(t)),
        "ms_per_img_p95": float(np.percentile(t, 95)), "fps": float(1000.0 / t.mean()),
        "n_images": len(ids), "n_predictions": len(coco_preds),
    })
    return res
