"""Metriques : Precision / Recall (appariement glouton, IoU 0.5) et AP COCO (pycocotools)."""
from __future__ import annotations

import contextlib
import io

import numpy as np


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU deux a deux entre a (N,4) et b (M,4) au format xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match_counts(gt: dict, preds: dict, conf: float, iou_thr: float = 0.5) -> tuple[int, int, int]:
    """Compte (TP, FP, FN) sur toutes les images pour un seuil de confiance donne.

    gt    : {image_id: (boxes xyxy (N,4), classes (N,))}
    preds : {image_id: (boxes xyxy (M,4), scores (M,), classes (M,))}
    Une prediction est TP si elle est de la bonne classe, IoU >= iou_thr avec une
    verite terrain non encore appariee (appariement glouton par score decroissant).
    """
    tp = fp = fn = 0
    for img_id in set(gt) | set(preds):
        gb, gc = gt.get(img_id, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
        pb, ps, pc = preds.get(img_id, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)))
        m = ps >= conf
        pb, ps, pc = pb[m], ps[m], pc[m]
        order = np.argsort(-ps, kind="stable")
        pb, pc = pb[order], pc[order]
        iou = box_iou_matrix(pb, gb)
        used = np.zeros(len(gb), dtype=bool)
        n_tp = 0
        for i in range(len(pb)):
            if len(gb) == 0:
                break
            cand = np.where((gc == pc[i]) & (~used) & (iou[i] >= iou_thr))[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                n_tp += 1
        tp += n_tp
        fp += len(pb) - n_tp
        fn += len(gb) - n_tp
    return tp, fp, fn


def precision_recall(gt: dict, preds: dict, conf: float, iou_thr: float = 0.5) -> dict:
    tp, fp, fn = match_counts(gt, preds, conf, iou_thr)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(conf=conf, precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


def best_f1(gt: dict, preds: dict, grid=None, iou_thr: float = 0.5) -> dict:
    grid = grid if grid is not None else np.arange(0.05, 0.95, 0.05)
    rows = [precision_recall(gt, preds, float(c), iou_thr) for c in grid]
    return max(rows, key=lambda d: d["f1"])


COCO_KEYS = ["mAP50-95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR500", "AR_small", "AR_medium", "AR_large"]


def _stat(ev, ap: bool, iou_thr=None, area="all", max_det=500) -> float:
    """Reimplementation de COCOeval.summarize pour un maxDets personnalise (500 au lieu de 100).

    Le `summarize()` de pycocotools cherche maxDets == 100 et renvoie -1 si on a configure
    d'autres valeurs : on calcule donc les statistiques directement depuis ev.eval.
    """
    p = ev.params
    aind = [i for i, a in enumerate(p.areaRngLbl) if a == area]
    mind = [i for i, m in enumerate(p.maxDets) if m == max_det]
    if ap:
        s = ev.eval["precision"]                       # [T, R, K, A, M]
        if iou_thr is not None:
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        s = ev.eval["recall"]                          # [T, K, A, M]
        if iou_thr is not None:
            t = np.where(np.isclose(p.iouThrs, iou_thr))[0]
            s = s[t]
        s = s[:, :, aind, mind]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else float("nan")


def coco_evaluate(gt_json: str, preds_coco: list[dict], img_ids: list[int]) -> dict:
    """AP COCO via pycocotools (maxDets = [1, 10, 500] car VisDrone est dense)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not preds_coco:
        return {k: 0.0 for k in COCO_KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(gt_json)
        dt = coco.loadRes(preds_coco)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = list(img_ids)
        ev.params.maxDets = [1, 10, 500]
        ev.evaluate()
        ev.accumulate()
    return {
        "mAP50-95": _stat(ev, True),
        "mAP50": _stat(ev, True, 0.5),
        "mAP75": _stat(ev, True, 0.75),
        "AP_small": _stat(ev, True, area="small"),
        "AP_medium": _stat(ev, True, area="medium"),
        "AP_large": _stat(ev, True, area="large"),
        "AR1": _stat(ev, False, max_det=1),
        "AR10": _stat(ev, False, max_det=10),
        "AR500": _stat(ev, False, max_det=500),
        "AR_small": _stat(ev, False, area="small"),
        "AR_medium": _stat(ev, False, area="medium"),
        "AR_large": _stat(ev, False, area="large"),
    }
