#!/usr/bin/env python
"""Inference sur une image ou un dossier d'images, en mode image entiere ou par tuiles.

Usage :
    python scripts/infer.py --kind yolo --weights runs/yolov8s_visdrone_640/weights/best.pt \
        --source datasets/VisDrone/VisDrone2019-DET-val/images --mode tiled --tile 640 --overlap 0.2 --limit 10
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from visdet.data import IMG_EXTS  # noqa: E402
from visdet.models import load_model, make_predictor  # noqa: E402
from visdet.viz import draw_boxes  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image ou dossier")
    ap.add_argument("--out", default="results/inference")
    ap.add_argument("--mode", choices=["full", "tiled"], default="tiled")
    ap.add_argument("--imgsz", type=int, default=640, help="taille d'entree (mode full)")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--border-margin", type=int, default=0)
    ap.add_argument("--include-full", action="store_true")
    ap.add_argument("--conf", type=float, default=0.25, help="seuil d'affichage")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.source)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        files = files[:args.limit]

    model = load_model(args.kind, args.weights)
    predict = make_predictor(model, mode=args.mode, imgsz=args.imgsz, conf=0.05, device=args.device,
                             tile=args.tile, overlap=args.overlap, merge_iou=args.merge_iou,
                             border_margin=args.border_margin, include_full=args.include_full)
    out = Path(args.out) / f"{args.kind}_{args.mode}"
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        img = cv2.imread(str(f))
        boxes, scores, cls = predict(img)
        vis = draw_boxes(img, boxes, scores, cls, conf_thr=args.conf)
        cv2.imwrite(str(out / f.name), vis)
        print(f"{f.name}: {int((scores >= args.conf).sum())} objets (conf >= {args.conf})")
    print(f"Images annotees dans {out}")


if __name__ == "__main__":
    main()
