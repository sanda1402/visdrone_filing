#!/usr/bin/env python
"""Exemples visuels : meme image traitee par YOLO (image entiere), RT-DETR (image entiere) et YOLO + tiling.

Chaque image du rapport est un montage [verite terrain | YOLO | RT-DETR | YOLO + tiling]
(ou seulement les modeles fournis). Par defaut, on choisit les images de la validation contenant
le plus de petits objets (cas ou le tiling est cense aider).

Usage :
    python scripts/compare_examples.py --yolo runs/yolov8s_visdrone_640/weights/best.pt \
        --rtdetr runs/rtdetr_l_visdrone_640/weights/best.pt --n 6
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from visdet.data import SMALL_MAX, SPLIT_DIRS  # noqa: E402
from visdet.evaluation import load_gt  # noqa: E402
from visdet.models import load_model, make_predictor  # noqa: E402
from visdet.viz import draw_boxes, put_title  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolo", help="poids YOLO (.pt)")
    ap.add_argument("--rtdetr", help="poids RT-DETR (.pt)")
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--images", nargs="*", help="noms de fichiers precis (sinon: images les plus riches en petits objets)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/examples")
    args = ap.parse_args()
    if not (args.yolo or args.rtdetr):
        sys.exit("Fournir --yolo et/ou --rtdetr")

    root = Path(args.root)
    images_dir = root / SPLIT_DIRS["val"] / "images"
    gt_json = root / "coco" / "instances_val.json"
    images, gt = load_gt(str(gt_json))
    by_name = {im["file_name"]: k for k, im in images.items()}

    if args.images:
        names = args.images
    else:
        small = {}
        for k, (b, _) in gt.items():
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            small[images[k]["file_name"]] = int((area < SMALL_MAX).sum())
        names = [n for n, _ in sorted(small.items(), key=lambda kv: -kv[1])[:args.n]]

    panels = []  # (titre, predict_fn or None)
    if args.yolo:
        m = load_model("yolo", args.yolo)
        panels.append(("YOLO (image entiere)", make_predictor(m, "full", imgsz=args.imgsz, conf=0.05, device=args.device)))
        panels.append(("YOLO + tiling", make_predictor(m, "tiled", conf=0.05, tile=args.tile, overlap=args.overlap, device=args.device)))
    if args.rtdetr:
        r = load_model("rtdetr", args.rtdetr)
        panels.insert(1 if args.yolo else 0, ("RT-DETR (image entiere)", make_predictor(r, "full", imgsz=args.imgsz, conf=0.05, device=args.device)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for name in names:
        img = cv2.imread(str(images_dir / name))
        k = by_name[name]
        gb, gc = gt.get(k, (np.zeros((0, 4)), np.zeros(0, dtype=int)))
        tiles = [put_title(draw_boxes(img, gb, None, gc, thickness=1), f"Verite terrain ({len(gb)})")]
        for title, fn in panels:
            b, s, c = fn(img)
            tiles.append(put_title(draw_boxes(img, b, s, c, conf_thr=args.conf), f"{title} ({int((s >= args.conf).sum())})"))
        # grille 2 colonnes pour garder des images lisibles
        rows = [np.hstack(tiles[i:i + 2]) if len(tiles[i:i + 2]) == 2 else np.hstack([tiles[i], np.zeros_like(tiles[i])])
                for i in range(0, len(tiles), 2)]
        cv2.imwrite(str(out / f"compare_{Path(name).stem}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("ecrit", out / f"compare_{Path(name).stem}.jpg")


if __name__ == "__main__":
    main()
