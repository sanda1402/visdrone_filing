#!/usr/bin/env python
"""Evaluation d'un detecteur (image entiere ou tuiles) sur VisDrone2019-DET-val.

Calcule Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95, AP petits/moyens/grands et temps
d'inference par image, avec le MEME protocole pour tous les modeles (comparaison equitable).
Resultats : results/metrics/<name>.json + ligne ajoutee a results/summary.csv

Exemples :
    # baselines (image entiere redimensionnee a 640, puis a 1280)
    python scripts/evaluate.py --kind yolo   --weights runs/yolov8s_visdrone_640/weights/best.pt --mode full --imgsz 640 --name yolo_full640
    python scripts/evaluate.py --kind rtdetr --weights runs/rtdetr_l_visdrone_640/weights/best.pt --mode full --imgsz 640 --name rtdetr_full640
    # methode proposee : YOLO + tiling
    python scripts/evaluate.py --kind yolo   --weights runs/yolov8s_visdrone_640/weights/best.pt --mode tiled --tile 640 --overlap 0.2 --name yolo_tiled640_o20
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visdet.data import SPLIT_DIRS  # noqa: E402
from visdet.evaluation import run_eval  # noqa: E402
from visdet.models import load_model, make_predictor  # noqa: E402
from visdet.utils import env_info, save_json, set_seed, upsert_csv  # noqa: E402


def add_common_args(ap, with_include_full=True):
    ap.add_argument("--kind", choices=["yolo", "rtdetr"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--split", default="val", choices=["train", "val", "test-dev"])
    ap.add_argument("--limit", type=int, default=None, help="n'evaluer que les N premieres images (test rapide)")
    ap.add_argument("--imgsz", type=int, default=640, help="taille d'entree en mode full")
    ap.add_argument("--conf", type=float, default=0.001, help="seuil de confiance bas pour le calcul des AP")
    ap.add_argument("--iou", type=float, default=0.6, help="seuil NMS du detecteur (ignore par RT-DETR)")
    ap.add_argument("--max-det", type=int, default=500)
    ap.add_argument("--pr-conf", type=float, default=0.25, help="seuil de confiance pour Precision/Recall")
    ap.add_argument("--device", default=None, help="ex: 0 ou cpu")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--merge-iou", type=float, default=0.5)
    ap.add_argument("--border-margin", type=int, default=0)
    if with_include_full:
        ap.add_argument("--include-full", action="store_true", help="ajoute une passe sur l'image entiere")
    ap.add_argument("--full-imgsz", type=int, default=1280)
    ap.add_argument("--tile-batch", type=int, default=16)


def paths(args):
    root = Path(args.root)
    return str(root / SPLIT_DIRS[args.split] / "images"), str(root / "coco" / f"instances_{args.split}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--mode", choices=["full", "tiled"], default="full")
    ap.add_argument("--name", required=True, help="nom de l'experience (ex: yolo_full640)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    images_dir, gt_json = paths(args)
    model = load_model(args.kind, args.weights)
    predict = make_predictor(model, mode=args.mode, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                             max_det=args.max_det, device=args.device, half=args.half, tile=args.tile,
                             overlap=args.overlap, merge_iou=args.merge_iou, include_full=args.include_full,
                             full_imgsz=args.full_imgsz, border_margin=args.border_margin, batch=args.tile_batch)
    res = run_eval(predict, images_dir, gt_json, limit=args.limit, pr_conf=args.pr_conf)

    config = {k: v for k, v in vars(args).items()}
    row = {"name": args.name, "kind": args.kind, "mode": args.mode,
           "imgsz": args.imgsz if args.mode == "full" else args.tile,
           "overlap": args.overlap if args.mode == "tiled" else None, **res}
    save_json({"config": config, "metrics": res, "environment": env_info()}, Path(args.out) / "metrics" / f"{args.name}.json")
    upsert_csv(Path(args.out) / "summary.csv", row, key="name")

    print(f"\n=== {args.name} ({res['n_images']} images) ===")
    for k in ["precision", "recall", "f1", "mAP50", "mAP50-95", "AP_small", "AP_medium", "AP_large",
              "ms_per_img_mean", "fps"]:
        print(f"{k:>18s} : {res[k]:.4f}")


if __name__ == "__main__":
    main()
