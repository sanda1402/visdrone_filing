#!/usr/bin/env python
"""Etude d'ablation de la strategie de tiling : taille de tuile, chevauchement, passe image entiere,
marge de bord. Chaque combinaison est evaluee avec le meme protocole que scripts/evaluate.py.

Resultats : results/ablation.csv (une ligne par configuration) puis tracer avec scripts/make_figures.py.

Exemples :
    # variation du chevauchement (tuile fixe 640)
    python scripts/ablation.py --kind yolo --weights runs/.../best.pt --tiles 640 --overlaps 0 0.1 0.2 0.3
    # variation de la taille de tuile (chevauchement fixe 0.2)
    python scripts/ablation.py --kind yolo --weights runs/.../best.pt --tiles 320 512 640 1024 --overlaps 0.2
    # effet de la passe image entiere et de la marge de bord
    python scripts/ablation.py --kind yolo --weights runs/.../best.pt --tiles 640 --overlaps 0.2 \
        --include-full 0 1 --border-margins 0 4
Astuce : ajouter --limit 100 pour un essai rapide avant de lancer l'ablation complete.
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import add_common_args, paths  # noqa: E402
from visdet.evaluation import run_eval  # noqa: E402
from visdet.models import load_model, make_predictor  # noqa: E402
from visdet.utils import env_info, save_json, set_seed, upsert_csv  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, with_include_full=False)
    ap.add_argument("--tiles", type=int, nargs="+", default=[640])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--include-full", dest="include_full", type=int, nargs="+", default=[0], choices=[0, 1])
    ap.add_argument("--border-margins", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    images_dir, gt_json = paths(args)
    model = load_model(args.kind, args.weights)

    grid = list(itertools.product(args.tiles, args.overlaps, args.include_full, args.border_margins))
    print(f"{len(grid)} configuration(s) a evaluer")
    for tile, ov, inc, bm in grid:
        name = f"{args.kind}_tile{tile}_ov{int(round(ov * 100))}_full{inc}_bm{bm}"
        predict = make_predictor(model, mode="tiled", conf=args.conf, iou=args.iou, max_det=args.max_det,
                                 device=args.device, half=args.half, tile=tile, overlap=ov,
                                 merge_iou=args.merge_iou, include_full=bool(inc),
                                 full_imgsz=args.full_imgsz, border_margin=bm, batch=args.tile_batch)
        print(f"\n--- {name}")
        res = run_eval(predict, images_dir, gt_json, limit=args.limit, pr_conf=args.pr_conf)
        row = {"name": name, "kind": args.kind, "tile": tile, "overlap": ov, "include_full": inc,
               "border_margin": bm, **res}
        upsert_csv(Path(args.out) / "ablation.csv", row, key="name")
        print(f"mAP50={res['mAP50']:.4f}  mAP50-95={res['mAP50-95']:.4f}  AP_small={res['AP_small']:.4f}  "
              f"recall={res['recall']:.4f}  {res['ms_per_img_mean']:.1f} ms/img")
    save_json({"args": vars(args), "environment": env_info()}, Path(args.out) / "metrics" / "ablation_config.json")
    print(f"\nTermine -> {Path(args.out) / 'ablation.csv'}")


if __name__ == "__main__":
    main()
