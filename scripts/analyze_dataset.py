#!/usr/bin/env python
"""Analyse du jeu VisDrone2019-DET : annotations, categories, distribution des tailles d'objets.

Sorties (dans --out, par defaut results/dataset_stats/) :
  - annotations.csv            : toutes les boites utiles (split, image, x, y, w, h, categorie, ...)
  - summary.json / summary.md  : chiffres cles (images, instances, % petits objets, densite, ...)
  - class_counts.png           : nombre d'instances par classe et par split
  - size_distribution.png      : histogramme de sqrt(aire) (echelle log), seuils COCO 32 / 96 px
  - size_by_class.png          : repartition petit / moyen / grand par classe
  - objects_per_image.png      : densite d'objets par image
  - occlusion_truncation.png   : repartition occlusion / troncature
Usage :
    python scripts/analyze_dataset.py --root datasets/VisDrone
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from visdet.data import (CLASS_NAMES, SMALL_MAX, MEDIUM_MAX, SPLIT_DIRS, list_images,  # noqa: E402
                         parse_annotation_file, size_category, split_paths)
from visdet.utils import save_json  # noqa: E402


def load_split(root: Path, split: str):
    images_dir, ann_dir = split_paths(root, split)
    rows, img_rows = [], []
    for p in list_images(images_dir):
        with Image.open(p) as im:
            W, H = im.size
        anns = [r for r in parse_annotation_file(ann_dir / (p.stem + ".txt"))
                if 1 <= r["cat"] <= 10 and r["score"] != 0]
        img_rows.append(dict(split=split, image=p.name, width=W, height=H, n_objects=len(anns)))
        for r in anns:
            rows.append(dict(split=split, image=p.name, x=r["x"], y=r["y"], w=r["w"], h=r["h"],
                             category=CLASS_NAMES[r["cat"] - 1], truncation=r["trunc"], occlusion=r["occ"]))
    return pd.DataFrame(rows), pd.DataFrame(img_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/VisDrone")
    ap.add_argument("--out", default="results/dataset_stats")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test-dev"], choices=list(SPLIT_DIRS))
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    boxes, imgs = [], []
    for s in args.splits:
        if not (root / SPLIT_DIRS[s] / "images").is_dir():
            print(f"[{s}] absent, ignore")
            continue
        b, i = load_split(root, s)
        boxes.append(b); imgs.append(i)
        print(f"[{s}] {len(i)} images, {len(b)} instances")
    if not boxes:
        sys.exit("Aucun split trouve.")
    B, I = pd.concat(boxes, ignore_index=True), pd.concat(imgs, ignore_index=True)
    B["area"] = B["w"] * B["h"]
    B["sqrt_area"] = np.sqrt(B["area"])
    B["size"] = B["area"].apply(size_category)
    B.to_csv(out / "annotations.csv", index=False)

    # ---- resume chiffre
    summary = {}
    for s, g in B.groupby("split"):
        gi = I[I.split == s]
        summary[s] = {
            "images": int(len(gi)),
            "instances": int(len(g)),
            "objects_per_image_mean": float(gi.n_objects.mean()),
            "objects_per_image_max": int(gi.n_objects.max()),
            "image_size_most_common": f"{gi.width.mode()[0]}x{gi.height.mode()[0]}",
            "pct_small(<32^2)": float(100 * (g["size"] == "small").mean()),
            "pct_medium": float(100 * (g["size"] == "medium").mean()),
            "pct_large(>=96^2)": float(100 * (g["size"] == "large").mean()),
            "median_sqrt_area_px": float(g.sqrt_area.median()),
            "pct_occluded(>0)": float(100 * (g.occlusion > 0).mean()),
            "pct_truncated": float(100 * (g.truncation > 0).mean()),
        }
    save_json(summary, out / "summary.json")
    md = pd.DataFrame(summary).T.round(2).to_markdown() if _has_tabulate() else pd.DataFrame(summary).T.round(2).to_string()
    (out / "summary.md").write_text(md + "\n")
    print(md)

    # ---- instances par classe et par split
    cc = B.groupby(["category", "split"]).size().unstack(fill_value=0).reindex(CLASS_NAMES)
    cc.to_csv(out / "class_counts.csv")
    ax = cc.plot(kind="bar", figsize=(9, 4), logy=True)
    ax.set_ylabel("nombre d'instances (log)"); ax.set_title("Instances par categorie")
    plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(out / "class_counts.png", dpi=150); plt.close()

    # ---- distribution des tailles
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(0, np.log10(max(B.sqrt_area.max(), 400)), 60)
    for s, g in B.groupby("split"):
        ax.hist(g.sqrt_area, bins=bins, alpha=0.5, label=s, density=True)
    ax.axvline(np.sqrt(SMALL_MAX), color="r", ls="--", label="32 px (petit)")
    ax.axvline(np.sqrt(MEDIUM_MAX), color="k", ls="--", label="96 px (grand)")
    ax.set_xscale("log"); ax.set_xlabel("sqrt(aire) de la boite (px)"); ax.set_ylabel("densite")
    ax.set_title("Distribution de la taille des objets"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "size_distribution.png", dpi=150); plt.close()

    # ---- petit / moyen / grand par classe
    sc = B.groupby(["category", "size"]).size().unstack(fill_value=0).reindex(CLASS_NAMES).fillna(0)
    sc = sc[[c for c in ["small", "medium", "large"] if c in sc.columns]]
    (sc.div(sc.sum(axis=1), axis=0) * 100).round(1).to_csv(out / "size_by_class_pct.csv")
    ax = sc.div(sc.sum(axis=1), axis=0).mul(100).plot(kind="bar", stacked=True, figsize=(9, 4))
    ax.set_ylabel("% des instances"); ax.set_title("Repartition petit / moyen / grand par categorie")
    plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(out / "size_by_class.png", dpi=150); plt.close()

    # ---- densite
    fig, ax = plt.subplots(figsize=(8, 4))
    for s, g in I.groupby("split"):
        ax.hist(g.n_objects, bins=50, alpha=0.5, label=s, density=True)
    ax.set_xlabel("objets par image"); ax.set_ylabel("densite"); ax.set_title("Densite des scenes"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "objects_per_image.png", dpi=150); plt.close()

    # ---- occlusion / troncature
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    B["occlusion"].value_counts().sort_index().plot(kind="bar", ax=axes[0], title="Occlusion (0/1/2)")
    B["truncation"].value_counts().sort_index().plot(kind="bar", ax=axes[1], title="Troncature (0/1)")
    plt.tight_layout(); plt.savefig(out / "occlusion_truncation.png", dpi=150); plt.close()
    print(f"Figures et tableaux ecrits dans {out}/")


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
