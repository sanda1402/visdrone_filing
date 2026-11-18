#!/usr/bin/env python
"""Preparation des donnees VisDrone2019-DET.

A partir des annotations brutes VisDrone, produit :
  1. des etiquettes au format YOLO  : <split>/labels/<image>.txt  (classe cx cy w h normalises)
  2. un fichier COCO par split       : <root>/coco/instances_<split>.json (pour pycocotools)
     avec les zones ignorees ("ignored regions") stockees dans images[i]["ignore"]
  3. le fichier de configuration Ultralytics : <root>/visdrone.yaml

Regles de filtrage (identiques a celles du script officiel Ultralytics) :
  - categorie 0 (ignored regions) : jamais une cible, memorisee comme zone ignoree
  - categorie 11 (others) et boites avec score 0 : ignorees
  - categories 1..10 -> classes YOLO 0..9

Usage :
    python scripts/prepare_data.py --root datasets/VisDrone
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from visdet.data import CLASS_NAMES, SPLIT_DIRS, list_images, parse_annotation_file, split_paths  # noqa: E402
from visdet.utils import save_json  # noqa: E402


def convert_split(root: Path, split: str) -> dict:
    images_dir, ann_dir = split_paths(root, split)
    labels_dir = images_dir.parent / "labels"
    labels_dir.mkdir(exist_ok=True)

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]}
    ann_id = 1
    n_boxes = n_skipped = 0

    for img_id, img_path in enumerate(list_images(images_dir), start=1):
        with Image.open(img_path) as im:
            W, H = im.size
        rows = parse_annotation_file(ann_dir / (img_path.stem + ".txt"))
        yolo_lines, ignore = [], []
        for r in rows:
            if r["cat"] == 0:
                ignore.append([r["x"], r["y"], r["w"], r["h"]])
                continue
            if r["score"] == 0 or not (1 <= r["cat"] <= 10):
                n_skipped += 1
                continue
            # rognage aux limites de l'image
            x1, y1 = max(0.0, r["x"]), max(0.0, r["y"])
            x2, y2 = min(float(W), r["x"] + r["w"]), min(float(H), r["y"] + r["h"])
            w, h = x2 - x1, y2 - y1
            if w <= 1 or h <= 1:
                n_skipped += 1
                continue
            cls = r["cat"] - 1
            yolo_lines.append(f"{cls} {(x1 + w / 2) / W:.6f} {(y1 + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id, "category_id": r["cat"],
                "bbox": [x1, y1, w, h], "area": w * h, "iscrowd": 0,
                "truncation": r["trunc"], "occlusion": r["occ"]})
            ann_id += 1
            n_boxes += 1
        (labels_dir / (img_path.stem + ".txt")).write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))
        coco["images"].append({"id": img_id, "file_name": img_path.name, "width": W, "height": H, "ignore": ignore})

    out = root / "coco" / f"instances_{split}.json"
    save_json(coco, out)
    print(f"[{split}] {len(coco['images'])} images, {n_boxes} boites conservees, {n_skipped} ignorees -> {out}")
    return {"images": len(coco["images"]), "boxes": n_boxes, "skipped": n_skipped}


def write_yaml(root: Path, splits: list[str]) -> Path:
    lines = [f"path: {root.resolve()}"]
    for key, split in (("train", "train"), ("val", "val"), ("test", "test-dev")):
        if split in splits:
            lines.append(f"{key}: {SPLIT_DIRS[split]}/images")
    lines.append("names:")
    lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    out = root / "visdrone.yaml"
    out.write_text("\n".join(lines) + "\n")
    print(f"Configuration Ultralytics ecrite : {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="datasets/VisDrone", help="dossier contenant VisDrone2019-DET-*")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test-dev"], choices=list(SPLIT_DIRS))
    args = ap.parse_args()

    root = Path(args.root)
    done, stats = [], {}
    for split in args.splits:
        if not (root / SPLIT_DIRS[split] / "images").is_dir():
            print(f"[{split}] dossier introuvable, ignore ({root / SPLIT_DIRS[split]})")
            continue
        stats[split] = convert_split(root, split)
        done.append(split)
    if not done:
        sys.exit("Aucun split trouve. Lancez d'abord scripts/download_data.sh")
    write_yaml(root, done)
    save_json(stats, root / "prepare_stats.json")


if __name__ == "__main__":
    main()
