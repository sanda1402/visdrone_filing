#!/usr/bin/env python
"""Entrainement / adaptation (fine-tuning) de YOLO ou RT-DETR sur VisDrone avec Ultralytics.

Toute la configuration est dans un fichier YAML (configs/train_*.yaml) : les cles `kind` et
`weights` choisissent le modele, toutes les autres sont transmises telles quelles a
`model.train(...)`. La configuration effective et les versions des bibliotheques sont enregistrees
a cote des poids (effective_config.json) pour documenter l'experience.

Usage :
    python scripts/train.py --config configs/train_yolo.yaml
    python scripts/train.py --config configs/train_rtdetr.yaml
    python scripts/train.py --config configs/train_yolo.yaml --set epochs=2 batch=8 fraction=0.05   # test rapide
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from visdet.models import load_model  # noqa: E402
from visdet.utils import env_info, save_json, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="CLE=VALEUR",
                    help="surcharge de parametres du YAML (ex: epochs=5 batch=8)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)

    kind, weights = cfg.pop("kind"), cfg.pop("weights")
    cfg["project"] = str(Path(cfg.get("project", "runs")).resolve())
    cfg["data"] = str(Path(cfg["data"]).resolve())
    set_seed(int(cfg.get("seed", 0)))

    model = load_model(kind, weights)
    effective = {"kind": kind, "weights": weights, **cfg, "environment": env_info()}
    print("Configuration effective :")
    print(yaml.safe_dump({k: v for k, v in effective.items() if k != "environment"}, sort_keys=False))

    model.train(**cfg)

    save_dir = Path(model.trainer.save_dir)
    save_json(effective, save_dir / "effective_config.json")
    print(f"\nEntrainement termine. Resultats : {save_dir}")
    print(f"Meilleurs poids : {save_dir / 'weights' / 'best.pt'}")
    print(f"Courbes d'apprentissage : {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
