"""Test de bout en bout sur un mini jeu de donnees synthetique au format VisDrone (sans GPU ni modele)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from visdet.evaluation import load_gt, run_eval

ROOT = Path(__file__).resolve().parents[1]


def _make_dataset(root: Path):
    for split, n in (("train", 3), ("val", 4)):
        base = root / f"VisDrone2019-DET-{split}"
        (base / "images").mkdir(parents=True)
        (base / "annotations").mkdir()
        rng = np.random.default_rng(0)
        for i in range(n):
            Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(base / "images" / f"{i:03d}.jpg")
            lines = ["10,10,20,30,1,4,0,0", "100,50,40,40,1,1,0,1", "5,5,10,10,0,5,0,0",   # score 0 -> ignoree
                     "200,200,50,50,0,0,0,0", "300,100,30,30,1,11,0,0",                  # zone ignoree / others
                     "380,280,60,60,1,9,1,2"]                                              # depasse de l'image -> rognee
            (base / "annotations" / f"{i:03d}.txt").write_text("\n".join(lines) + "\n")


def test_prepare_analyze_evaluate(tmp_path):
    root = tmp_path / "VisDrone"
    _make_dataset(root)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/prepare_data.py"), "--root", str(root),
                        "--splits", "train", "val"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lab = (root / "VisDrone2019-DET-val/labels/000.txt").read_text().strip().splitlines()
    assert len(lab) == 3                       # 3 boites utiles (cat 4, 1, 9)
    assert lab[0].split()[0] == "3"            # categorie 4 (car) -> classe 0-based 3
    assert (root / "visdrone.yaml").exists()

    gt_json = root / "coco" / "instances_val.json"
    images, gt = load_gt(str(gt_json))
    assert len(images) == 4 and all(len(im["ignore"]) == 1 for im in images.values())

    # detecteur "oracle" : renvoie exactement la verite terrain -> metriques parfaites
    by_name = {im["file_name"]: k for k, im in images.items()}

    # warmup (3 appels) consomme 3 images : on cale l'oracle en re-initialisant l'index apres l'echauffement
    class Oracle:
        def __init__(self):
            self.calls = 0
            self.names = sorted(by_name)

        def __call__(self, img):
            self.calls += 1
            idx = 0 if self.calls <= 3 else self.calls - 4   # echauffement -> image 0 ; puis images dans l'ordre
            k = by_name[self.names[idx % len(self.names)]]
            b, c = gt[k]
            return b.copy(), np.ones(len(b)), c.copy()

    res = run_eval(Oracle(), str(root / "VisDrone2019-DET-val/images"), str(gt_json), progress=False)
    assert res["precision"] == 1.0 and res["recall"] == 1.0
    assert res["mAP50-95"] > 0.99 and res["mAP50"] > 0.99

    # detecteur vide -> tout est nul
    res0 = run_eval(lambda img: (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)),
                    str(root / "VisDrone2019-DET-val/images"), str(gt_json), progress=False)
    assert res0["mAP50"] == 0 and res0["recall"] == 0

    r = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_dataset.py"), "--root", str(root),
                        "--out", str(tmp_path / "stats"), "--splits", "train", "val"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "stats" / "size_distribution.png").exists()
