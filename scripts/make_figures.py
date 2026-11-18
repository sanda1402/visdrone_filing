#!/usr/bin/env python
"""Genere les tableaux comparatifs et graphiques du rapport a partir des resultats reels.

  - results/comparison.md / comparison.csv : tableau YOLO vs RT-DETR vs YOLO+tiling (summary.csv)
  - results/figures/comparison_bars.png    : barres mAP50, mAP50-95, AP_small, Recall
  - results/figures/speed_accuracy.png     : AP_small en fonction du temps d'inference
  - results/figures/learning_curves.png    : courbes d'apprentissage (results.csv d'Ultralytics)
  - results/figures/ablation_*.png         : courbes d'ablation (ablation.csv)

Usage :
    python scripts/make_figures.py \
        --runs YOLO=runs/yolov8s_visdrone_640/results.csv RT-DETR=runs/rtdetr_l_visdrone_640/results.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLS = ["name", "precision", "recall", "f1", "mAP50", "mAP50-95", "AP_small", "AP_medium", "AP_large",
        "ms_per_img_mean", "fps"]


def comparison(res: Path, fig: Path):
    f = res / "summary.csv"
    if not f.exists():
        print("summary.csv absent : lancez scripts/evaluate.py d'abord"); return
    df = pd.read_csv(f)
    t = df[[c for c in COLS if c in df.columns]].round(4)
    t.to_csv(res / "comparison.csv", index=False)
    try:
        md = t.to_markdown(index=False)
    except ImportError:
        md = t.to_string(index=False)
    (res / "comparison.md").write_text(md + "\n")
    print(md)

    metrics = [m for m in ["mAP50", "mAP50-95", "AP_small", "recall"] if m in t.columns]
    ax = t.set_index("name")[metrics].plot(kind="bar", figsize=(max(7, 1.4 * len(t)), 4))
    ax.set_ylabel("score"); ax.set_title("Comparaison des methodes (VisDrone val)")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(fig / "comparison_bars.png", dpi=150); plt.close()

    fig2, ax = plt.subplots(figsize=(6, 4))
    for _, r in t.iterrows():
        ax.scatter(r["ms_per_img_mean"], r["AP_small"], s=60)
        ax.annotate(r["name"], (r["ms_per_img_mean"], r["AP_small"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("temps d'inference moyen (ms / image)"); ax.set_ylabel("AP petits objets")
    ax.set_title("Compromis precision / temps"); plt.tight_layout()
    plt.savefig(fig / "speed_accuracy.png", dpi=150); plt.close()


def learning_curves(runs: list[str], fig: Path):
    if not runs:
        return
    series = {}
    for item in runs:
        label, path = item.split("=", 1)
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        series[label] = df
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, df in series.items():
        tl = [c for c in df.columns if c.startswith("train/") and "loss" in c]
        vl = [c for c in df.columns if c.startswith("val/") and "loss" in c]
        axes[0].plot(df["epoch"], df[tl].sum(axis=1), label=label)
        axes[1].plot(df["epoch"], df[vl].sum(axis=1), label=label)
        if "metrics/mAP50-95(B)" in df:
            axes[2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label=label)
    for ax, title in zip(axes, ["Loss d'entrainement (somme)", "Loss de validation (somme)", "mAP@0.5:0.95 (validation)"]):
        ax.set_title(title); ax.set_xlabel("epoque"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(fig / "learning_curves.png", dpi=150); plt.close()
    print("learning_curves.png ecrit")


def ablation(res: Path, fig: Path):
    f = res / "ablation.csv"
    if not f.exists():
        print("ablation.csv absent : lancez scripts/ablation.py d'abord"); return
    df = pd.read_csv(f)
    show = [c for c in ["name", "tile", "overlap", "include_full", "border_margin", "precision", "recall", "mAP50",
                        "mAP50-95", "AP_small", "ms_per_img_mean"] if c in df.columns]
    t = df[show].round(4).sort_values([c for c in ["tile", "overlap", "include_full", "border_margin"] if c in df.columns])
    t.to_csv(res / "ablation_table.csv", index=False)
    try:
        (res / "ablation_table.md").write_text(t.to_markdown(index=False) + "\n")
    except ImportError:
        (res / "ablation_table.md").write_text(t.to_string(index=False) + "\n")

    for varying, other in (("overlap", "tile"), ("tile", "overlap")):
        if df[varying].nunique() < 2:
            continue
        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
        for key, g in df.groupby(other):
            g = g.sort_values(varying)
            for ax, m in zip(axes, ["AP_small", "mAP50-95", "ms_per_img_mean"]):
                ax.plot(g[varying], g[m], marker="o", label=f"{other}={key}")
        for ax, m in zip(axes, ["AP petits objets", "mAP@0.5:0.95", "temps (ms / image)"]):
            ax.set_xlabel(varying); ax.set_ylabel(m); ax.grid(alpha=0.3)
        axes[0].legend(); plt.tight_layout()
        plt.savefig(fig / f"ablation_{varying}.png", dpi=150); plt.close()
        print(f"ablation_{varying}.png ecrit")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", default=[], metavar="LABEL=results.csv")
    args = ap.parse_args()
    res = Path(args.results)
    fig = res / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    comparison(res, fig)
    learning_curves(args.runs, fig)
    ablation(res, fig)


if __name__ == "__main__":
    main()
