#!/usr/bin/env bash
# Pipeline complet : donnees -> analyse -> entrainements -> evaluations -> ablation -> figures.
# Duree : plusieurs heures sur GPU (surtout les deux entrainements). Pour un essai rapide :
#   QUICK=1 bash scripts/run_all.sh     (5% des donnees, 2 epoques, 50 images d'evaluation)
set -euo pipefail

ROOT=datasets/VisDrone
YOLO_RUN=runs/yolov8s_visdrone_640
DETR_RUN=runs/rtdetr_l_visdrone_640
EXTRA_TRAIN=""; LIMIT=""
if [ "${QUICK:-0}" = "1" ]; then
  EXTRA_TRAIN="--set epochs=2 fraction=0.05 name=quick_${RANDOM}"; LIMIT="--limit 50"
  YOLO_RUN=""; DETR_RUN=""
fi

bash scripts/download_data.sh "$ROOT"
python scripts/prepare_data.py --root "$ROOT"
python scripts/analyze_dataset.py --root "$ROOT"

if [ -z "$YOLO_RUN" ]; then echo "Mode QUICK : renseignez les poids a la main (voir README)"; exit 0; fi

python scripts/train.py --config configs/train_yolo.yaml   $EXTRA_TRAIN
python scripts/train.py --config configs/train_rtdetr.yaml $EXTRA_TRAIN

Y=$YOLO_RUN/weights/best.pt; D=$DETR_RUN/weights/best.pt
python scripts/evaluate.py --kind yolo   --weights $Y --mode full  --imgsz 640  --name yolo_full640    $LIMIT
python scripts/evaluate.py --kind yolo   --weights $Y --mode full  --imgsz 1280 --name yolo_full1280   $LIMIT
python scripts/evaluate.py --kind rtdetr --weights $D --mode full  --imgsz 640  --name rtdetr_full640  $LIMIT
python scripts/evaluate.py --kind yolo   --weights $Y --mode tiled --tile 640 --overlap 0.2 --name yolo_tiled640_ov20 $LIMIT

python scripts/ablation.py --kind yolo --weights $Y --tiles 640 --overlaps 0 0.1 0.2 0.3 $LIMIT
python scripts/ablation.py --kind yolo --weights $Y --tiles 320 512 1024 --overlaps 0.2 $LIMIT

python scripts/compare_examples.py --yolo $Y --rtdetr $D --n 6
python scripts/make_figures.py --runs YOLO=$YOLO_RUN/results.csv RT-DETR=$DETR_RUN/results.csv
