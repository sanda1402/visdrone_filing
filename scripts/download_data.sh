#!/usr/bin/env bash
# Telechargement de VisDrone2019-DET et des poids pre-entraines.
#
# Source officielle du jeu de donnees : https://github.com/VisDrone/VisDrone-Dataset
# (liens Google Drive / Baidu). Les memes archives sont miroir sur les "releases" du depot
# Ultralytics (utilisees par leur fichier VisDrone.yaml) : c'est ce miroir qui est utilise ici,
# car il se telecharge en ligne de commande sans compte.
#
# Usage :  bash scripts/download_data.sh [dossier_donnees]
set -euo pipefail

ROOT="${1:-datasets/VisDrone}"
BASE="https://github.com/ultralytics/assets/releases/download/v0.0.0"
mkdir -p "$ROOT"
cd "$ROOT"

for NAME in VisDrone2019-DET-train VisDrone2019-DET-val VisDrone2019-DET-test-dev; do
  if [ -d "$NAME/images" ]; then
    echo "[ok] $NAME deja present"
    continue
  fi
  echo "[..] telechargement de $NAME.zip"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail -o "$NAME.zip" "$BASE/$NAME.zip"
  else
    wget -O "$NAME.zip" "$BASE/$NAME.zip"
  fi
  unzip -q -o "$NAME.zip" && rm -f "$NAME.zip"
done

echo
echo "Donnees pretes dans $ROOT"
echo "Si le telechargement echoue, recuperez les archives depuis le depot officiel"
echo "(https://github.com/VisDrone/VisDrone-Dataset) et decompressez-les dans $ROOT."
echo
echo "Poids pre-entraines (COCO) : telecharges automatiquement par Ultralytics au premier"
echo "lancement (yolov8s.pt, rtdetr-l.pt). Sources : https://github.com/ultralytics/assets/releases"
echo "Suite : python scripts/prepare_data.py --root $ROOT"
