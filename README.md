# Détection d'objets dans les images aériennes : YOLO vs RT-DETR et amélioration multi-échelle par *Image Tiling*

Projet de rattrapage - Master 1 Intelligence Artificielle et Big Data - Computer Vision
Auteur : **RASOAMAROMAKA Sanda Nomen'Ny Avo Michaël**

## 1. Problématique

Dans les images prises par drone, les objets (piétons, vélos, voitures...) ne mesurent souvent que quelques dizaines de pixels
sur des images d'environ 2000 x 1500 px. Une fois l'image redimensionnée à l'entrée du réseau (par ex. 640 px), ils deviennent
minuscules et les détecteurs les ratent.

> **Question :** dans quelle mesure une stratégie multi-échelle fondée sur le découpage des images en tuiles avec chevauchement
> améliore-t-elle la détection des petits objets, et quel est son impact sur la précision et le temps d'inférence ?

Ce dépôt :
1. analyse **VisDrone2019-DET** (annotations, catégories, distribution des tailles) ;
2. entraîne / adapte **YOLO** et **RT-DETR** sur les mêmes données (configurations documentées) ;
3. implémente la **méthode proposée** : découpage en tuiles avec chevauchement + fusion des prédictions (NMS par classe) ;
4. compare les trois approches (Precision, Recall, mAP@0.5, mAP@0.5:0.95, AP petits objets, temps d'inférence) ;
5. réalise une **étude d'ablation** (taille de tuile, chevauchement, passe image entière, marge de bord).

## 2. Organisation du dépôt

```
visdrone-tiling/
├── README.md                  # ce fichier
├── requirements.txt           # dépendances et versions minimales
├── configs/
│   ├── train_yolo.yaml        # hyperparamètres d'entraînement YOLO
│   ├── train_rtdetr.yaml      # hyperparamètres d'entraînement RT-DETR
│   └── tiling.yaml            # paramètres par défaut du tiling
├── visdet/                    # bibliothèque du projet
│   ├── data.py                # constantes VisDrone, lecture des annotations
│   ├── tiling.py              # DECOUPAGE EN TUILES + FUSION (méthode proposée)
│   ├── metrics.py             # Precision/Recall, AP COCO (pycocotools)
│   ├── evaluation.py          # protocole d'évaluation unifié + chronométrage
│   ├── models.py              # chargement YOLO / RT-DETR, fonctions de prédiction
│   ├── viz.py, utils.py       # visualisation, versions/seed/CSV
├── scripts/
│   ├── download_data.sh       # téléchargement des données et sources des poids
│   ├── prepare_data.py        # annotations VisDrone -> YOLO + COCO
│   ├── analyze_dataset.py     # analyse des annotations, classes, tailles
│   ├── train.py               # entraînement (YOLO ou RT-DETR)
│   ├── infer.py               # inférence image entière / tuiles
│   ├── evaluate.py            # évaluation (mode full ou tiled)
│   ├── ablation.py            # étude d'ablation
│   ├── compare_examples.py    # exemples visuels côte à côte
│   ├── make_figures.py        # tableaux, courbes d'apprentissage, courbes d'ablation
│   └── run_all.sh             # pipeline complet
├── tests/                     # tests unitaires et test de bout en bout (sans GPU)
├── results/                   # tableaux, courbes, métriques JSON (générés)
├── report/                    # rapport PDF (10-15 pages) : report/rapport.pdf
└── datasets/                  # données (non versionnées)
```

## 3. Installation

Prérequis : Python 3.10+, un GPU NVIDIA conseillé (Google Colab / Kaggle conviennent ; le CPU ne permet que des essais).

```bash
git clone <URL_DE_VOTRE_DEPOT> && cd visdrone-tiling
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
# 1) installer PyTorch adapté à votre GPU : https://pytorch.org/get-started/locally/
pip install -r requirements.txt
pip freeze > requirements-lock.txt                        # versions exactes utilisées (à committer)
python -c "from visdet.utils import env_info; print(env_info())"
pytest -q                                                 # vérifie le code sans GPU ni données
```

**Google Colab :** *Exécution > Modifier le type d'exécution > GPU*, puis `!git clone ...`, `%cd visdrone-tiling`, `!pip install -r requirements.txt`.

## 4. Données et poids pré-entraînés (sources et procédures)

| Ressource | Source | Procédure |
|---|---|---|
| VisDrone2019-DET (train 6471 / val 548 / test-dev 1610 images)* | https://github.com/VisDrone/VisDrone-Dataset (Zhu et al.) | `bash scripts/download_data.sh` (miroir des archives sur les *releases* Ultralytics) ou téléchargement manuel depuis le dépôt officiel, puis décompression dans `datasets/VisDrone/` |
| YOLO : `yolov8s.pt` (pré-entraîné COCO) | https://github.com/ultralytics/assets/releases | téléchargé automatiquement par Ultralytics au 1er lancement |
| RT-DETR : `rtdetr-l.pt` (pré-entraîné COCO) | idem | idem |

\* effectifs à vérifier avec `results/dataset_stats/summary.json` généré par `analyze_dataset.py`.

Arborescence attendue après téléchargement :
`datasets/VisDrone/VisDrone2019-DET-{train,val,test-dev}/{images,annotations}/`

## 5. Exécution pas à pas

```bash
# 1. Étiquettes YOLO + fichiers COCO + datasets/VisDrone/visdrone.yaml
python scripts/prepare_data.py --root datasets/VisDrone

# 2. Analyse du jeu de données (figures et chiffres dans results/dataset_stats/)
python scripts/analyze_dataset.py --root datasets/VisDrone

# 3. Entraînement (mêmes données, même taille d'entrée, même nb d'époques)
python scripts/train.py --config configs/train_yolo.yaml
python scripts/train.py --config configs/train_rtdetr.yaml
#    essai rapide : ... --set epochs=2 fraction=0.05

# 4. Évaluation : mêmes images, mêmes métriques, même protocole de chronométrage
Y=runs/yolov8s_visdrone_640/weights/best.pt ; D=runs/rtdetr_l_visdrone_640/weights/best.pt
python scripts/evaluate.py --kind yolo   --weights $Y --mode full  --imgsz 640  --name yolo_full640
python scripts/evaluate.py --kind yolo   --weights $Y --mode full  --imgsz 1280 --name yolo_full1280   # baseline "haute résolution"
python scripts/evaluate.py --kind rtdetr --weights $D --mode full  --imgsz 640  --name rtdetr_full640
python scripts/evaluate.py --kind yolo   --weights $Y --mode tiled --tile 640 --overlap 0.2 --name yolo_tiled640_ov20

# 5. Ablation (chevauchement, taille de tuile, passe image entière, marge de bord)
python scripts/ablation.py --kind yolo --weights $Y --tiles 640 --overlaps 0 0.1 0.2 0.3
python scripts/ablation.py --kind yolo --weights $Y --tiles 320 512 1024 --overlaps 0.2
python scripts/ablation.py --kind yolo --weights $Y --tiles 640 --overlaps 0.2 --include-full 0 1 --border-margins 0 4

# 6. Exemples visuels, tableaux et courbes pour le rapport
python scripts/compare_examples.py --yolo $Y --rtdetr $D --n 6
python scripts/make_figures.py --runs YOLO=runs/yolov8s_visdrone_640/results.csv RT-DETR=runs/rtdetr_l_visdrone_640/results.csv
```
Tout enchaîner : `bash scripts/run_all.sh`. Inférence sur vos images : `python scripts/infer.py --help`.

## 6. Méthode proposée : tiling avec chevauchement (`visdet/tiling.py`)

1. **Découpage** : tuiles `tile x tile` avec chevauchement `overlap` (fraction) ; les dernières tuiles sont recalées sur le bord pour couvrir toute l'image.
2. **Détection** : le détecteur est lancé sur chaque tuile à sa résolution native (`imgsz = tile`) : un objet de 20 px reste un objet de 20 px au lieu d'être réduit d'un facteur 3 ; option `--include-full` : passe supplémentaire sur l'image entière pour les grands objets.
3. **Repositionnement** des boîtes dans le repère de l'image d'origine.
4. **Filtre de bord** (option `--border-margin`) : suppression des boîtes collées à un bord interne de tuile (probablement tronquées).
5. **Fusion** : NMS par classe (`--merge-iou`, 0,5 par défaut) pour éliminer les doublons des zones de chevauchement.

Coût : le nombre de tuiles croît avec la taille de l'image et avec `1/(1-overlap)^2`, donc le temps d'inférence aussi. L'étude d'ablation mesure ce compromis.

## 7. Protocole d'évaluation (identique pour toutes les méthodes)

- Jeu **VisDrone2019-DET-val** (les annotations de test-challenge ne sont pas publiques).
- Les prédictions dont le centre tombe dans une *ignored region* sont écartées ; boîtes `score = 0` et catégorie *others* ignorées.
- **AP COCO** (pycocotools, `maxDets = 500` car les scènes sont denses) : mAP@0.5, mAP@0.5:0.95, AP petits (< 32²), moyens, grands. Seuil de confiance 0,001.
- **Precision / Recall / F1** à IoU 0,5 et confiance 0,25 (appariement glouton par classe) ; le F1 maximal et son seuil sont aussi enregistrés.
- **Temps d'inférence** : moyenne par image, batch 1, décodage de l'image exclu, tiling et fusion **inclus**, 3 images d'échauffement, synchronisation GPU.
- Graine fixée (`--seed`), versions et GPU enregistrés dans chaque `results/metrics/*.json`.

## 8. Configurations utilisées

Voir `configs/train_yolo.yaml` et `configs/train_rtdetr.yaml` ; la configuration effective (et les versions des bibliothèques) est
sauvegardée dans `runs/<nom>/effective_config.json` à chaque entraînement.

| | YOLO | RT-DETR |
|---|---|---|
| Modèle initial | `yolov8s.pt` (COCO) | `rtdetr-l.pt` (COCO) |
| Entrée / époques / batch | 640 / 100 / 16 | 640 / 100 / 8 |
| Graine / arrêt précoce | 0 / patience 30 | 0 / patience 30 |

## 9. Résultats

> **À compléter après exécution.** Les tableaux sont générés par `scripts/make_figures.py`
> (`results/comparison.md`, `results/ablation_table.md`) ; ne recopiez que des chiffres produits par les scripts.

| Méthode | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | AP petits | ms / image |
|---|---|---|---|---|---|---|
| YOLO (640) | | | | | | |
| YOLO (1280) | | | | | | |
| RT-DETR (640) | | | | | | |
| YOLO + tiling | | | | | | |

Figures : `results/figures/` (comparaison, compromis précision/temps, courbes d'apprentissage, ablation) et `results/examples/`.
Analyse complète : `report/rapport.pdf`.

## 10. Limites connues

Voir le rapport : coût d'inférence proportionnel au nombre de tuiles, objets plus grands que la tuile fragmentés, sensibilité aux seuils de fusion,
comparaison dépendante du budget d'entraînement et du GPU, un seul jeu de données et une seule graine. Pistes : entraînement sur tuiles
(*fine-tuning* type SAHI), fusion WBF, tuiles adaptatives selon la densité.

## 11. Références et ressources externes

- Zhu P. et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI, 2021 - jeu VisDrone : https://github.com/VisDrone/VisDrone-Dataset
- Jocher G. et al., **Ultralytics** (YOLOv8, RT-DETR, entraînement) : https://github.com/ultralytics/ultralytics
- Zhao Y. et al., *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR), CVPR 2024
- Redmon J. et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016
- Akyon F. C. et al., *Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection* (SAHI), ICIP 2022 - inspiration de l'approche par tuiles
- Lin T.-Y. et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014 - métriques AP et tailles petit/moyen/grand ; bibliothèque `pycocotools`

*(Vérifiez les références complètes avant de les reporter dans le rapport.)*
