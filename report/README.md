# Dossier `report/`

Placez ici le rapport final : **`report/rapport.pdf`** (10 à 15 pages).

## Plan conseillé (et nombre de pages indicatif)

| Section | Pages | Contenu |
|---|---|---|
| 1. Introduction | 1 | Contexte (images de drone, petits objets), problématique, hypothèse, contributions |
| 2. État de l'art | 1,5-2 | YOLO (détecteurs à une étape, FPN/PAN), RT-DETR (transformer temps réel), VisDrone, détection par tuiles (SAHI), fusion (NMS, WBF) |
| 3. Données | 1,5-2 | VisDrone2019-DET : splits, 10 classes, annotations, **figures de `results/dataset_stats/`** (distribution des tailles, % petits objets, densité, occlusion) |
| 4. Méthodologie | 2-3 | Configurations d'entraînement (copier `configs/`), stratégie de tiling (algorithme, paramètres, fusion), protocole d'évaluation (métriques, temps) |
| 5. Résultats | 2-3 | Tableau `results/comparison.md`, courbes d'apprentissage, exemples visuels `results/examples/` |
| 6. Ablation | 1-2 | Tableau `results/ablation_table.md`, courbes `results/figures/ablation_*.png`, interprétation |
| 7. Discussion et limites | 1 | Coût d'inférence, objets coupés, sensibilité aux seuils, un seul jeu de données, une seule graine |
| 8. Conclusion et perspectives | 0,5 | Réponse à la problématique, pistes |
| Références | 0,5 | Toutes les ressources externes citées |

**Règle d'or :** chaque chiffre du rapport doit provenir d'un fichier de `results/` produit par les scripts du dépôt.
Citez le commit exact utilisé (`git rev-parse --short HEAD`) et le matériel (`results/metrics/*.json`, clé `environment`).
