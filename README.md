# AIMCON-JMI-2026: Metadata-Driven Shortcut Learning in Multimodal Skin Lesion Classification

**Target Venue:** AIM-CON 2026, Springer CCIS — Track 1 (Multimodal AI + Healthcare)  
**Conference Dates:** 31 Oct – 1 Nov 2026

## Research Question

Can multimodal skin lesion classifiers achieve high aggregate accuracy primarily through
demographic shortcuts (age, sex, anatomical site) rather than visual understanding — and
does this failure mode go undetected under standard evaluation metrics used by regulatory
bodies such as FDA and CDSCO?

## Setup

```bash
conda create -n aimcon python=3.11 -y
conda activate aimcon
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Dataset

```bash
# Requires Kaggle API credentials in ~/.kaggle/kaggle.json
python data/download_isic2019.py
```

## Run All Experiments

```bash
bash experiments/run_all.sh
```

Results are saved to `outputs/`.

## Results

ISIC 2019, 8 classes, held-out test set (n = 3,801). Macro-averaged over classes.

**Baselines (exp01)**

| Model | Mean sensitivity | Macro F1 | Macro AUC |
|---|---|---|---|
| Image only | 0.774 | 0.748 | 0.967 |
| Metadata only (age, sex, site) | 0.317 | 0.169 | 0.709 |
| Image + metadata | 0.795 | 0.763 | 0.969 |

**Occlusion test on the fusion model (exp02)**

| Condition | Mean sensitivity | Macro F1 | Macro AUC |
|---|---|---|---|
| Nothing occluded | 0.795 | 0.763 | 0.969 |
| Image occluded | 0.163 | 0.053 | 0.618 |
| Metadata occluded | 0.784 | 0.780 | 0.969 |

Removing the image collapses the fusion model; removing the metadata barely moves the aggregate numbers. Metadata alone is weak (macro AUC 0.71) but above chance, and metadata is clearly associated with the label (age vs malignancy, point-biserial r = 0.37). Per-class reliance, subgroup and fairness results (exp03–exp14) are in `outputs/tables/`.

## Repository Structure

```
AIMCON_JMI_26/
├── config/config.yaml          # All hyperparameters and paths
├── data/                       # Dataset download, Dataset class, preprocessing
├── models/                     # image_only, metadata_only, fusion architectures
├── training/                   # Training loop, losses, callbacks
├── experiments/                # 9 experiment scripts + run_all.sh
├── evaluation/                 # Metrics, subgroup eval, fairness
├── outputs/                    # Auto-saved results, figures, tables
├── paper/                      # results_summary.md
└── tests/                      # Sanity checks
```
