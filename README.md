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
