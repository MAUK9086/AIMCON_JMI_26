#!/bin/bash
# Master script — runs all experiments sequentially
# Usage: bash experiments/run_all.sh
# Run from repo root

set -e  # Exit on any error

CONFIG="config/config.yaml"

echo "============================================"
echo "AIMCON_JMI_26 — Full Experiment Pipeline"
echo "============================================"

# Verify GPU
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — ensure PyTorch 2.7.0+cu128 is installed'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run in logical order
python experiments/exp09_correlation.py --config $CONFIG
python experiments/exp01_baseline.py --config $CONFIG
python experiments/exp02_occlusion.py --config $CONFIG
python experiments/exp03_gradients.py --config $CONFIG
python experiments/exp04_subgroup.py --config $CONFIG
python experiments/exp05_metadata_dropout.py --config $CONFIG
python experiments/exp06_class_reweighting.py --config $CONFIG
python experiments/exp07_age_stratified.py --config $CONFIG
python experiments/exp08_site_stratified.py --config $CONFIG

echo "============================================"
echo "All experiments complete."
echo "Results in: outputs/"
echo "============================================"
