#!/bin/bash
# Master script — runs all experiments sequentially
# Usage: bash experiments/run_all.sh
# Run from repo root

set -e  # Exit on any error

CONFIG="config/config.yaml"

# Detect python binary (conda envs on Windows may only have 'python', not 'python3')
if command -v python &> /dev/null; then
    PYTHON=python
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "ERROR: python not found. Activate your conda environment first."
    exit 1
fi

echo "============================================"
echo "AIMCON_JMI_26 — Full Experiment Pipeline"
echo "============================================"

# Verify GPU
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA not available — ensure PyTorch 2.7.0+cu128 is installed'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Run in logical order
$PYTHON experiments/exp09_correlation.py --config $CONFIG
$PYTHON experiments/exp01_baseline.py --config $CONFIG
$PYTHON experiments/exp02_occlusion.py --config $CONFIG
$PYTHON experiments/exp03_gradients.py --config $CONFIG
$PYTHON experiments/exp04_subgroup.py --config $CONFIG
$PYTHON experiments/exp05_metadata_dropout.py --config $CONFIG
$PYTHON experiments/exp06_class_reweighting.py --config $CONFIG
$PYTHON experiments/exp07_age_stratified.py --config $CONFIG
$PYTHON experiments/exp08_site_stratified.py --config $CONFIG
$PYTHON experiments/exp10_bootstrap_ci.py --config $CONFIG
$PYTHON experiments/exp11_modality_reliance.py --config $CONFIG
$PYTHON experiments/exp12_occlusion_confusion.py --config $CONFIG
$PYTHON experiments/exp13_metadata_collapse.py --config $CONFIG
$PYTHON experiments/exp14_power_analysis.py --config $CONFIG
$PYTHON experiments/exp12b_scc_profile.py --config $CONFIG
$PYTHON experiments/exp12c_fallback_comparison.py --config $CONFIG
$PYTHON experiments/exp13b_sink_analysis.py --config $CONFIG
$PYTHON experiments/exp13c_metadata_space.py --config $CONFIG

echo "============================================"
echo "All experiments complete."
echo "Results in: outputs/"
echo "============================================"
