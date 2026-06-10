# CLAUDE.md — Instructions for Claude Code

## Project
AIMCON_JMI_26: Metadata Shortcut Learning in Multimodal Skin Lesion Classification
Branch: v1 | Repo: https://github.com/MAUK9086/AIMCON_JMI_26

## Critical Hardware Notes
- GPU: NVIDIA RTX Pro 4000 Blackwell (sm_120, 24GB VRAM)
- Required: PyTorch 2.7.0+cu128 — DO NOT use any older PyTorch version
- CUDA 12.8 required — check with torch.version.cuda before running
- If CUDA not available: raise an error with instructions, do not silently fall to CPU

## Code Style
- All scripts take config path as argument: python exp01.py --config config/config.yaml
- All randomness seeded from config.project.seed
- All outputs (models, figures, tables) saved to paths defined in config
- Every script prints: [EXP_NAME] Starting... and [EXP_NAME] Done. Saved to: <path>
- Use tqdm for all loops that take > 5 seconds
- Use torch.amp.autocast for mixed precision training (faster on Blackwell)

## Experiment Order
Run run_all.sh which calls: exp09 → exp01 → exp02 → exp03 → exp04 → exp05 → exp06 → exp07 → exp08
(exp09 is correlation analysis — no training needed, fast, run first)

## Do Not
- Do not hardcode paths — always read from config.yaml
- Do not retrain exp01 models inside exp02–exp09 — load from outputs/models/
- Do not drop NaN rows from metadata — impute as per config
- Do not use DataParallel — single GPU only

## Git
- All code goes to branch v1
- Commit after each experiment script is written and passes tests/test_pipeline.py
