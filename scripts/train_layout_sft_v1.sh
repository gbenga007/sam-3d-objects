#!/usr/bin/env bash
# Layout-only native SFT (2026-06-23). MoT modality-aware unfreeze: FREEZE the shape transformer
# + shared adaLN/t_embedder (geometry preserved BY CONSTRUCTION — paper §C.2 "freeze shape,
# finetune layout"); TRAIN the FULL layout transformer (.6drotation_normalized.-keyed per-block
# norms/self-attn/mlp/cross_attn + the scale/translation/rotation read heads) at the Table-5 SFT
# LR 1e-5. Fixes the joint_mot_v1 bug where only the ~0.03M read heads trained fast while the
# layout transformer crawled at 2e-6. No L2-SP needed (shape frozen). Dimensions-focused:
# W,H,D loss weight 1.0, translation co-trained (corroboration). MoGe-2 metric pointmaps.
# Clean A/B vs joint_mot_v1: same recipe/data/warm-start, ONLY the param-group routing + LR change.
set -euo pipefail
export LIDRA_SKIP_INIT=true
cd /mnt/source/sam-3d-objects

CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
  sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml --dataset omninocs-mixed \
  --omninocs-sources nocs_real275 objectron arkitscenes \
  --omninocs-root /mnt/source/datasets_sam3d/OmniNOCS \
  --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
  --objectron-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --arkitscenes-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --skip-missing-rgb \
  --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps \
  --max-records-per-source 3334 --balanced-sampling \
  --heldout-per-source nocs_real275:64,objectron:200,arkitscenes:200 \
  --overfit-samples 0 --seed 0 --epochs 6 \
  --eval-every 1 --eval-every-steps 2000 --eval-steps-max-samples 150 \
  --checkpoint-every-steps 1000 --checkpoint-every 1 --log-step-every 1 \
  --unfreeze-ss-backbone --unfreeze-ss-cross-attn \
  --ss-geometry-lr 0 --ss-pose-lr 1e-5 --ss-cond-lr 1e-5 \
  --trans-loss-weight 1.0 --flow-loss-weight 0.0 \
  --weight-decay 1e-4 --stage1-steps 4 --batch-size 1 \
  --output artifacts/metric_scale/checkpoints/layout_sft_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/layout_sft_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/layout_sft_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/layout_sft_v1_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name layout_sft_v1
