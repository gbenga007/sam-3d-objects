#!/usr/bin/env bash
# Unified-recipe cross-dataset-transfer run (2026-07-03). ONE architecture for the
# whole paper: the live_v2 winning recipe (MoGe-2 anchor, MetricScaleHead + SLAT
# cross-attn W,H,D readout, layout transformer unfrozen @1e-5 via fixed routing)
# EXTENDED with joint_mot's translation loss (weight 1.0) so the pose pathway
# trains alongside the dims head instead of drifting. NOCS-Real275 is HELD OUT
# entirely (train = objectron + arkitscenes) for an apples-to-apples mAP transfer
# row against OmniNOCS Table 5 (NOCSformer 43.5/10.6, CubeRCNN 14.9/4.1).
# 8 epochs (~39h) to fit the July 6 deadline; intra-epoch best checkpoints.
set -euo pipefail
export LIDRA_SKIP_INIT=true
cd /mnt/source/sam-3d-objects

CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
  sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml --dataset omninocs-mixed \
  --omninocs-sources objectron arkitscenes \
  --omninocs-root /mnt/source/datasets_sam3d/OmniNOCS \
  --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
  --objectron-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --arkitscenes-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --skip-missing-rgb \
  --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps \
  --max-records-per-source 3334 --balanced-sampling \
  --heldout-per-source objectron:200,arkitscenes:200 \
  --overfit-samples 0 --seed 0 --epochs 8 \
  --eval-every 1 --eval-every-steps 2000 --eval-steps-max-samples 150 \
  --checkpoint-every-steps 1000 --checkpoint-every 1 --log-step-every 1 \
  --lr 1e-4 --slat-lr 1e-6 --slat-lr-warmup-steps 1500 \
  --ss-pose-lr 1e-5 --weight-decay 1e-4 \
  --stage1-steps 4 --stage2-steps 1 --batch-size 1 \
  --unfreeze-slat-cross-attn --unfreeze-ss-layout-for-head --no-ss-grad-checkpoint \
  --trans-loss-weight 1.0 \
  --p-uncond-scale-token 0.1 \
  --output artifacts/metric_scale/checkpoints/unified_nonocs_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/unified_nonocs_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/unified_nonocs_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/unified_nonocs_v1_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name unified_nonocs_v1
