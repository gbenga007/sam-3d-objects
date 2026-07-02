#!/usr/bin/env bash
# Joint-MoT metric-reconstruction training (2026-06-17). Unfreezes the SS generator (MoT):
# the pose decoder predicts metric W,H,D + translation directly (NO MetricScaleHead/scale-token).
# MoGe-2 metric pointmaps (deployment depth source). Discriminative LR (geom slow / pose fast) +
# L2-SP geometry anchor. Warm-started from the pretrained SAM3D ss_generator by construction.
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
  --unfreeze-ss-backbone --ss-geometry-lr 2e-6 --ss-pose-lr 1e-4 \
  --trans-loss-weight 1.0 --flow-loss-weight 10.0 \
  --weight-decay 1e-4 --stage1-steps 4 --batch-size 1 \
  --output artifacts/metric_scale/checkpoints/joint_mot_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/joint_mot_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/joint_mot_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/joint_mot_v1_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name joint_mot_v1
