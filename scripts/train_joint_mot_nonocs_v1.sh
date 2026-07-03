#!/usr/bin/env bash
# Cross-dataset-transfer training (2026-07-03): joint layout SFT with NOCS-Real275
# HELD OUT entirely, to give an apples-to-apples row against OmniNOCS Table 5's
# transfer protocol (NOCSformer 43.5/10.6, CubeRCNN 14.9/4.1 — neither trained on
# NOCS). Sources: objectron + arkitscenes only (~6.7k records). Uses the FIXED
# modality routing (shape transformer frozen, full layout transformer trained) at
# the validated ss-pose-lr 1e-5 — the best-known pose recipe, NOT joint_mot_v1's
# buggy-routing 1e-4. Evaluate mAP on NOCS-Real275 afterwards (never seen).
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
  --overfit-samples 0 --seed 0 --epochs 6 \
  --eval-every 1 --eval-every-steps 2000 --eval-steps-max-samples 150 \
  --checkpoint-every-steps 1000 --checkpoint-every 1 --log-step-every 1 \
  --unfreeze-ss-backbone --ss-geometry-lr 0 --ss-pose-lr 1e-5 \
  --trans-loss-weight 1.0 --flow-loss-weight 0.0 \
  --weight-decay 1e-4 --stage1-steps 4 --batch-size 1 \
  --output artifacts/metric_scale/checkpoints/joint_mot_nonocs_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/joint_mot_nonocs_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/joint_mot_nonocs_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/joint_mot_nonocs_v1_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name joint_mot_nonocs_v1
