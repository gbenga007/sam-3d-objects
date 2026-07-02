#!/usr/bin/env bash
# Native random-tau flow-matching layout SFT (2026-06-23). The paper's L_CFM (§C.2): noise every
# modality at ONE random tau, single forward, velocity MSE on the LAYOUT modalities (scale=W,H,D +
# translation). Shape+rotation x1 = the model's own frozen self-prediction (no GT mesh / rotation).
# Modality-aware unfreeze: FREEZE shape transformer + shared adaLN/t_embedder, TRAIN the full layout
# transformer (collect_ss_backbone_params). Cheaper than sample-then-regress (1 grad forward vs 4-step
# sampling backprop) AND pretraining-consistent. Table-5 SFT LR 1e-5. MoGe-2 metric pointmaps.
# Eval = predict_pose_metric (real inference W,H,D) every 2000 steps.
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
  --native-fm --fm-shape-steps 4 --fm-scale-weight 1.0 --fm-trans-weight 1.0 \
  --ss-geometry-lr 0 --ss-pose-lr 1e-5 --ss-cond-lr 1e-5 \
  --flow-loss-weight 0.0 --weight-decay 1e-4 --batch-size 1 \
  --output artifacts/metric_scale/checkpoints/native_fm_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/native_fm_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/native_fm_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/native_fm_v1_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name native_fm_v1
