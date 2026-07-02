#!/usr/bin/env bash
# Train nocs_sceneholdout_ss_ratio_v1
#
# Adds the SS decoder aspect-ratio loss on top of the SLAT-conditioned v3 setup.
# New flags vs v3:
#   --unfreeze-ss-decoder         unfreezes SS decoder (73M params) for ratio-loss gradient
#   --ss-ratio-loss-weight 0.05   weight on scale-invariant voxel aspect ratio loss
#
# Warm-start: v3_best.pt (1.74% MAPE)
#
# Monitor: tail -f /tmp/ss_ratio_v1.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

LOG=artifacts/metric_scale/logs/ss_ratio_v1.log
mkdir -p artifacts/metric_scale/logs

setsid -f bash -lc '
  cd /mnt/source/sam-3d-objects
  export LIDRA_SKIP_INIT=1
  export ATTN_BACKEND=sdpa
  export SPARSE_ATTN_BACKEND=sdpa
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec /opt/conda/envs/sam3d/bin/python \
    sam3d_objects/training/finetune_metric_scale.py \
    --config checkpoints/hf/pipeline.yaml \
    --dataset omninocs-nocs-real275 \
    --omninocs-root /mnt/source/datasets_sam3d/OmniNOCS \
    --omninocs-sources nocs_real275 \
    --annotations-root /mnt/source/datasets_sam3d/OmniNOCS/omninocs_release_nocs_real275 \
    --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
    --skip-missing-rgb \
    --split-group record \
    --split train \
    --overfit-samples 0 \
    --heldout-samples 64 \
    --seed 0 \
    --epochs 10 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --unfreeze-ss-decoder \
    --ss-ratio-loss-weight 0.05 \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --load-checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/nocs_sceneholdout_ss_ratio_v1.pt \
    --best-output artifacts/metric_scale/checkpoints/nocs_sceneholdout_ss_ratio_v1_best.pt \
    --metrics-output artifacts/metric_scale/metrics/nocs_sceneholdout_ss_ratio_v1_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/nocs_sceneholdout_ss_ratio_v1_manifest.json \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name nocs_sceneholdout_ss_ratio_v1
' > "$LOG" 2>&1 < /dev/null &

echo "Launched PID $! — logs at $LOG"
