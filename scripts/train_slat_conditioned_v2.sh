#!/usr/bin/env bash
# Train nocs_sceneholdout_slat_conditioned_v2
#
# v2 differences vs. v1 (which NaN'd cross_attn after step 1):
#   - --slat-lr 1e-6  (was 1e-5; first step at 1e-5 corrupted cross_attn)
#   - --slat-lr-warmup-steps 500   (linear 0 → 1e-6 warmup on SLAT group)
#   - --fp32-slat-cross-attn       (cross_attn + norm2 run in fp32; default on)
#   - --max-consecutive-nan-skips 50 (watchdog aborts instead of silent spinning)
#   - --epochs 3 (trimmed from 10; fp32 cross_attn adds ~12% step time, full
#     dataset at 4.2s/step ≈ 18.7h/epoch, so 3 epochs ≈ 2.3 days)
#
# Warm-start from the same baseline as v1 (NOT v1's resume.pt — that's NaN-poisoned).
#
# Prereqs:
#   - GPU free (check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
#   - Warm-start checkpoint: artifacts/metric_scale/checkpoints/nocs_sceneholdout_1024dim_baseline_v2_best.pt
#
# Monitor: tail -f /tmp/slat_conditioned_v2.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

LOG=/tmp/slat_conditioned_v2.log

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
    --epochs 3 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --slat-lr-warmup-steps 500 \
    --fp32-slat-cross-attn \
    --max-consecutive-nan-skips 50 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --load-checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_1024dim_baseline_v2_best.pt \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v2.pt \
    --best-output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v2_best.pt \
    --metrics-output artifacts/metric_scale/metrics/nocs_sceneholdout_slat_conditioned_v2_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/nocs_sceneholdout_slat_conditioned_v2_manifest.json \
    --wandb \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name nocs_sceneholdout_slat_conditioned_v2
' > "$LOG" 2>&1 < /dev/null &

echo "Launched PID $! — logs at $LOG"
