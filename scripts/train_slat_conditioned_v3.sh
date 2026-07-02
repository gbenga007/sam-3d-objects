#!/usr/bin/env bash
# Train nocs_sceneholdout_slat_conditioned_v3
#
# v3 = continuation of v2 (which completed 2 epochs before deployment crash).
#
# Differences vs. v2:
#   - --resume-from v2_best.pt  (loads epoch=2 + cross_attn weights; starts from ep3)
#   - --epochs 5                (runs epochs 3-5, i.e. 3 more epochs)
#   - --slat-lr-warmup-steps 0  (cross_attn already stable; no warmup needed)
#
# Monitor: tail -f /tmp/slat_conditioned_v3.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

LOG=/tmp/slat_conditioned_v3.log

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
    --epochs 5 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --slat-lr-warmup-steps 0 \
    --fp32-slat-cross-attn \
    --max-consecutive-nan-skips 50 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --resume-from artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v2_best.pt \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3.pt \
    --best-output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v3_best.pt \
    --metrics-output artifacts/metric_scale/metrics/nocs_sceneholdout_slat_conditioned_v3_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/nocs_sceneholdout_slat_conditioned_v3_manifest.json \
    --wandb-run-name nocs_sceneholdout_slat_conditioned_v3
' > "$LOG" 2>&1 < /dev/null &

echo "Launched PID $! — logs at $LOG"
