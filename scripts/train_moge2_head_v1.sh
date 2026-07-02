#!/usr/bin/env bash
# Stage C/D head training — MoGe-2-anchored MetricScaleHead + MetricScaleDecoder
# on the CACHED latents (frozen generator; no pipeline forwards at all).
# See planning/MOGE2_METRIC_HEAD_PLAN_2026-06-10.md.
#
# Cache: feature_caches/moge2_mixed_10k.pt (train=8482, heldout=375, encoded at
# SS=4/SLAT=1 flow steps with MoGe-2 metric pointmaps injected).
# Checkpoint = head + decoder only => inference-valid by construction.
#
# Baselines to beat: mixed_scratch_10k_best.pt (MoGe-v1 anchor, 22.3% overall)
# and the global-calibration-constant baseline (no head).
#
# Resume after a crash: just re-run (cached epochs are minutes; no resume logic).
# Monitor: tail -f /tmp/mixed_moge2_v1.log

set -euo pipefail

LOG=/tmp/mixed_moge2_v1.log

setsid -f bash -lc '
  cd /mnt/source/sam-3d-objects
  export LD_LIBRARY_PATH=/opt/conda/envs/sam3d/lib:${LD_LIBRARY_PATH:-}
  export LIDRA_SKIP_INIT=1
  exec /opt/conda/envs/sam3d/bin/python \
    sam3d_objects/training/finetune_metric_scale.py \
    --load-feature-cache artifacts/metric_scale/feature_caches/moge2_mixed_10k.pt \
    --epochs 200 \
    --batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --eval-every 1 \
    --seed 0 \
    --output artifacts/metric_scale/checkpoints/mixed_moge2_v1.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_moge2_v1_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_moge2_v1_eval.jsonl \
    --wandb --wandb-mode offline \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_moge2_v1
' > "$LOG" 2>&1 < /dev/null &

echo "Launched mixed_moge2_v1 head training — log at $LOG"
