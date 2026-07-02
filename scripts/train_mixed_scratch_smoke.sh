#!/usr/bin/env bash
# SMOKE TEST for from-scratch mixed training (no warm-start, no SS-decoder fine-tuning).
#
# Purpose: cheaply confirm that cold-start mixed training is STABLE before committing
# ~2 weeks to the full run. The danger is NaN divergence from a random adapter init at
# full slat-lr — the exact failure mode that killed slat_conditioned_v1 on its first
# optimizer step. So this smoke test:
#   * uses a SHORT warmup (100 steps) to reach FULL slat-lr quickly and stress-test it
#   * runs ~1 epoch over a tiny per-source cap (~430 steps total, ~20-30 min on one A10)
#   * relies on --max-consecutive-nan-skips (default 50) to abort fast if it diverges
#
# PASS criteria: loss descends, no NaN-skip cascade, run completes 1 epoch + eval.
# FAIL: NaN watchdog aborts, or loss is flat/exploding -> tune warmup/slat-lr/clip first.
#
# Differs from the real run ONLY in dataset size / epochs / warmup. The init config
# (no --load-checkpoint, no --unfreeze-ss-decoder, no --ss-ratio-loss-weight) is identical.
#
# Monitor: tail -f /tmp/mixed_scratch_smoke.log

set -euo pipefail

LOG=/tmp/mixed_scratch_smoke.log
mkdir -p artifacts/metric_scale/logs

setsid -f bash -lc '
  cd /mnt/source/sam-3d-objects
  export LD_LIBRARY_PATH=/opt/conda/envs/sam3d/lib:${LD_LIBRARY_PATH:-}
  export LIDRA_SKIP_INIT=1
  export ATTN_BACKEND=flash_attn
  export SPARSE_ATTN_BACKEND=flash_attn
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  exec /opt/conda/envs/sam3d/bin/python \
    sam3d_objects/training/finetune_metric_scale.py \
    --config checkpoints/hf/pipeline.yaml \
    --dataset omninocs-mixed \
    --omninocs-sources nocs_real275 objectron arkitscenes \
    --omninocs-root /mnt/source/datasets_sam3d/OmniNOCS \
    --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
    --objectron-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
    --arkitscenes-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
    --skip-missing-rgb \
    --max-records-per-source 160 \
    --balanced-sampling \
    --heldout-per-source nocs_real275:16,objectron:16,arkitscenes:16 \
    --overfit-samples 0 \
    --seed 0 \
    --epochs 1 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --slat-lr-warmup-steps 100 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --checkpoint-every 1 \
    --eval-every-steps 150 \
    --eval-steps-max-samples 24 \
    --checkpoint-every-steps 100 \
    --output artifacts/metric_scale/checkpoints/mixed_scratch_smoke.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_scratch_smoke_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_scratch_smoke_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_scratch_smoke_manifest.json \
    --wandb-mode disabled
' > "$LOG" 2>&1 < /dev/null &

echo "Launched SMOKE TEST PID $! — logs at $LOG"
echo "Watch: tail -f $LOG  |  PASS = loss descends, no NaN cascade, completes 1 epoch + eval"
