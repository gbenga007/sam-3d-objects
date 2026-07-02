#!/usr/bin/env bash
# FROM-SCRATCH mixed metric-scale training (single A10).
#
# Trains the metric adapters (MetricScaleHead + MetricScaleDecoder + SLAT cross-attn)
# from RANDOM init directly on the mixed corpus (NOCS + Objectron + ARKitScenes), on top
# of the FROZEN public SAM3D backbone. No warm-start, no two-phase curriculum.
#
# Why from-scratch:
#   * Single-phase recipe -> cleanest reproducibility (one command from public weights).
#   * Removes the "are mixed gains just inherited from NOCS pretraining?" confound.
#   * No dependency on the (now bug-tainted) checkpoint chain.
#
# Differences from mixed_v1 (the broken run) — ALL THREE are deliberate:
#   * REMOVED --load-checkpoint        -> random init instead of warm-start.
#   * REMOVED --unfreeze-ss-decoder    -> SS decoder stays frozen. This is what caused the
#                                         pipeline-skip blow-up AND the unsaved-weights bug.
#   * REMOVED --ss-ratio-loss-weight   -> no aspect-ratio supervision (separate future work).
#   With the SS decoder frozen, the checkpoint is inference-valid BY CONSTRUCTION — there is
#   no fine-tuned backbone component that could be silently dropped on save.
#
# Cold-start tuning vs mixed_v1:
#   * --slat-lr-warmup-steps 1500 (was 500) — gentler ramp from random init (v1 NaN'd early).
#   * --epochs 12 (was 6) — cold start needs more passes; warm-started mixed_v1 was still
#     improving at ep6, so a random init will need at least that many.
#
# !!! RUN scripts/train_mixed_scratch_smoke.sh FIRST and confirm it is stable !!!
#
# Single A10 wall-clock: ~9 days got 6 warm-started epochs, so ~12 cold epochs ≈ 2+ weeks.
# For a faster schedule, run the multi-GPU equivalent on Nautilus: take
# scripts/train_mixed_v2_multi_gpu.sh and delete the same three flags removed here.
#
# Monitor: tail -f /tmp/mixed_scratch.log

set -euo pipefail

LOG=/tmp/mixed_scratch.log
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
    --max-records-per-source 30000 \
    --balanced-sampling \
    --heldout-per-source nocs_real275:64,objectron:200,arkitscenes:200 \
    --overfit-samples 0 \
    --seed 0 \
    --epochs 12 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --slat-lr-warmup-steps 1500 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/mixed_scratch.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_scratch_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_scratch_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_scratch_manifest.json \
    --wandb --wandb-mode offline \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_scratch
' > "$LOG" 2>&1 < /dev/null &

echo "Launched FROM-SCRATCH run PID $! — logs at $LOG"
echo "Monitor: tail -f $LOG"
