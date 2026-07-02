#!/usr/bin/env bash
# Train mixed_v1 — Phase 3a mixed-dataset metric scale training.
#
# Sources: NOCS-Real275 + Objectron + ARKitScenes (all real, no synthetic gap).
# Sampling strategy: Option C —
#   --max-records-per-source 30000  (caps ARKitScenes 170K → 30K; Objectron uses the
#                                    Phase 3a downloaded RGB; NOCS is below the cap)
#   --balanced-sampling             (WeightedRandomSampler with weight=1/source_count,
#                                    replacement=True; gives 1:1:1 expected per batch)
#   --heldout-per-source 64/200/200 (NOCS heldout continuity + new Obj/ARKit heldouts)
#
# Carries over all dials from ss_ratio_v1 (cross-attn unfrozen, SS decoder unfrozen,
# aspect-ratio loss, fp32 cross-attn, slat-lr warmup, CFG dropout, gradient clipping).
#
# Warm-start: ss_ratio_v1_best.pt (1.23% MAPE @ ep6)
# Expected per-epoch cost: ~76K samples × ~3 s/step ≈ 63 hours/epoch on A100-80GB
# 6 epochs total ≈ 16 days. Each example seen ~5× by end of run.
#
# Monitor: tail -f /tmp/mixed_v1.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

# Log to /tmp (local SSD) to avoid Ceph MDS hangs during long writes.
# A cron job mirrors /tmp/mixed_v1.log -> artifacts/metric_scale/logs/mixed_v1.log hourly.
LOG=/tmp/mixed_v1.log
mkdir -p artifacts/metric_scale/logs

setsid -f bash -lc '
  cd /mnt/source/sam-3d-objects
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
    --epochs 6 \
    --eval-every 1 \
    --lr 1e-4 \
    --slat-lr 1e-6 \
    --slat-lr-warmup-steps 500 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --unfreeze-ss-decoder \
    --ss-ratio-loss-weight 0.05 \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --load-checkpoint /tmp/warm_start.pt \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/mixed_v1.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_v1_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_v1_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_v1_manifest.json \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_v1
' > "$LOG" 2>&1 < /dev/null &

echo "Launched PID $! — logs at $LOG (mirrored hourly to artifacts/metric_scale/logs/mixed_v1.log)"
