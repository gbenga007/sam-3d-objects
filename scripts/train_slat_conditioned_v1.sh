#!/usr/bin/env bash
# Train nocs_sceneholdout_slat_conditioned_v1
#
# SLAT cross-attention conditioned on metric scale token.
# Full NOCS-Real275 train split (16,118 samples), eval against frozen-SLAT
# feature cache every epoch.
#
# Prereqs:
#   - GPU free (check: nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
#   - Eval cache present at /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt
#   - Warm-start checkpoint: artifacts/metric_scale/checkpoints/nocs_sceneholdout_1024dim_baseline_v2_best.pt
#
# To resume from a saved checkpoint mid-run:
#   Add: --resume-from artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v1_resume.pt
#
# Monitor: tail -f /tmp/slat_conditioned_v1.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

LOG=/tmp/slat_conditioned_v1.log

ATTN_BACKEND=sdpa \
SPARSE_ATTN_BACKEND=sdpa \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup \
/root/.local/bin/micromamba run -n sam3d-objects python \
  sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml \
  --dataset omninocs-nocs-real275 \
  --omninocs-root /mnt/dest/OmniNOCS \
  --omninocs-sources nocs_real275 \
  --annotations-root /mnt/dest/OmniNOCS/omninocs_release_nocs_real275 \
  --rgb-root /mnt/dest/OmniNOCS/real_test \
  --skip-missing-rgb \
  --split-group record \
  --split train \
  --overfit-samples 0 \
  --seed 0 \
  --epochs 10 \
  --eval-every 1 \
  --lr 1e-4 \
  --slat-lr 1e-5 \
  --weight-decay 1e-4 \
  --stage1-steps 4 \
  --stage2-steps 1 \
  --batch-size 1 \
  --unfreeze-slat-cross-attn \
  --load-checkpoint artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v1_resume.pt \
  --eval-feature-cache /tmp/metric_scale_omninocs_sceneholdout_train12890_scene6_cache.pt \
  --checkpoint-every 1 \
  --output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v1.pt \
  --best-output artifacts/metric_scale/checkpoints/nocs_sceneholdout_slat_conditioned_v1_best.pt \
  --metrics-output artifacts/metric_scale/metrics/nocs_sceneholdout_slat_conditioned_v1_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/nocs_sceneholdout_slat_conditioned_v1_manifest.json \
  --wandb \
  --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale \
  --wandb-run-name nocs_sceneholdout_slat_conditioned_v1 \
  > "$LOG" 2>&1 &

echo "Launched PID $! — logs at $LOG"
