#!/usr/bin/env bash
# Deployment-condition eval cache: re-encode ONLY the 464 heldout records at the
# stock 25/25 inference flow steps (the Stage C training cache is encoded at 4/1).
# See planning/MOGE2_METRIC_HEAD_PLAN_2026-06-10.md §2b.
#
# Same dataset/heldout/seed args as the Stage C run => identical heldout split;
# the ONLY difference is --stage1-steps 25 --stage2-steps 25 (and the output path).
#
# Resume after a crash: re-run this script (partial-save/resume is built in).
# Monitor: tail -f /tmp/moge2_cache_heldout_25steps.log

set -euo pipefail

LOG=/tmp/moge2_cache_heldout_25steps.log

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
    --max-records-per-source 3334 \
    --heldout-per-source nocs_real275:64,objectron:200,arkitscenes:200 \
    --overfit-samples 0 \
    --seed 0 \
    --stage1-steps 25 \
    --stage2-steps 25 \
    --cache-latents --cache-only --cache-heldout-only \
    --save-feature-cache artifacts/metric_scale/feature_caches/moge2_mixed_10k_heldout_25steps.pt \
    --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps
' > "$LOG" 2>&1 < /dev/null &

echo "Launched 25/25 heldout cache — log at $LOG"
