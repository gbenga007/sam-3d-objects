#!/usr/bin/env bash
# LIVE MoGe-2 + SLAT-injection metric-scale run — the heavyweight.
#
# This is train_mixed_scratch_10k.sh (the 22.3%-overall baseline recipe: from
# scratch, SS decoder FROZEN, SLAT cross-attn unfrozen + scale-token injection,
# ~10k balanced records, 12 epochs, A100 80GB) with ONE deliberate change:
#
#   --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps
#
# i.e. the input pointmap (the metric ANCHOR; see project_pointmap_scale_anchor)
# is swapped from live MoGe-v1 (affine-invariant) to the precomputed MoGe-2
# metric pointmaps. The cached head A/Bs (v1a 27.2%, binned, factored) all sit
# below the 22.3% live baseline because they LACK the live SLAT-injection
# capacity. This run puts the MoGe-2 anchor into the FULL live-injection recipe
# to test whether anchor + injection together beat 22.3% overall.
#
# Everything else is byte-identical to train_mixed_scratch_10k.sh so this is an
# apples-to-apples depth-source swap against that checkpoint. The MoGe-2
# pointmaps (4,276 frames, manifest-backed) were precomputed from the SAME
# recipe (seed 0, split train, 3334/source) and cover both train and heldout
# frames; Moge2PointmapStore raises a HARD KeyError on any miss (never silent
# MoGe-v1 fallback), so a coverage gap fails loudly rather than contaminating.
#
# Resume after a crash: just re-run this script (random init, no warm-start).
# Resume is epoch-granular; the intra-epoch recovery checkpoint preserves weights.
#
# Monitor: tail -f /tmp/mixed_moge2_live_v1.log

set -euo pipefail

LOG=/tmp/mixed_moge2_live_v1.log
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
    --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps \
    --max-records-per-source 3334 \
    --balanced-sampling \
    --heldout-per-source nocs_real275:64,objectron:200,arkitscenes:200 \
    --overfit-samples 0 \
    --seed 0 \
    --epochs 12 \
    --eval-every 1 \
    --eval-every-steps 2000 \
    --eval-steps-max-samples 150 \
    --checkpoint-every-steps 1000 \
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
    --output artifacts/metric_scale/checkpoints/mixed_moge2_live_v1.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_moge2_live_v1_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_moge2_live_v1_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_moge2_live_v1_manifest.json \
    --wandb --wandb-mode offline \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_moge2_live_v1
' > "$LOG" 2>&1 < /dev/null &

echo "Launched LIVE MoGe-2 injection run PID $! — logs at $LOG"
echo "Monitor: tail -f $LOG"
