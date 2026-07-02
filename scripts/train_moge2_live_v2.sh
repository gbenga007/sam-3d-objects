#!/usr/bin/env bash
# LIVE MoGe-2 + SLAT-injection + SS layout transformer (hybrid).
#
# Same recipe as train_moge2_live_v1.sh (the 20.2%-overall best: MoGe-2 metric
# anchor, SLAT cross-attn unfrozen, scale-token injection, ~10k balanced, 12
# epochs, A100 80GB) with ONE addition:
#
#   --unfreeze-ss-layout-for-head  (+ --ss-pose-lr 1e-5)
#
# This enables the SS generator's layout transformer (the MoT 6drotation_normalized
# modality blocks: norms, self-attn, mlp) to receive gradients through the
# MetricScaleHead's ss_scale_features input (3-dim log-SSI scale readout).
# The shape transformer stays frozen via the MoT stop-grad k/v detach.
#
# The gradient chain is:
#   loss -> MetricScaleDecoder -> SLAT feats -> SLAT cross-attn
#   -> scale_token -> MetricScaleHead -> ss_scale_features (3-dim)
#   -> SS layout transformer
#
# Everything else is identical to v1 so this is a clean A/B:
# does letting the MoT layout blocks adapt their scale token representations
# improve on the 20.2% / ARKit 28.6% ceiling of mixed_moge2_live_v1_best.pt?
#
# Resume after a crash: re-run this script (random init, epoch-granular resume).
# Monitor: tail -f /tmp/mixed_moge2_live_v2.log

set -euo pipefail

LOG=/tmp/mixed_moge2_live_v2.log
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
    --ss-pose-lr 1e-5 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --unfreeze-ss-layout-for-head \
    --no-ss-grad-checkpoint \
    --p-uncond-scale-token 0.1 \
    --log-step-every 1 \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/mixed_moge2_live_v2.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_moge2_live_v2_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_moge2_live_v2_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_moge2_live_v2_manifest.json \
    --wandb --wandb-mode offline \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_moge2_live_v2
' > "$LOG" 2>&1 < /dev/null &

echo "Launched hybrid LIVE MoGe-2 + SS layout run PID $! — logs at $LOG"
echo "Monitor: tail -f $LOG"
