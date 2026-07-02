#!/usr/bin/env bash
# FROM-SCRATCH mixed metric-scale training — ~10k-record SFT on A100 80GB.
#
# Same recipe as train_mixed_scratch.sh (random init, SS decoder FROZEN so the
# checkpoint is inference-valid by construction, no aspect-ratio loss), with two
# deliberate changes for "high quality in decent time":
#
#   1. DATA SIZE: --max-records-per-source 3334 -> ~10,002 balanced records total
#      (was 30000/source = ~76k). A SAM3D author reported ~10k-scale SFT gives
#      significant improvement and ~1k already helps; 76k was over-provisioning.
#      ~7.6x fewer steps/epoch, and on the A100 (~3x the A10) the wall-clock is a
#      different world from the run that died at 24% of epoch 1.
#
#   2. INTRA-EPOCH EVAL + CHECKPOINT: --eval-every-steps / --checkpoint-every-steps
#      so a restart no longer wipes a whole epoch and the held-out curve is visible
#      within the hour. (The dead run ran 18.5h and saved nothing — end-of-epoch only.)
#
# Everything else is the validated, smoke-passed config: stage1-steps 4, stage2-steps
# 1, slat-lr 1e-6 with a 1500-step warmup, SLAT gradient-checkpointing on. We are NOT
# adding the A100-only per-step speedups (checkpointing off / stage1-steps 2) here —
# they are a separate, smoke-tested change for later if more throughput is wanted.
#
# !!! RUN scripts/train_mixed_scratch_smoke.sh FIRST and confirm it is stable !!!
#
# Resume after a crash: just re-run this script (random init, no warm-start file).
# Resume is epoch-granular; the intra-epoch recovery checkpoint preserves weights.
#
# Monitor: tail -f /tmp/mixed_scratch_10k.log

set -euo pipefail

LOG=/tmp/mixed_scratch_10k.log
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
    --output artifacts/metric_scale/checkpoints/mixed_scratch_10k.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_scratch_10k_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_scratch_10k_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_scratch_10k_manifest.json \
    --wandb --wandb-mode offline \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_scratch_10k
' > "$LOG" 2>&1 < /dev/null &

echo "Launched FROM-SCRATCH 10k run PID $! — logs at $LOG"
echo "Monitor: tail -f $LOG"
