#!/usr/bin/env bash
# Multi-GPU FROM-SCRATCH metric-scale training via HuggingFace Accelerate.
# Single-NODE only (one accelerate process group). For Nautilus, prefer the
# k8s path: scripts/submit_multi_gpu_job.sh (defaults to 2 GPUs/pod).
#
# From-scratch recipe (matches train_mixed_scratch.sh): SS decoder FROZEN, no
# warm-start → checkpoints are inference-valid by construction. Do NOT re-add
# --unfreeze-ss-decoder / --ss-ratio-loss-weight / --load-checkpoint.
#
# Each rank loads the full SAM3D backbone (~4GB bf16) plus activations —
# confirmed to fit in 24GB A10 at batch-size=1.
#
# Effective batch size = NUM_GPUS × batch-size = NUM_GPUS × 1
# LR scaling (sqrt rule):  lr_new = lr_base × sqrt(NUM_GPUS)
#   - 1 GPU:  lr=1e-4,    slat-lr=1e-6
#   - 2 GPUs: lr=1.4e-4,  slat-lr=1.4e-6
#   - 4 GPUs: lr=2e-4,    slat-lr=2e-6
#
# Prereqs:
#   pip install accelerate
#   - First run: `accelerate config` (or pass --config_file)
#   - For an interactive cluster session, just `accelerate launch ...`
#
# SLURM equivalent (8 GPUs on one node):
#   #SBATCH --gres=gpu:8
#   #SBATCH --ntasks=1
#   #SBATCH --cpus-per-task=16
#   srun bash scripts/train_mixed_v2_multi_gpu.sh
#
# Monitor: tail -f /tmp/mixed_scratch_multi_gpu.log
# Wandb:   https://wandb.ai/reformed-tulip/sam3d-metric-scale

set -euo pipefail

# ----- Edit these for your cluster --------------------------------------
# Set NUM_GPUS to the number of GPUs ON THIS NODE (single-node only).
NUM_GPUS=${NUM_GPUS:-2}           # accelerate launch --num_processes
LR=${LR:-1.4e-4}                  # sqrt(2) × 1e-4
SLAT_LR=${SLAT_LR:-1.4e-6}        # sqrt(2) × 1e-6
EPOCHS=${EPOCHS:-12}
LOG=/tmp/mixed_scratch_multi_gpu.log
# ------------------------------------------------------------------------

cd /mnt/source/sam-3d-objects

export LIDRA_SKIP_INIT=1
export ATTN_BACKEND=flash_attn
export SPARSE_ATTN_BACKEND=flash_attn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Disable HuggingFace tokenizers parallelism warning under multi-process.
export TOKENIZERS_PARALLELISM=false

setsid -f bash -lc "
  accelerate launch \
    --num_processes ${NUM_GPUS} \
    --mixed_precision bf16 \
    --dynamo_backend no \
    sam3d_objects/training/finetune_metric_scale_multi_gpu.py \
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
    --epochs ${EPOCHS} \
    --eval-every 1 \
    --lr ${LR} \
    --slat-lr ${SLAT_LR} \
    --slat-lr-warmup-steps 1500 \
    --weight-decay 1e-4 \
    --stage1-steps 4 \
    --stage2-steps 1 \
    --batch-size 1 \
    --unfreeze-slat-cross-attn \
    --p-uncond-scale-token 0.1 \
    --log-step-every 10 \
    --checkpoint-every 1 \
    --output artifacts/metric_scale/checkpoints/mixed_scratch_multi.pt \
    --best-output artifacts/metric_scale/checkpoints/mixed_scratch_multi_best.pt \
    --metrics-output artifacts/metric_scale/metrics/mixed_scratch_multi_eval.jsonl \
    --manifest-output artifacts/metric_scale/manifests/mixed_scratch_multi_manifest.json \
    --wandb \
    --wandb-entity reformed-tulip \
    --wandb-project sam3d-metric-scale \
    --wandb-run-name mixed_scratch_multi_${NUM_GPUS}gpu
" > "$LOG" 2>&1 < /dev/null &

echo "Launched ${NUM_GPUS}-GPU from-scratch training — PID $! — logs at $LOG"
echo "Monitor with: tail -f $LOG"
