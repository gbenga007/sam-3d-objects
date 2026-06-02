#!/usr/bin/env bash
# Submit the multi-GPU FROM-SCRATCH metric-scale training job to Nautilus via
# envsubst. Cluster equivalent of scripts/train_mixed_scratch.sh: frozen SS
# decoder, no warm-start → inference-valid checkpoints. Defaults to 2 GPUs/pod
# (the schedulable sweet spot on Nautilus — see note below).
#
# Reads the WANDB_API_KEY from your shell environment so it's never written
# to disk. Set it before running:
#
#   export WANDB_API_KEY='your-key-from-wandb.ai/authorize'
#   ./scripts/submit_multi_gpu_job.sh
#
# Override defaults by exporting variables before invoking:
#   NUM_GPUS=2 EPOCHS=12 ./scripts/submit_multi_gpu_job.sh
#
# Cheap cluster SMOKE TEST first (tiny data, 1 epoch — confirms it schedules,
# the container builds, and DDP runs end-to-end before committing days):
#   MAX_RECORDS=160 EPOCHS=1 JOB_NAME=mixed-scratch-multi-smoke \
#     ./scripts/submit_multi_gpu_job.sh
#
# To preview rendered YAML without applying, pass --dry-run:
#   ./scripts/submit_multi_gpu_job.sh --dry-run

set -euo pipefail

# ─── Required ───────────────────────────────────────────────────────────
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY env var is required."
  echo "  Get it from https://wandb.ai/authorize, then:"
  echo "    export WANDB_API_KEY='wandb_v1_...'"
  exit 1
fi

# ─── Tunables (override via env before running) ─────────────────────────
# DEFAULT = 2 GPUs/pod. This is the schedulable sweet spot on Nautilus: a
# single pod asking for 8–16 GPUs needs ONE node with that many free A10s at
# the same instant, which almost never exists → pods sit Pending forever.
# 2 GPUs/pod schedules reliably. For more throughput, launch SEVERAL 2-GPU
# jobs, or move to a true multi-node setup (separate effort).
export num_gpus=${NUM_GPUS:-2}
export job_name=${JOB_NAME:-mixed-scratch-multi-${num_gpus}gpu}
export epochs=${EPOCHS:-12}
# LR sqrt-scaling vs the 1-GPU base (lr=1e-4, slat-lr=1e-6).
#   2 GPUs: sqrt(2)×base ≈ 1.4e-4 / 1.4e-6
export lr=${LR:-1.4e-4}
export slat_lr=${SLAT_LR:-1.4e-6}
# From-scratch recipe (frozen SS decoder, no warm-start) → checkpoint is
# inference-valid by construction. No warm_start_path needed. Distinct run_name
# so cluster checkpoints never clobber the local mixed_scratch run on the
# shared PVC.
export max_records=${MAX_RECORDS:-30000}
export run_name=${RUN_NAME:-${job_name//-/_}}

# ─── Auto-scale resources from num_gpus ─────────────────────────────────
# Lean defaults to improve cluster scheduling latency on Nautilus.
# - CPU: 2 cores per GPU (data loading + general PyTorch work)
# - Memory: 4 GiB per GPU (SAM3D forward pass uses ~5GB but shares model copies)
# - Ephemeral storage: 50 GiB for container scratch (checkpoints go to PVC)
# Bump these if pods get OOM-killed or evicted due to disk pressure.
export cpu_cores=${CPU_CORES:-$(( num_gpus * 2 ))}
export memory_gi=${MEMORY_GI:-$(( num_gpus * 4 ))}
export ephemeral_gi=${EPHEMERAL_GI:-50}

# ─── Wandb key from env (never written to disk) ─────────────────────────
export wandb_api_key="${WANDB_API_KEY}"

# ─── Apply ──────────────────────────────────────────────────────────────
TEMPLATE_PATH="$(dirname "$0")/../unimatch_job.yaml"

echo "=== Job configuration ==="
echo "  job_name:        $job_name"
echo "  num_gpus:        $num_gpus"
echo "  epochs:          $epochs"
echo "  lr:              $lr"
echo "  slat_lr:         $slat_lr"
echo "  max_records:     $max_records"
echo "  run_name:        $run_name"
echo "  cpu_cores:       $cpu_cores"
echo "  memory_gi:       $memory_gi"
echo "  ephemeral_gi:    $ephemeral_gi"
echo "  wandb_api_key:   <redacted, from env>"
echo

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== Rendered YAML (dry-run) ==="
  envsubst < "$TEMPLATE_PATH" | sed "s|${wandb_api_key}|<WANDB_KEY_REDACTED>|g"
  exit 0
fi

echo "Applying job to cluster…"
envsubst < "$TEMPLATE_PATH" | kubectl apply -f -
echo
echo "Watch logs with:  kubectl logs -f job/$job_name"
echo "Cancel job with:  kubectl delete job/$job_name"
