#!/usr/bin/env bash
# Clean A/B: re-run binned v2 + v1a anchor on the SIZE-OUTLIER-CLEANED mixed cache
# (moge2_mixed_10k_clean.pt — the 41m Objectron cups etc. dropped). Same schedule as
# the dirty A/B so numbers are comparable across the clean/dirty axis.
# Monitor: tail -f /tmp/mixed_moge2_{binned_v2,anchor_v1}_clean.log
set -u
cd /mnt/source/sam-3d-objects

CACHE=artifacts/metric_scale/feature_caches/moge2_mixed_10k_clean.pt
ANCHORS="artifacts/metric_scale/metrics/pose_scale_train_moge2.jsonl,artifacts/metric_scale/metrics/pose_scale_heldout_moge2.jsonl"

if [ ! -f "$CACHE" ]; then echo "clean cache missing: $CACHE"; exit 1; fi

export LD_LIBRARY_PATH=/opt/conda/envs/sam3d/lib:${LD_LIBRARY_PATH:-}
export LIDRA_SKIP_INIT=1

run() {  # $1 = decoder, $2... = extra args
  local dec="$1"; shift
  echo "=== $(date -u +%F\ %T) launching decoder=$dec (clean) ==="
  /opt/conda/envs/sam3d/bin/python sam3d_objects/training/finetune_metric_scale.py \
    --load-feature-cache "$CACHE" \
    --decoder "$dec" --anchor-tables "$ANCHORS" \
    --epochs 1000 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
    --eval-every 5 --seed 0 "$@" \
    --output       "artifacts/metric_scale/checkpoints/mixed_moge2_${dec}_clean.pt" \
    --best-output  "artifacts/metric_scale/checkpoints/mixed_moge2_${dec}_clean_best.pt" \
    --metrics-output "artifacts/metric_scale/metrics/mixed_moge2_${dec}_clean_eval.jsonl" \
    > "/tmp/mixed_moge2_${dec}_clean.log" 2>&1
  echo "=== $(date -u +%F\ %T) decoder=$dec exit=$? ==="
  grep "Saved best" "/tmp/mixed_moge2_${dec}_clean.log" | tail -1
}

# binned (v2) first — the experiment of record this session — then anchor (v1a) baseline
run binned --bin-ce-weight 1.0 --scale-loss-weight 1.0 --prop-loss-weight 1.0 \
           --num-bins 128 --bin-log-min -5.0 --bin-log-max 2.0
run anchor  --scale-loss-weight 1.0 --prop-loss-weight 1.0
echo "CLEAN A/B COMPLETE"
