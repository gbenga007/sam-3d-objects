#!/usr/bin/env bash
# Chain launcher: waits for the train-split anchor extraction to finish, validates
# the table, then runs the v1b/v1a 1000-epoch A/B on the mixed cache sequentially.
# Guarded by a marker so a parallel manual launch can't double-run.
# Monitor: tail -f /tmp/v1ab_chain.log /tmp/mixed_moge2_factored_v1.log /tmp/mixed_moge2_anchor_v1.log
set -u
cd /mnt/source/sam-3d-objects

TABLE=artifacts/metric_scale/metrics/pose_scale_train_moge2.jsonl
ANCHORS="$TABLE,artifacts/metric_scale/metrics/pose_scale_heldout_moge2.jsonl"
MARKER=/mnt/source/sam-3d-objects/artifacts/metric_scale/metrics/.v1ab_launched
MIN_RECORDS=8000

if [ -e "$MARKER" ]; then
  echo "marker $MARKER exists — A/B already launched, exiting"
  exit 0
fi

echo "waiting for extraction process to exit..."
while pgrep -f "pose_scale_heldout_compare.py.*train_8482_meta" > /dev/null; do
  sleep 120
done

N=$(wc -l < "$TABLE" 2>/dev/null || echo 0)
if [ "$N" -lt "$MIN_RECORDS" ]; then
  echo "REFUSING to launch: anchor table has only $N records (< $MIN_RECORDS)."
  echo "Extraction likely died — re-run pose_scale_heldout_compare.py (resumable), then re-run this."
  exit 1
fi
echo "anchor table complete: $N records. Launching A/B."
touch "$MARKER"

export LD_LIBRARY_PATH=/opt/conda/envs/sam3d/lib:${LD_LIBRARY_PATH:-}
export LIDRA_SKIP_INIT=1

for dec in factored anchor; do
  echo "=== $(date -u +%F\ %T) launching decoder=$dec ==="
  /opt/conda/envs/sam3d/bin/python sam3d_objects/training/finetune_metric_scale.py \
    --load-feature-cache artifacts/metric_scale/feature_caches/moge2_mixed_10k.pt \
    --decoder "$dec" --anchor-tables "$ANCHORS" \
    --scale-loss-weight 1.0 --prop-loss-weight 1.0 \
    --epochs 1000 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
    --eval-every 5 --seed 0 \
    --output "artifacts/metric_scale/checkpoints/mixed_moge2_${dec}_v1.pt" \
    --best-output "artifacts/metric_scale/checkpoints/mixed_moge2_${dec}_v1_best.pt" \
    --metrics-output "artifacts/metric_scale/metrics/mixed_moge2_${dec}_v1_eval.jsonl" \
    > "/tmp/mixed_moge2_${dec}_v1.log" 2>&1
  echo "=== $(date -u +%F\ %T) decoder=$dec exit=$? ==="
  grep "Saved best" "/tmp/mixed_moge2_${dec}_v1.log" | tail -1
done
echo "A/B COMPLETE"
