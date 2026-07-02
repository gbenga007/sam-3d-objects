#!/usr/bin/env bash
# v2 binned scale head (OmniNOCS-style): softmax-CE over iso-scale bins on top of
# the winning v1a anchor features + GT-normalized L1 + max-pinned proportions.
# Same cache / anchor tables / schedule as the v1a/v1b A/B (run_v1ab_after_extraction.sh)
# so the heldout numbers are directly comparable to v1a (27.2 mean / 15.7 median).
# Monitor: tail -f /tmp/mixed_moge2_binned_v2.log
set -u
cd /mnt/source/sam-3d-objects

TABLE=artifacts/metric_scale/metrics/pose_scale_train_moge2.jsonl
ANCHORS="$TABLE,artifacts/metric_scale/metrics/pose_scale_heldout_moge2.jsonl"
CACHE=artifacts/metric_scale/feature_caches/moge2_mixed_10k.pt
MIN_RECORDS=8000

N=$(wc -l < "$TABLE" 2>/dev/null || echo 0)
if [ "$N" -lt "$MIN_RECORDS" ]; then
  echo "REFUSING to launch: anchor table has only $N records (< $MIN_RECORDS)."
  exit 1
fi
echo "anchor table: $N records. Launching binned v2."

export LD_LIBRARY_PATH=/opt/conda/envs/sam3d/lib:${LD_LIBRARY_PATH:-}
export LIDRA_SKIP_INIT=1

/opt/conda/envs/sam3d/bin/python sam3d_objects/training/finetune_metric_scale.py \
  --load-feature-cache "$CACHE" \
  --decoder binned --anchor-tables "$ANCHORS" \
  --bin-ce-weight 1.0 --scale-loss-weight 1.0 --prop-loss-weight 1.0 \
  --num-bins 128 --bin-log-min -5.0 --bin-log-max 2.0 \
  --epochs 1000 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 \
  --eval-every 5 --seed 0 \
  --output       artifacts/metric_scale/checkpoints/mixed_moge2_binned_v2.pt \
  --best-output  artifacts/metric_scale/checkpoints/mixed_moge2_binned_v2_best.pt \
  --metrics-output artifacts/metric_scale/metrics/mixed_moge2_binned_v2_eval.jsonl \
  > /tmp/mixed_moge2_binned_v2.log 2>&1

echo "binned v2 exit=$?"
grep "Saved best" /tmp/mixed_moge2_binned_v2.log | tail -1
