#!/usr/bin/env bash
# Joint-MoT v2 (2026-06-24): direct smooth-L1 regression on decoded metric dims +
# properly unfrozen full layout MoT blocks (collect_ss_backbone_params, SS_LAYOUT_KEY routing).
#
# Key differences from native_fm_v2:
#   - Loss: smooth_l1(log(decoded W,H,D), log(GT W,H,D))  ← directly coupled to eval metric
#     instead of velocity MSE at random tau (which is only loosely coupled).
#   - ODE steps WITH grad: predict_pose_metric runs --stage1-steps Euler steps with
#     gradient (backprops through the ODE decode into the layout MoT blocks).
#     Keep --stage1-steps low for memory: each grad step stores layout-transformer
#     activations. stage1-steps=1 is safe (~20GB); bump to 2 if headroom allows.
#   - canon_ext now deterministic per item (shape_sample_seed): stable regression target
#     for scale_per_axis across steps. This was the missing fix in joint_mot_v1.
#
# Key differences from joint_mot_v1:
#   - Uses the CORRECT MoT unfreezing (collect_ss_backbone_params routes by SS_LAYOUT_KEY →
#     trains full per-block layout transformer: norms, self-attn, mlp).
#   - joint_mot_v1 only unfroze latent_mapping.* read heads (~0.03M) — plateaued quickly.
#   - Adds HyperSim (50k synthetic, metric depth GT) to diversify beyond NOCS/Obj/ARKit.
#   - canon_ext seeded in training loop (code fix in finetune_metric_scale.py:3134).
#
# Memory: stage1-steps=1 → ~20GB on A100-40GB. Bump to 2 if no OOM after first epoch.
set -euo pipefail
export LIDRA_SKIP_INIT=true
cd /mnt/source/sam-3d-objects

CONDA_PREFIX=/opt/conda/envs/sam3d /opt/conda/envs/sam3d/bin/python \
  sam3d_objects/training/finetune_metric_scale.py \
  --config checkpoints/hf/pipeline.yaml --dataset omninocs-mixed \
  --omninocs-sources nocs_real275 objectron arkitscenes hypersim \
  --omninocs-root /mnt/source/datasets_sam3d/OmniNOCS \
  --rgb-root /mnt/source/datasets_sam3d/OmniNOCS/real_test \
  --objectron-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --arkitscenes-rgb-root /mnt/source/datasets_sam3d/OmniNOCS \
  --hypersim-rgb-root /mnt/source/datasets_sam3d/OmniNOCS/hypersim \
  --skip-missing-rgb \
  --moge2-pointmap-dir artifacts/metric_scale/moge2_pointmaps \
  --max-records-per-source 3334 --balanced-sampling \
  --heldout-per-source nocs_real275:64,objectron:200,arkitscenes:200,hypersim:200 \
  --overfit-samples 0 --seed 0 --epochs 6 \
  --eval-every 1 --eval-every-steps 2000 --eval-steps-max-samples 150 \
  --checkpoint-every-steps 1000 --checkpoint-every 1 --log-step-every 1 \
  --unfreeze-ss-backbone --unfreeze-ss-cross-attn \
  --stage1-steps 1 \
  --ss-geometry-lr 0 --ss-pose-lr 1e-5 --ss-cond-lr 1e-5 \
  --flow-loss-weight 0.0 --weight-decay 1e-4 --batch-size 1 \
  --output artifacts/metric_scale/checkpoints/joint_mot_v2.pt \
  --best-output artifacts/metric_scale/checkpoints/joint_mot_v2_best.pt \
  --metrics-output artifacts/metric_scale/metrics/joint_mot_v2_eval.jsonl \
  --manifest-output artifacts/metric_scale/manifests/joint_mot_v2_manifest.json \
  --wandb --wandb-mode offline --wandb-entity reformed-tulip \
  --wandb-project sam3d-metric-scale --wandb-run-name joint_mot_v2
