#!/bin/bash

export ENTITY=22307100070-fdsmsyp
export PROJECT=Disperse-Loss
export WANDB_KEY=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


python train.py \
  --data-path data/cifar10 \
  --model SiT-XS/1 \
  --image-size 32 \
  --num-classes 10 \
  --epochs 80 \
  --prediction noise \
  --loss-weight likelihood \
  --global-batch-size 32 \
  --global-seed 0 \
  --vae ema \
  --num-workers 4 \
  --log-every 100 \
  --ckpt-every 5000 \
  --sample-every 5000 \
  --cfg-scale 4.0 \
  --wandb \
  --disp

# ---- sampling (DDP) ----
RESULTS_DIR="results"
SAMPLE_DIR="samples"
NPROC_PER_NODE=1
NUM_FID_SAMPLES=50000
PER_PROC_BATCH_SIZE=64

CKPT_PATH=$(ls -t "${RESULTS_DIR}"/*/checkpoints/*.pt 2>/dev/null | head -n 1)
if [ -z "${CKPT_PATH}" ]; then
  echo "No checkpoint found under ${RESULTS_DIR}. Aborting sampling."
  exit 1
fi

torchrun --standalone --nproc_per_node=${NPROC_PER_NODE} sample_ddp.py ODE \
  --model SiT-XS/1 \
  --image-size 32 \
  --num-classes 10 \
  --cfg-scale 4.0 \
  --prediction noise \
  --loss-weight likelihood \
  --path-type Linear \
  --ckpt "${CKPT_PATH}" \
  --sample-dir "${SAMPLE_DIR}" \
  --num-fid-samples ${NUM_FID_SAMPLES} \
  --per-proc-batch-size ${PER_PROC_BATCH_SIZE}

# ---- FID ----
REAL_DIR="data/cifar10_real_images"
SAMPLE_FOLDER=$(ls -dt "${SAMPLE_DIR}"/* 2>/dev/null | head -n 1)
if [ -z "${SAMPLE_FOLDER}" ]; then
  echo "No sample folder found under ${SAMPLE_DIR}. Aborting FID."
  exit 1
fi

python fid_folders.py "${REAL_DIR}" "${SAMPLE_FOLDER}"
