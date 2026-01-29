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
