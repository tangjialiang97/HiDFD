#!/bin/bash
{
 CUDA_VISIBLE_DEVICES=0 python train_pre.py \
--select_dir /cifar10_select_dir/ \
--shuffle --batch_size 100 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 400 \
--num_D_steps 1 --G_lr 2e-4 --D_lr 2e-4 \
--dataset C10_select \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 2000 --save_every 2000 --num_best_copies 1 --num_save_copies 0 --seed 0 \
--loss adcgan --G_lambda 1.0 --D_lambda 1.0 --D_reg_lambda 0.1 --G_reg_lambda 0.1 --experiment_name C10_select
}
