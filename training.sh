#!/bin/bash 

CUDA_VISIBLE_DEVICES=4,5,6,7 /home/junyi/.conda/envs/mainpyenv/bin/torchrun  --nproc-per-node=1 --master-port=13599 train-atlas.py 1000 10 whole --batch_size 1 --save_path /data04/junyi/models/alltracts_smoothl1_model_new --init_atlas FA_init_atlas.pt --tracts_num -1
# CUDA_VISIBLE_DEVICES=0,2,6,7 /home/junyi/.conda/envs/mainpyenv/bin/torchrun  --nproc-per-node=4 --master-port=13599 train-atlas.py 1000 10 whole --batch_size 1 --save_path /data04/junyi/models/tracts_4_model --init_atlas FA_init_atlas.pt --tracts_num 4
# CUDA_VISIBLE_DEVICES=0,2,6,7 /home/junyi/.conda/envs/mainpyenv/bin/torchrun  --nproc-per-node=4 --master-port=13599 train-atlas.py 1000 10 whole --batch_size 1 --save_path /data04/junyi/models/tracts_1_model --init_atlas FA_init_atlas.pt --tracts_num 1

