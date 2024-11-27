import subprocess
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Run the training script

subprocess.run([
    '/home/junyi/.conda/envs/mainpyenv/bin/torchrun',
    '--nproc-per-node=4',
    '--master-port=13599',
    './train-atlas.py', 
    '1000', '10', 'whole',
    '--batch_size', '1',
    '--save_path', '/data04/junyi/models/tracts_17_model',
    '--init_atlas', 'FA_init_atlas.pt',
    '--tracts_num', '17'
])