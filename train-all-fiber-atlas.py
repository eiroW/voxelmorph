import os
import numpy as np
import torch
from torch import nn
import functools
from typing import Union
import subprocess as sp
gpus_list = ''
for i in range(torch.cuda.device_count()):
    used,all=torch.cuda.mem_get_info(device=i)
    if used/all >=0.80:
        gpus_list+=f'{i},'

os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['TORCHELASTIC_ERROR_FILE'] = 'error_history.json'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list[:-1]
os.environ['OMP_NUM_THREADS'] = '2'

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
def main():
    sp.run(f'torchrun  --nproc-per-node=4 train-atlas.py 500 10 {tracts} --batch_size 4 --save_path snapshot1',shell=True,check=True)
if __name__=='__main__':
    for tracts in backbone_tracts:
        main()

