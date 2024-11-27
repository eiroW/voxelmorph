import pathlib
import subprocess
from multiprocessing import Pool
import os
from joblib import Parallel, delayed

# subprocess.run(['python', 'pretrain_process.py'])
backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
def run(tract,i):
    device = i % JOBS + 3
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    save_path = pathlib.Path(f'/data04/junyi/models/models_sep/{tract}_trained_{tract}.pt')
    if save_path.exists() & check_log(f'./logs/{tract}_trained.log'):
        print(f'{tract} already trained')
        return None
    else:
        print(f'training {tract} at device {device} on port {PORTS[i]}')
        # subprocess.run([
        #             '/home/junyi/.conda/envs/mainpyenv/bin/torchrun',
        #             '--nproc-per-node=1',
        #             f'--master-port={PORTS[i]}',
        #             'train-atlas.py', f'{EPOCHS}', '10', tract,
        #             '--batch_size', '4',
        #             '--save_path_prefix', f'/data04/junyi/models/models_sep/{tract}_trained',
        #             '--init_atlas', 'FA_init_atlas.pt'])
        return None
def check_log(logfile):
    if not pathlib.Path(logfile).exists():
        return False
    with open(logfile, 'r') as f:
        lines = f.readlines()
    if lines != [] and 'Epoch 999' in lines[-1]:
        return True
    return False
if __name__ == '__main__':
    EPOCHS = 1000
    PORTS = [12345+i for i in range(len(backbone_tracts))]
    JOBS = 2
    tracts = [tract for i, tract in enumerate(backbone_tracts) if not check_log(f'./logs/{tract}_trained.log')]
    # tracts = ['SLF_II','MLF']
    l = Parallel(n_jobs=JOBS)(delayed(run)(tract, i) for i, tract in enumerate(tracts))
