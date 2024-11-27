import pathlib
from voxelmorph.py import utils
from joblib import Parallel, delayed
from itertools import combinations
import pandas as pd
import nibabel as nib
import numpy as np
def compute_dice(gt,pred,is_WGC=False):
    gt = load_nii(gt)
    pred = load_nii(pred)
    if is_WGC:
        dice = utils.dice(gt,pred,labels=[1,2,3],)
        print(dice)
        dice = dice.mean()
    else:
        dice = utils.dice(gt,pred,)[0]
    return dice
def load_nii(path):
    data = nib.load(path).get_fdata()
    # data[...,110:] = 0
    return data
if __name__ == '__main__':
    for dataset in ['HCP_test','ABCD','PPMI',]:
        RAWPATH = pathlib.Path(f'/data04/junyi/Datasets/DTI/{dataset}/')
        backbone_tracts = (['AF', 'ATR', 'CA', 'CC',
                        'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                        'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                        'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                        'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
                        )
        dice =pd.read_csv(f'./dice_{dataset}.csv',index_col=0)
        dice = dice[(dice['Method']!='DTITK') | (dice['Tract']!='WGC')]
        # for tract in backbone_tracts:
        #     new_dice = pd.DataFrame(columns=dice.columns)
        #     print(tract)
        #     seg_path = sorted(RAWPATH.glob(f'*_tmp_seg/{tract}_warped.nii.gz'))
        #     new_dice['Dice'] = Parallel(n_jobs=20)(delayed(compute_dice)(sub1,sub2)for sub1,sub2 in combinations(seg_path,2))
        #     new_dice['subid'] = range(len(seg_path)*(len(seg_path)-1)//2)
        #     new_dice['Tract'] = tract
        #     new_dice['Method'] = 'DTITK'
        #     print(len(seg_path))
        #     dice = pd.concat([dice,new_dice])

        WGCseg = sorted(RAWPATH.glob('*_tmp_seg/SegmentationMap_GMWMCSF_MNI_warped*'))
        print(WGCseg)
        seg_list = [load_nii(i) for i in WGCseg]
        all_seg = np.stack(seg_list,axis=3)
        nib.save(nib.Nifti1Image(all_seg,np.eye(4)),f'./WGC_{dataset}.nii.gz')
        new_dice = pd.DataFrame(columns=dice.columns)
        new_dice['Dice'] = Parallel(n_jobs=20)(delayed(compute_dice)(sub1,sub2,is_WGC=True) for sub1,sub2 in combinations(WGCseg,2))
        new_dice['subid'] = range(len(WGCseg)*(len(WGCseg)-1)//2)
        new_dice['Tract'] = 'WGC'
        new_dice['Method'] = 'DTITK'
        dice = pd.concat([dice,new_dice])
        print(new_dice)
        dice.to_csv(f'./dice_{dataset}.csv')
        new_dice.to_csv(f'./dice_{dataset}_WGC.csv')
        


