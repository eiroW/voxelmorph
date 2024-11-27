import nibabel as nib
import numpy as np
import pathlib
import re
import seaborn as sns
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt    
backbone_tracts = (['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
                   )
def heatmap(segpath, tract, output):
    # Load the masks
    # affine = nib.load(segpath[0]).affine
    # header = nib.load(segpath[0]).header
    mask_list = list(map(nib.load,
        sorted(path for path in pathlib.Path(segpath).glob(f'*_tmp_seg/{tract}_warped.nii.gz'))
        # sorted(path for path in pathlib.Path(segpath).glob(f'{tract}*.nii.gz'))
               ))
    affine = mask_list[0].affine
    header = mask_list[0].header
    masks = [mask.get_fdata() for mask in mask_list]
    masks_arr = np.stack(masks, axis=0)
    print(masks_arr.shape)
    union = np.sum(masks_arr>0, axis=0)>0
    volume_sub = np.sum(masks_arr>0, axis=(1,2,3))
    print(volume_sub)
    proportion = volume_sub/union.sum()
    print(proportion.mean())
    # print(f'{(heat_map>0.8).sum()/(heat_map>0).sum() * 100:.2f}% of the voxels are in the majority class')
    # nib.save(nib.Nifti1Image(heat_map, affine, header), output)
    return proportion.mean()
    # np.save(output, np.stack(mask_list, axis=0))
# method = 'alltracts'
methods_list = ['DTITK']
datasets_list = ['HCP_test','ABCD','PPMI',]
df_all = pd.DataFrame(columns=['method','dataset','area','tract'])
for tract in backbone_tracts:
    for method, dataset in product(methods_list, datasets_list):
        # seg_dir = f'/data04/junyi/results/tracts_related/seg_{method}_{dataset}/'
        # print(seg_dir)
        seg_dir = f'/data04/junyi/Datasets/DTI/{dataset}'
        print(f'Processing {method}_{tract}_{dataset}')
        heat_path = f'./heatmap_{method}_{tract}_{dataset}.nii.gz'
        result = heatmap(seg_dir,tract,heat_path)
        df_all = df_all.append({'method':method,'dataset':dataset,'tract':tract,'proportion':result},ignore_index=True)
    df_all.to_csv(f'./New_proportion{method}.csv')
    