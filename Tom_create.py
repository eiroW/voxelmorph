import os
import pathlib
from typing import Union
from tqdm import tqdm
import tractseg
import nibabel as nib
from multiprocessing import Pool
import numpy as np

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']

# os.environ['CUDA_VISIBLE_DEVISES'] = "0"
curpath = pathlib.Path().absolute()

def download_data():
    if not pathlib.Path('../HCP_100').exists():
        os.mkdir(f'{curpath.parent}/HCP_100')
    with open('HCP_100_unrelated.txt') as f:
        ids = f.readlines()
    for id in ids:
        os.system(f'scp -r junyi@node2:/data02/HCP/{id[:-1]} {curpath.parent}/HCP_100/')

def raw2TOM(individual):
    
    raw_data = individual/'T1w/Diffusion/data.nii.gz'
    MNI_path = pathlib.Path(tractseg.__file__).parent/'resources/MNI_FA_template.nii.gz'
    if not (raw_data.parent/'FA.nii.gz').exists():
        os.system(f'calc_FA -i {raw_data} -o {raw_data.parent}/FA.nii.gz --bvals {raw_data.parent}/bvals --bvecs {raw_data.parent}/bvecs --brain_mask {raw_data.parent}/nodif_brain_mask.nii.gz')
    if  not (raw_data.parent/'Diffusion_MNI.bvecs').exists():
        os.system(f'flirt -ref {MNI_path} -in {raw_data.parent}/FA.nii.gz -out {raw_data.parent}/FA_MNI.nii.gz -omat {raw_data.parent}/FA_2_MNI.mat -dof 6 -cost mutualinfo -searchcost mutualinfo')
        os.system(f'flirt -ref {MNI_path} -in {raw_data} -out {raw_data.parent}/Diffusion_MNI.nii.gz -applyxfm -init {raw_data.parent}/FA_2_MNI.mat -dof 6')
        os.system(f'cp {raw_data.parent}/bvals {raw_data.parent}/Diffusion_MNI.bvals')
        os.system(f'rotate_bvecs -i {raw_data.parent}/bvecs -t {raw_data.parent}/FA_2_MNI.mat -o {raw_data.parent}/Diffusion_MNI.bvecs')
    
    MNI_data  = raw_data.parent/'Diffusion_MNI.nii.gz'
    if  not (raw_data.parent/'tractseg_output/peaks.nii.gz').exists():
        os.system(f'TractSeg -i {MNI_data} -o {MNI_data.parent}/tractseg_output --raw_diffusion_input --brain_mask {MNI_data.parent}/nodif_brain_mask.nii.gz ' )
        os.system(f'TractSeg -i {MNI_data.parent}/tractseg_output/peaks.nii.gz -o {MNI_data.parent}/tractseg_output --output_type endings_segmentation')
        os.system(f'TractSeg -i {MNI_data.parent}/tractseg_output/peaks.nii.gz -o {MNI_data.parent}/tractseg_output --output_type TOM')


def tck2vtk(loaddir: str , savedir: str):
    tom_dir = sorted(list(pathlib.Path(loaddir).glob('**/TOM_new/AF*')))
    seg_dir = sorted(list(pathlib.Path(loaddir).glob('**/bundle_segmentations/AF*')))
    tom_trk_dir = pathlib.Path(savedir)/'tracts/AFref'
    
    if not tom_trk_dir.exists():
        os.mkdir(tom_trk_dir)
        for i,tom_trk in enumerate(tom_dir):
            os.system(f'tckgen -algorithm FACT {tom_trk} {tom_trk_dir}/AF{i:03d}.tck -minlength 40 -maxlength 250 -select 2000 -force -nthreads 24 -seed_image {seg_dir[i]}')
    
    if not (pathlib.Path(savedir)/'vtk/AFref').exists():
        os.mkdir(pathlib.Path(savedir)/'vtk/AFref')
    for tom_trk in tom_trk_dir.iterdir():
        if tom_trk.suffix in ['.tck']:
            os.system(f'tckconvert {tom_trk} {savedir}/vtk/AFref/{tom_trk.stem}.vtk')

def TOMconcat(TOM_dir: pathlib.Path):
    if not pathlib.Path(TOM_dir.parent / 'TOM_new').exists():
        os.mkdir(f'{TOM_dir.parent}/TOM_new')
    for tract in backbone_tracts:
        TOM_list = [nib.load(TOM) for TOM in TOM_dir.iterdir() if ((f'{tract}_left.nii'==TOM.stem) or (f'{tract}_right.nii'==TOM.stem))]
        if len(TOM_list)!=0 :
            print(len(TOM_list))
            affine,header= TOM_list[0].affine,TOM_list[0].header
            
            TOM = TOM_list[0].get_fdata()+TOM_list[1].get_fdata()
            nib.save(nib.Nifti1Image(TOM, affine, header),
                    TOM_dir.parent/'TOM_new'/f'{tract}.nii.gz')
        elif (TOM_dir/f'{tract}.nii.gz').exists():
            
            TOM_path = TOM_dir/f'{tract}.nii.gz'
            cmd = f'cp {TOM_path} {TOM_dir.parent}/TOM_new/'
            print(cmd)
            # os.system(cmd) 

            
if __name__ == '__main__':
    data_dir = pathlib.Path('/data01/junyi/datasets/HCP_100')
    
    p = Pool(5)
    for TOM_dir in sorted(data_dir.glob('**/tract*/TOM')):
        # TOMconcat(TOM_dir,)
        p.apply_async(TOMconcat,(TOM_dir,))
    p.close()
    p.join()
    # tck2vtk('../HCP_100','..')
    pass