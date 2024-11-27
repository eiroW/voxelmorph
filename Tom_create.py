import os
import pathlib
from typing import Union
from tqdm import tqdm
import tractseg
import nibabel as nib
from multiprocessing import Pool
import numpy as np
import subprocess
import io
import re
import pandas as pd

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
# os.environ['CUDA_VISIBLE_DEVISES'] = "0"
curpath = pathlib.Path('/data04/junyi/Datasets/HCP_new')

def download_data():
    curpath.mkdir(exist_ok=True, parents=True)
    with open('HCP_100_unrelated.txt') as f:
        ids = f.readlines()
    count = 0
    hcp_1200 = pd.read_csv('./HCP_S1200_demographics_Restricted.csv')
    # drop hcp_100 subjects
    hcp_1200 = hcp_1200[~hcp_1200['Subject'].isin(ids)]
    # banlanced sample male and female with similiar age

    hcp_1200 = hcp_1200.sample(frac=1)
    hcp_male = hcp_1200[hcp_1200['Gender']=='M']
    hcp_female = hcp_1200[hcp_1200['Gender']=='F']
    sublist = []
    sub_age_list = []
    for i in range(30):
        sub = hcp_male.iloc[i]
        sub_name = sub['Subject']
        subage = sub['Age_in_Yrs']
        if subage in sub_age_list:
            continue
        try :
            sub_f = hcp_female[hcp_female['Age_in_Yrs']==subage].iloc[0]
        except IndexError:
            continue
        sub_f_name = sub_f['Subject']
        sublist.append(sub_name)
        sublist.append(sub_f_name)
        sub_age_list.append(subage)

    for sub in sublist:
        if pathlib.Path(curpath/str(sub)).exists():
            continue
        print('Downloading',sub)
        subprocess.run(f'scp -r junyi@node2:/data02/HCP_3T/Diffusion_Preprocessed/{sub} {curpath}', shell=True)

def raw2TOM(individual: pathlib.Path,device: Union[int, str] = '0'):
    name = individual.name
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    raw_data = list(individual.glob('**/*data.nii.gz'))
    assert len(raw_data) == 1
    raw_data = raw_data[0]
    MNI_path = pathlib.Path(tractseg.__file__).parent /'resources/MNI_FA_template.nii.gz'
    raw_bvals = list(individual.glob(f'**/bvals'))
    
    assert len(raw_bvals) == 1
    raw_bvals = raw_bvals[0]
    
    raw_bvecs = list(individual.glob(f'**/bvecs'))
    assert len(raw_bvecs) == 1
    raw_bvecs = raw_bvecs[0]
    
    raw_mask = list(individual.glob('**/*mask.nii.gz'))
    assert len(raw_mask) == 1
    raw_mask = raw_mask[0]
    print(raw_data,raw_bvals,raw_bvecs,raw_mask)
    if not (raw_data.parent / 'FA.nii.gz').exists():
        subprocess.run(['calc_FA',
                        '-i', str(raw_data),
                        '-o', str(raw_data.parent / 'FA.nii.gz'),
                        '--bvals', str(raw_bvals),
                        '--bvecs', str(raw_bvecs),
                        '--brain_mask',
                        str(raw_mask)])
    
    if not (raw_data.parent / 'mask_MNI.nii.gz').exists():
        subprocess.run(['flirt', 
                        '-ref', str(MNI_path),
                        '-in', str(raw_data.parent / 'FA.nii.gz'),
                        '-out', str(raw_data.parent / 'FA_MNI.nii.gz'),
                        '-omat', str(raw_data.parent / 'FA_2_MNI.mat'), 
                        '-dof', '6', 
                        '-cost', 'mutualinfo', 
                        '-searchcost', 'mutualinfo'],)
        subprocess.run(['flirt', 
                        '-ref', str(MNI_path), 
                        '-in', str(raw_data), 
                        '-out', str(raw_data.parent / 'Diffusion_MNI.nii.gz'), 
                        '-applyxfm', 
                        '-init', str(raw_data.parent / 'FA_2_MNI.mat'), 
                        '-dof', '6'],)
        subprocess.run(['flirt',
                        '-ref', str(MNI_path),
                        '-in', str(raw_mask),
                        '-out', str(raw_data.parent / 'mask_MNI.nii.gz'),
                        '-applyxfm',
                        '-init', str(raw_data.parent / 'FA_2_MNI.mat'),
                        ])
        subprocess.run(['cp', 
                        str(raw_bvals), 
                        str(raw_data.parent / 'Diffusion_MNI.bvals')],)
        subprocess.run(['rotate_bvecs', 
                        '-i', str(raw_bvecs), 
                        '-t', str(raw_data.parent / 'FA_2_MNI.mat'), 
                        '-o', str(raw_data.parent / 'Diffusion_MNI.bvecs')],)
    
    MNI_data = raw_data.parent / 'Diffusion_MNI.nii.gz'
    if not is_completed(raw_data.parent):
        print(MNI_data.parent)
        subprocess.run(['rm', '-r', str(MNI_data.parent / 'tractseg_output')])
        subprocess.run(['TractSeg', 
                        '-i', str(MNI_data), 
                        '-o', str(MNI_data.parent / 'tractseg_output'),
                        '--bvals', str(MNI_data.parent / 'Diffusion_MNI.bvals'),
                        '--bvecs', str(MNI_data.parent / 'Diffusion_MNI.bvecs'),
                        '--raw_diffusion_input', 
                        '--brain_mask', str(MNI_data.parent / 'mask_MNI.nii.gz'),
                        ],)
        subprocess.run(['TractSeg', '-i', str(MNI_data.parent / 'tractseg_output/peaks.nii.gz'), 
                        '-o', str(MNI_data.parent / 'tractseg_output'),
                        '--output_type', 'endings_segmentation',
                        
                        ],)
        subprocess.run(['TractSeg', 
                        '-i', str(MNI_data.parent / 'tractseg_output/peaks.nii.gz'), 
                        '-o', str(MNI_data.parent / 'tractseg_output'),
                        '--output_type', 'TOM',
                        
                        ],)
def is_completed(individual: pathlib.Path):
    peaks_exist = (individual / 'tractseg_output/peaks.nii.gz').exists()
    endings = (individual / 'tractseg_output/endings_segmentation')
    TOM = (individual / 'tractseg_output/TOM')
    seg = (individual / 'tractseg_output/bundle_segmentations')
    if not endings.exists() or not TOM.exists() or not seg.exists():
        return False
    full = len(list(endings.iterdir())) == 72 and len(list(TOM.iterdir())) == 72 and len(list(seg.iterdir())) == 72
    print(peaks_exist, full)
    return peaks_exist and full


def tck2vtk(tract:str, loaddir: str , savedir: str):
    tom_dir = sorted(list(pathlib.Path(loaddir).glob(f'**/TOM/{tract}_left*')))
    seg_dir = sorted(list(pathlib.Path(loaddir).glob(f'**/bundle_segmentations/{tract}_left*')))
    tom_trk_dir = pathlib.Path(savedir)/f'tracts/{tract}ref'
    
    if not tom_trk_dir.exists():
        os.mkdir(tom_trk_dir)
        for i, tom_trk in enumerate(tom_dir):
            print(f'tckgen -algorithm FACT {tom_trk} {tom_trk_dir}/{tract}{i:03d}.tck -minlength 40 -maxlength 250 -select 2000 -force -nthreads 24 -seed_image {seg_dir[i]} -mask {seg_dir[i]}')
            subprocess.run(f'tckgen -algorithm FACT {tom_trk} {tom_trk_dir}/{tract}{i:03d}.tck -minlength 40 -maxlength 250 -select 2000 -force -nthreads 24 -seed_image {seg_dir[i]} -mask {seg_dir[i]}' ,shell=True)
    
    if not (pathlib.Path(savedir)/f'vtk/{tract}ref').exists():
        os.mkdir(pathlib.Path(savedir)/f'vtk/{tract}ref')
    for tom_trk in tom_trk_dir.iterdir():
        if tom_trk.suffix in ['.tck']:
            subprocess.run(f'tckconvert {tom_trk} {savedir}/vtk/{tract}ref/{tom_trk.stem}.vtk -force',shell=True)

def tract_concat(tract_dir: pathlib.Path):
    TOM_dir = tract_dir/'TOM'
    seg_dir = tract_dir/'bundle_segmentations'
    end_seg_dir = tract_dir/'endings_segmentations'
    (tract_dir/'TOM_new').mkdir(exist_ok=True)
    (tract_dir/'seg_new').mkdir(exist_ok=True)
    (tract_dir/'end_seg_new').mkdir(exist_ok=True)
    
    for tract in backbone_tracts:
        seg_list = load_tract(seg_dir,tract)
        TOM_list = load_tract(TOM_dir,tract)
        # end_seg_list = load_end_seg(end_seg_dir,tract,'e')
        # begin_seg_list = load_end_seg(end_seg_dir,tract,'b')
        if len(TOM_list)!=0 and seg_list!=0:
            # print(len(TOM_list))
            affine,header= TOM_list[0].affine,TOM_list[0].header
            
            TOM = np.sum([tom.get_fdata() for tom in TOM_list],axis=0)
            seg = np.logical_or.reduce([seg.get_fdata() for seg in seg_list])
            print(TOM.shape,seg.shape)
            nib.save(nib.Nifti1Image(TOM, affine, header),
                    tract_dir/'TOM_new'/f'{tract}.nii.gz')
            nib.save(nib.Nifti1Image(seg, affine, header),
                    tract_dir/'seg_new'/f'{tract}.nii.gz')
        # if len(end_seg_list)!=0 and len(begin_seg_list)!=0:
        #     end_seg = np.logical_or.reduce([seg.get_fdata() for seg in end_seg_list])
        #     nib.save(nib.Nifti1Image(end_seg, affine, header),
        #             tract_dir/'end_seg_new'/f'{tract}_e.nii.gz')
        #     beg_seg = np.logical_or.reduce([seg.get_fdata() for seg in begin_seg_list])
        #     nib.save(nib.Nifti1Image(beg_seg, affine, header),
        #             tract_dir/'end_seg_new'/f'{tract}_b.nii.gz')
def load_tract(path: pathlib.Path,tract:str):
    try:
        seg_list = [nib.load(seg) for seg in [path/f'{tract}_left.nii.gz',path/f'{tract}_right.nii.gz']]
    except FileNotFoundError: 
            seg_list = [nib.load(path/f'{tract}.nii.gz')]
    # seg = np.logical_or.reduce([seg.get_fdata() for seg in seg_list])
    return seg_list
def load_end_seg(path: pathlib.Path,tract:str,type='b'):
    
    try:
        seg_list = [nib.load(seg) for seg in [path/(f'{tract}_left_{type}.nii.gz'),path/f'{tract}_right_{type}.nii.gz']]

    except FileNotFoundError: 
            seg_list = [nib.load(path/f'{tract}_{type}.nii.gz')]
    # seg = np.logical_or.reduce([seg.get_fdata() for seg in seg_list])
    return seg_list
            
if __name__ == '__main__':
    import sys
    
    from joblib import Parallel, delayed
    # download_data()



    data_dir = pathlib.Path('/data04/junyi/Datasets/HCP_new')
    # Create TOMs parallelly
    # Attention: VRAM limited.
    njobs = 8
    ngpus = 2
    p = Pool(njobs)
    for i,individual in enumerate(data_dir.iterdir()):
        # raw2TOM(individual,)
        if individual.is_dir() and 'seg' not in individual.name:
            p.apply_async(raw2TOM,(individual,i%ngpus+1))
    p.close()
    p.join()
    
    # Concat TOMs parallelly
    p = Pool(20)
    for TOM_dir in sorted(data_dir.glob('**/tract*/')):
        p.apply_async(tract_concat,(TOM_dir,))
    p.close()
    p.join()