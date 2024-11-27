import subprocess
import nibabel as nib
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from fibergeneration import fibergen
DTITK_ROOT= '/data05/learn2reg/dtitk-2.3.1-Linux-x86_64'


def combine_trans(SubID2):
    cmd2 = f'{DTITK_ROOT}/bin/dfRightComposeAffine -aff {OUTPUTDIR}/{SubID2}_dtitk.aff -df {OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo.df.nii.gz -out {OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo_combined.df.nii.gz '
    subprocess.run(cmd2,shell=True)

def warp_fa(SubID2,source_fa):
    subprocess.run(f'mkdir {OUTPUTDIR}/{SubID2}_tmp',shell=True)
    fa = nib.load(source_fa)
    fa_data = fa.get_fdata()
    fa_affine = fa.affine
    nib.save(nib.Nifti1Image(fa_data, fa_affine), f'{OUTPUTDIR}/{SubID2}_tmp/{SubID2}_FA.nii.gz')
    # cmd1 = f'python ./template_resample.py {source_fa} {OUTPUTDIR}/{SubID2}_tmp/{SubID2}_resample_fa.nii.gz'
    cmd3 = f' {DTITK_ROOT}/bin/deformationScalarVolume -in {OUTPUTDIR}/{SubID2}_tmp/{SubID2}_FA.nii.gz -out {OUTPUTDIR}/{SubID2}_fa_warped.nii.gz -trans {OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo_combined.df.nii.gz'
    subprocess.run(cmd3,shell=True)
    subprocess.run(f'rm -r {OUTPUTDIR}/{SubID2}_tmp',shell=True)
def warp_seg(SubID2,target_seg):
    target_seg = Path(target_seg)
    seg_name = target_seg.name
    subprocess.run(f'mkdir {OUTPUTDIR}/{SubID2}_tmp_seg',shell=True)
    seg = nib.load(target_seg)
    seg_data = seg.get_fdata()
    seg_affine = seg.affine
    nib.save(nib.Nifti1Image(seg_data, seg_affine), f'{OUTPUTDIR}/{SubID2}_tmp_seg/{seg_name}')
    new_name = seg_name.replace('.nii.gz','_warped.nii.gz')
    cmd3 = f' {DTITK_ROOT}/bin/deformationScalarVolume -in {OUTPUTDIR}/{SubID2}_tmp_seg/{seg_name} -out {OUTPUTDIR}/{SubID2}_tmp_seg/{new_name} -trans {OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo_combined.df.nii.gz -interp 1'
    subprocess.run(cmd3,shell=True)
    # subprocess.run(f'rm -r {OUTPUTDIR}/{SubID2}_tmp_seg',shell=True)

def warp_TOM(SubID2,source_tom):
    tom_path = Path(source_tom)
    subprocess.run(f'mkdir {OUTPUTDIR}/{SubID2}_tmp_TOM',shell=True)
    subprocess.run(f'mkdir {OUTPUTDIR}/{SubID2}_TOM',shell=True)
    if not Path(f'{OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo_combined.df.nii.gz').exists():
        print(f'No combined deformation field found for {SubID2}')
        return
    for tom in tom_path.iterdir():
        tom_name = tom.name
        tom_data = nib.load(tom)
        tom_affine = tom_data.affine
        tom_data = tom_data.get_fdata()
        split_tom = [tom_data[...,i] for i in range(tom_data.shape[-1])]
        split_tom_name = [tom_name.replace('.nii.gz',f'_{i}.nii.gz') for i in range(tom_data.shape[-1])]
        for i in range(tom_data.shape[-1]):
            nib.save(nib.Nifti1Image(split_tom[i], tom_affine), f'{OUTPUTDIR}/{SubID2}_tmp_TOM/{split_tom_name[i]}')
            new_name = split_tom_name[i].replace('.nii.gz','_warped.nii.gz')
            cmd3 = f'{DTITK_ROOT}/bin/deformationScalarVolume -in {OUTPUTDIR}/{SubID2}_tmp_TOM/{split_tom_name[i]} -out {OUTPUTDIR}/{SubID2}_tmp_TOM/{new_name} -trans {OUTPUTDIR}/{SubID2}_dtitk_aff_diffeo_combined.df.nii.gz'
            print(cmd3)
            # subprocess.run(cmd3,shell=True)
        
        # back_tom = [nib.load(f'{OUTPUTDIR}/{SubID2}_tmp_TOM/{split_tom_name[i].replace(".nii.gz","_warped.nii.gz")}').get_fdata() for i in range(tom_data.shape[-1])]

        # back_tom = np.stack(back_tom,axis=-1)

        # affine = np.diag([1.25,1.25,1.25,1])
        # nib.save(nib.Nifti1Image(back_tom, affine), f'{OUTPUTDIR}/{SubID2}_TOM/{tom_name.replace(".nii.gz","_warped.nii.gz")}')
    subprocess.run(f'rm -r {OUTPUTDIR}/{SubID2}_tmp_TOM',shell=True)
    # fibergen(f'{OUTPUTDIR}/{SubID2}_TOM/',f'{OUTPUTDIR}/{SubID2}_TOM/',SubID2)

        
    
def warp_pipeline(SubID2):
    combine_trans(SubID2)
    warp_fa(SubID2,f'{ROOTPATH}/{SubID2}/T1w/Diffusion/FA_MNI.nii.gz')
    tract_seg_path = Path(f'{ROOTPATH}/{SubID2}/T1w/Diffusion/tractseg_output/seg_new/')
    for seg in tract_seg_path.iterdir():
        warp_seg(SubID2,seg)
    warp_seg(SubID2,f'{ROOTPATH}/segmentations/{SubID2}/SegmentationMap_GMWMCSF_MNI.nii.gz')
    warp_TOM(SubID2,f'{ROOTPATH}/{SubID2}/T1w/Diffusion/tractseg_output/TOM_new/')

if __name__ == '__main__':
    OUTPUTDIR = '/data04/junyi/Datasets/DTI/HCP_test/'
    ROOTPATH = '/data04/junyi/Datasets/HCP_test/'
    sub_list = sorted([p.name.replace('_dtitk.nii.gz','') for p in Path(OUTPUTDIR).iterdir() if '_dtitk.nii.gz' in p.name] )
    print(sub_list)
    Parallel(n_jobs=10)(delayed(warp_pipeline)(sub) for sub in sub_list)