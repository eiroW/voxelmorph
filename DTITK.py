import subprocess
import os
import pathlib
import shutil
from multiprocessing import Pool
import tractseg
import conversion
# os.environ('DTITK_ROOT')='/data05/learn2reg/dtitk-2.3.1-Linux-x86_64'
# os.environ('PATH') = os.environ('PATH') + ':' + os.environ('DTITK_ROOT') + '/bin:' + os.environ('DTITK_ROOT') + '/utilities:' + os.environ('DTITK_ROOT') + '/scripts'

def mksubtxt(subtxt,num):
    direct = subtxt.parent
    print(direct)
    path_list = sorted([p for p in direct.iterdir() if '_dtitk.nii.gz' in p.name])[:num]
    with open(subtxt, 'w') as f:
        for p in path_list:
            f.write(f'{p.name}\n')
def mksubtxt2(subtxt,num):
    direct = subtxt.parent
    print(direct)
    path_list = sorted([p for p in direct.iterdir() if '_dtitk.nii.gz' in p.name])[:num]
    with open(subtxt, 'w') as f:
        for p in path_list:
            new_name = p.name.replace('_dtitk.nii.gz','_dtitk_aff.nii.gz')
            f.write(f'{new_name}\n')
def extract_dti():
    path_list = [p for p in dataset_path.iterdir() if 'seg' not in p.name]
    Pool(10).map(extract_dti_single, path_list)
def extract_dti_single(p):
    name = p.name
    dwi_path = p / 'Diffusion_MNI.nii.gz'
    bvec_path = p / 'Diffusion_MNI.bvecs'
    bval_path = p / 'Diffusion_MNI.bvals'
    mask_path = p / 'mask_MNI.nii.gz'
    cmd = f"dtifit -k {dwi_path} -o {OUTPUT_DIR}/{name} -m {mask_path} -r {bvec_path} -b {bval_path} "
    print(cmd)
    subprocess.run(cmd, shell=True)
    cmd = (" ").join(['sh', './run_dtitk.sh', str(OUTPUT_DIR), p.name,])
    print(cmd)
    subprocess.run(cmd, shell=True)

def register_mask():
    path_list = [p for p in pathlib.Path('/data04/junyi/Datasets/datasets/HCP_100/').iterdir() if 'seg' not in p.name]
    Pool(10).map(register_mask_single, path_list)
def register_mask_single(p):
    name = p.name
    raw_mask = p / 'T1w/Diffusion/nodif_brain_mask.nii.gz'
    raw_data = p / 'T1w/Diffusion/Diffusion_MNI.nii.gz'
    MNI_path = pathlib.Path(tractseg.__file__).parent /'resources/MNI_FA_template.nii.gz'
    subprocess.run(['flirt',
                        '-ref', str(MNI_path),
                        '-in', str(raw_mask),
                        '-out', str(raw_data.parent / 'mask_MNI.nii.gz'),
                        '-applyxfm',
                        '-init', str(raw_data.parent / 'FA_2_MNI.mat'),
                        ])

# dataset_path = pathlib.Path('/data04/junyi/Datasets/ABCD_Reg/data/DWI/')
# OUTPUT_DIR = pathlib.Path('/data04/junyi/Datasets/DTI/ABCD/')
# OUTPUT_DIR.mkdir(exist_ok=True)
# extract_dti()
mksubtxt(pathlib.Path('/data04/junyi/Datasets/DTI/PPMI/sub.txt'),num=100)
mksubtxt2(pathlib.Path('/data04/junyi/Datasets/DTI/PPMI/sub_affine.txt'),num=100)