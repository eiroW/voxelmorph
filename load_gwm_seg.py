import tarfile
import os
import pathlib
import subprocess
from joblib import Parallel, delayed


def load_gwm_seg(path:pathlib.Path,save_path,sub_id):
    path = path / f'{sub_id}.tar.gz'
    if not path.exists():
        print(f'{path} not found')
        return None
    with tarfile.open(path, 'r') as tar:
        tar.extract(f'segmentations//{sub_id}/SegmentationMap_GMWMCSF.nii.gz', path=save_path)
        
def apply_transform(seg_path: pathlib.Path,):
    ID = seg_path.name
    transform_path = seg_path.parent.parent/f'{ID}/T1w/Diffusion/FA_2_MNI.mat'
    if transform_path.exists():
        seg_path /= f'SegmentationMap_GMWMCSF.nii.gz'
        if (seg_path.parent/'SegmentationMap_GMWMCSF_MNI.nii.gz').exists():
            return
        subprocess.run(['flirt',
                        '-in', seg_path,
                        '-ref', transform_path.parent/'FA_MNI.nii.gz',
                        '-out', seg_path.parent/'SegmentationMap_GMWMCSF_MNI.nii.gz',
                        '-applyxfm',
                        '-init', transform_path,
                        '-interp', 'nearestneighbour'])

if __name__ == '__main__':
    path = pathlib.Path('/data02/SWM-HCP-S1200/segmentations/')
    save_path = pathlib.Path('/data04/junyi/Datasets/HCP_new')
    # transform = np.loadtxt(transform_path)
    
    for p in save_path.iterdir():
        if p.is_dir() and 'seg' not in p.name:
            load_gwm_seg(path,save_path,p.name)
    Parallel(n_jobs=20)(delayed(apply_transform)(seg_path) for seg_path in (save_path/'segmentations').iterdir())