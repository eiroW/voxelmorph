import subprocess
from pathlib import Path
from joblib import Parallel, delayed
from tractseg.libs import img_utils
import re
@delayed
def fibergen(segpath,TOMpath,tractpath,vtkpath,ending_seg_path=None):
    print(TOMpath,segpath)
    
    tractpath.mkdir(exist_ok=True,parents=True)
    vtkpath.mkdir(exist_ok=True,parents=True)
    tract = TOMpath.name.split('.')[0]
    print(tract)
    tmp_dir = TOMpath.parent/'tmp'/tract
    # tmp_dir.mkdir(exist_ok=True)
    fixel_dir = tmp_dir/'fixel'
    fixel_dir.mkdir(exist_ok=True,parents=True)

    if not (fixel_dir/'amplitudes.nii.gz').exists() or not (fixel_dir/'directions.nii.gz').exists() or not (fixel_dir/'index.nii.gz').exists():
        img_utils.peaks2fixel(TOMpath, fixel_dir)
    nthreads = 8
    if not (fixel_dir/'sh.nii.gz').exists():
        subprocess.run(['fixel2sh',
                        f'{fixel_dir}/amplitudes.nii.gz',
                        f'{fixel_dir}/sh.nii.gz',
                        '-quiet',
                        ])
    if ending_seg_path is None:
        subprocess.run(['tckgen',
                        '-algorithm', 'iFOD2',
                        f'{fixel_dir}/sh.nii.gz',
                        f"{tractpath}/{tract}.tck",
                        '-seed_image', f'{segpath}/{tract}_warped.nii.gz',
                        '-mask', f'{segpath}/{tract}_warped.nii.gz',
                        '-minlength', '40',
                        '-maxlength', '250',
                        '-select', '2000',
                        '-nthreads', f'{nthreads}',
                        '-force',
                        '-quiet',
                        ])
        print(f'tckgen -algorithm iFOD2 {fixel_dir}/sh.nii.gz {tractpath}/{tract}.tck -seed_image {segpath}/{tract}_warped.nii.gz -mask {segpath}/{tract}_warped.nii.gz -minlength 40 -maxlength 250 -select 2000 -nthreads {nthreads} -force -quiet')
    else:
        subprocess.run(['tckgen',
                        '-algorithm', 'iFOD2',
                        f'{fixel_dir}/sh.nii.gz',
                        f"{tractpath}/{tract}.tck",
                        '-seed_image', f'{segpath}/{tract}.nii.gz',
                        '-mask', f'{segpath}/{tract}.nii.gz',
                        '-include', f'{ending_seg_path}/{tract}_b.nii.gz',
                        '-include', f'{ending_seg_path}/{tract}_e.nii.gz',
                        '-minlength', '40',
                        '-maxlength', '250',
                        '-select', '2000',
                        '-nthreads', f'{nthreads}',
                        '-force',
                        '-quiet',
                        ])
    # subprocess.call("tckgen -algorithm iFOD2 " +
    #                 f'{tmp_dir}' + "/fixel/sh.nii.gz " +
    #                 f"{tractpath}/{tract}.tck" +
    #                 " -seed_image " + f'{segpath}/{tract}.nii.gz' +
    #                 " -mask " + f'{segpath}/{tract}.nii.gz' +
    #                 '-include', f'{ending_seg_path}/{tract}_b.nii.gz' +
    #                 '-include', f'{ending_seg_path}/{tract}_e.nii.gz' +
    #                 " -minlength 40 -maxlength 250 -select " + str(2000) +
    #                 " -force -quiet"
    #                 + f'-nthreads {nthreads}',
    #                 shell=True)
    # img_utils.
    # subprocess.run(['tckgen',
    #                 '-algorithm', 'FACT',
    #                 f'{TOMpath}',
    #                 f'{tractpath}/{tract}.tck',
    #                 '-minlength', '40',
    #                 '-maxlength', '250',
    #                 '-select', '2000',
    #                 '-nthreads', '4',
    #                 '-seed_image', f'{segpath}/{tract}.nii.gz',
    #                 '-mask', f'{segpath}/{tract}.nii.gz',
    #                 # '-include', f'{ending_seg_path}/{tract}_b.nii.gz',
    #                 # '-include', f'{ending_seg_path}/{tract}_e.nii.gz',
    #                 '-force',
    #                 '-quiet'
    #                 ])
    subprocess.run(f'tckconvert {tractpath}/{tract}.tck {vtkpath}/{tract}.vtk -force',shell=True)
def individual(segpath,TOMpath,tractpath,vtkpath,ending_seg_path=None):
    
    Parallel(n_jobs=24)(fibergen(segpath,TOMpath/tract,tractpath,vtkpath,ending_seg_path) for tract in sorted(TOMpath.iterdir()) if tract.suffix == '.gz')
def main():
    # root_dir = Path('/data04/junyi/Datasets/ABCD_Reg/data/DWI')
    # output_dir = Path('/data04/junyi/Datasets/tracts_ABCD/')
    # individual_paths = [path/'tractseg_output/' for path in sorted(root_dir.iterdir()) if 'seg' not in path.name]
    # individual_names = [path.parent.name for path in individual_paths if path.is_dir()]
    # # # individual_paths = [path/'T1w/Diffusion/tractseg_output/' for path in sorted(root_dir.iterdir()) if 'seg' not in path.name]
    # # # individual_names = [path.name for path in sorted(root_dir.iterdir()) if 'seg' not in path.name]
    
    
    # TOMpaths = [path/'TOM' for path in individual_paths if path.is_dir()]
    # segpaths = [path/'bundle_segmentations' for path in individual_paths if path.is_dir()]
    # ending_seg_paths = [path/'endings_segmentations' for path in individual_paths if path.is_dir()]
    # tractpaths = [output_dir/name/'tracts' for name in individual_names]
    # vtkpaths = [output_dir/name/'vtk' for name in individual_names]
    # print(individual_paths,tractpaths)

    # # TOMpaths = [path/'TOM' for path in individual_paths if path.is_dir()]
    # # segpaths = [path/'bundle_segmentations' for path in individual_paths if path.is_dir()]
    # # ending_seg_paths = [path/'endings_segmentations' for path in individual_paths if path.is_dir()]
    # # tractpaths = [output_dir/name/'tracts' for name in individual_names]
    # # vtkpaths = [output_dir/name/'vtk' for name in individual_names]
    # Parallel(n_jobs=4)(individual(segpath,TOMpath,tractpath,vtkpath,ending_seg_path) for segpath,TOMpath,tractpath,vtkpath,ending_seg_path in zip(segpaths,TOMpaths,tractpaths,vtkpaths,ending_seg_paths))
    for dataset in ['ABCD','PPMI',]:
        # dataset ='PPMI'
        tom_path = Path(f'/data04/junyi/Datasets/DTI/{dataset}/')
        for tract in tom_path.glob('*_TOM'):
            print(tract)
            seg_path = tom_path/tract.name.replace('_TOM','_tmp_seg')
            vtk_path = tom_path/tract.name.replace('_TOM','_vtk')
            tract_path = tom_path/tract.name.replace('_TOM','_tracts')
            individual(seg_path, tract, tract_path, vtk_path)
if __name__ == '__main__':
    main()