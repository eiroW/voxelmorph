from nilearn.image import resample_img
from scipy.ndimage import map_coordinates
import nibabel as nib
import numpy as np
def apply_dtitk_transform(subid,img_path):

    transform_path = f'/data04/junyi/Datasets/DTI/PPMI/{subid}_dtitk_aff_diffeo_combined.df.nii.gz'
    transform_img = nib.load(transform_path)
    transform = resample_img(transform_img, target_affine=np.diag([1.25,1.25,1.25,1]), target_shape=(145,174,145)).get_fdata()/np.array([1.25, 1.25, 1.25])[None,None,None,:]
    print(transform[71,86,72])
    img = nib.load(img_path).get_fdata()
    sample_coord = np.indices(img.shape) + transform.transpose(3,0,1,2)
    transformed_img = map_coordinates(img, sample_coord, order=1)
    nib.save(nib.Nifti1Image(transformed_img, np.diag([1.25,1.25,1.25,1])),'test.nii.gz')
    nib.save(nib.Nifti1Image(img, np.diag([1.25,1.25,1.25,1])),'test_mov.nii.gz')

if __name__ == '__main__':
    apply_dtitk_transform('3577_S185756_2012','/data04/junyi/Datasets/PPMI_Reg/data/DWI/3577_S185756_2012/FA_MNI.nii.gz')


