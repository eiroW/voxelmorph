export DTITK_ROOT=/data05/learn2reg/dtitk-2.3.1-Linux-x86_64
export PATH=${PATH}:${DTITK_ROOT}/bin:${DTITK_ROOT}/utilities:${DTITK_ROOT}/scripts

cd $1
cp /data04/junyi/Datasets/DTI/HCP_train/mean_diffeomorphic_initial6.nii.gz ./template.nii.gz
cp /data04/junyi/Datasets/DTI/HCP_train/mask.nii.gz ./
dti_rigid_sn template.nii.gz sub.txt NMI
dti_diffeomorphic_sn template.nii.gz sub_affine.txt ./mask.nii.gz 6 0.002
