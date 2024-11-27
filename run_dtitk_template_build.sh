export DTITK_ROOT=/data05/learn2reg/dtitk-2.3.1-Linux-x86_64
export PATH=${PATH}:${DTITK_ROOT}/bin:${DTITK_ROOT}/utilities:${DTITK_ROOT}/scripts

cd $1

# TVMean -in sub10.txt -out mean_initial.nii.gz
# TVResample

# TVResample -in mean_initial.nii.gz -align center -size 128 128 64 -vsize 1.5 1.75 2.25

dti_affine_population mean_initial.nii.gz sub.txt EDS 3
TVtool -in mean_affine3.nii.gz -tr
BinaryThresholdImageFilter mean_affine3_tr.nii.gz mask.nii.gz 0.01 100 1 0
dti_diffeomorphic_population mean_affine3.nii.gz sub_aff.txt mask.nii.gz 0.002