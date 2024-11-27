export DTITK_ROOT=/data05/learn2reg/dtitk-2.3.1-Linux-x86_64
export PATH=${PATH}:${DTITK_ROOT}/bin:${DTITK_ROOT}/utilities:${DTITK_ROOT}/scripts

cd $1
fsl_to_dtitk $2