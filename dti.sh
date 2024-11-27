export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/NA-MIC/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/qt-loadable-modules
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib/Slicer-5.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib/Teem-1.12.0

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib/Python/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data01/software/slicer/Slicer-5.2.2-linux-amd64/lib/Slicer-5.2/cli-modules

/data01/software/slicer/Slicer-5.2.2-linux-amd64/NA-MIC/Extensions-31382/SlicerDMRI/lib/Slicer-5.2/cli-modules/DWIToDTIEstimation $1 --mask $2 $3 $4 -e WLS
