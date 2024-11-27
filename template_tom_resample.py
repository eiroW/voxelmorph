import argparse
import nibabel as nib
import numpy as np

def main():
    #-----------------
    # Parse arguments
    #-----------------
    parser = argparse.ArgumentParser(
        description="Dipy tractography",
        epilog="Written by Jianzhong He, vsmallerx@gmail.com. ")
    parser.add_argument("-v", "--version",
        action="version", default=argparse.SUPPRESS,
        version='1.0',
        help="Show program's version number and exit")
    parser.add_argument(
        'input',
        help='dti name.')
    parser.add_argument(
        'output',
        help='output name.')
    parser.add_argument(
        '--pad', action="store_true",
        help='nopad')

    args = parser.parse_args()
    
    img = nib.load(args.input)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    # data = np.transpose(data, [1,2,3,0])

    if args.pad:
        print(data.shape)
        npad = ((48, 48), (0, 0), (0, 0), (0, 0))
        data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)


    for idx in range(3):
        data_ch = data[..., idx]
        padded_data_ch = nib.Nifti1Image(data_ch.astype(np.float32), np.eye(4))

        print(padded_data_ch.shape)
        nib.save(padded_data_ch, args.output.replace('.nii.gz', str(idx)+'.nii.gz'))
	
if __name__ == '__main__':
    main()
