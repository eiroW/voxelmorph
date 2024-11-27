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
        help='masl name.')

    args = parser.parse_args()
    
    img = nib.load(args.input)
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    print(data.shape)
    if len(data.shape) == 5:
        npad = ((0, 0), (48, 48), (0, 0), (0, 0), (0, 0))
    else:
        npad = ((0, 0), (48, 48), (0, 0))
        data = np.flip(data, 0)

    data = np.pad(data, pad_width=npad, mode='constant', constant_values=0)
    print(data.shape)

    padded_img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
    print(padded_img.affine)
    nib.save(padded_img, args.output)
	
if __name__ == '__main__':
    main()
