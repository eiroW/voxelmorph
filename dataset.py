import nibabel as nib
import pathlib
from torch.utils.data import Dataset,DataLoader
import numpy as np
import logging
import os
from multiprocessing import Pool
logger = logging.getLogger()

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']

class FiberDatasetDir(Dataset):
    def __init__(self, img_data_dir: pathlib.Path, backbone_tracts: list,transform=None, FA_only=True):

        self.img_data_dir = sorted(list(pathlib.Path(img_data_dir).iterdir()))
        self.transform = transform
        tmp = nib.load(list(self.img_data_dir[0].glob('**/FA_MNI*'))[0])
        self.config = (tmp.affine,tmp.header)
        self.shape = tmp.get_fdata().shape
        self.FA_only = FA_only
        data_tmp = pathlib.Path('/data04/junyi/tmp')
        self.backbone_tracts = backbone_tracts
        if not data_tmp.exists():
            os.mkdir(data_tmp)
        logger.info('Data preprocess start...')

        p = Pool(10)
        for i, img_data_path in enumerate(self.img_data_dir) :
            data_tmp_path =data_tmp/f'data{i:04d}.npz'
            logger.info(f'Saving {img_data_path} at {data_tmp_path}...')
            if not data_tmp_path.exists():
                p.apply_async(np.savez,(data_tmp_path,),kwds=self._pre_load(img_data_path))    
        p.close()
        p.join()

        logger.info('Data preprocess done')
        self.data_tmp = sorted(list(data_tmp.iterdir()))
        

    def _pre_load(self,image_dir:pathlib.Path):
        struct_dict = {}
        FA_path = sorted(list(image_dir.glob('**/FA_MNI*')))[0]
        struct_dict['FA'] = self.transform(nib.load(FA_path).get_fdata()[::-1],(128, 160, 128))[np.newaxis, ...]
        if not self.FA_only:
            for tract in self.backbone_tracts:
                logger.debug(tract)
                TOM_path = sorted(list((image_dir/'T1w/Diffusion/tractseg_output/TOM_new/').glob(f'{tract}.nii.gz')))[0]
                struct_dict[tract] = self.transform(nib.load(TOM_path).get_fdata()[::-1],(128, 160, 128)).transpose(-1,0,1,2)
        return struct_dict


    def __len__(self):
        return len(self.img_data_dir)

    def __getitem__(self, idx):
        images_dir_path = self.data_tmp[idx]
        logger.debug(images_dir_path)
        
        struct_dict = {key:item for key,item in np.load(images_dir_path).items() if key in self.backbone_tracts}if not self.FA_only else {}
        struct_dict['FA'] = np.load(images_dir_path)['FA']
        logger.debug(struct_dict.keys())
        return struct_dict
    
def center_crop(image,size:tuple):
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        raw_shape = np.array(image.shape)
    elif len(image.shape) == 4:
        assert image.shape[-1] in [1,3]
        raw_shape = np.array(image.shape[:-1])
    else:
        raise ValueError
    indice_down = raw_shape//2 -np.array(size)//2
    indice_up = raw_shape//2 +np.array(size)//2
    return image[indice_down[0]:indice_up[0],indice_down[1]:indice_up[1],indice_down[2]:indice_up[2]]

def main():

    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    fd = FiberDatasetDir('/data01/junyi/datasets/HCP_100',backbone_tracts,center_crop,FA_only=False)
    dataloader = DataLoader(fd,2,num_workers=2)
    next(iter(dataloader))
if __name__=='__main__':
    main()