import numpy as np
import torch
import functools
from matplotlib import pyplot as plt
import pathlib
import os
import subprocess
from itertools import product
from joblib import Parallel, delayed
from multiprocessing import Pool
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph.torch import networks, layers
import nibabel as nib
from torch.utils.data import DataLoader,Dataset
import logging.config
from tractseg.libs import img_utils
from torch.profiler import profile, record_function, ProfilerActivity


def calculate_time(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kargs)
        end.record()
        torch.cuda.synchronize()
        print(f'The epoch lasted {start.elapsed_time(end) / 1000} s\n')
        return result

    return wrapper


backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
# backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
#                    'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',]
def extract_seg(TOM: np.array, threshold=1e-2):
    """
    Extracts a segmentation mask from a tensor of shape (3, H, W, D) representing a 3D volume.

    Parameters:
        TOM (np.array): Tensor of shape (3, H, W, D) representing a 3D volume.
        threshold (float): Threshold value for segmentation. Default is 1e-2.

    Returns:
        np.array: Segmentation mask of shape (H, W, D) with values of 0 or 1.
    """
    assert TOM.shape[0] == 3
    mask = np.sqrt(np.sum(TOM**2, 0, keepdims=True)) > threshold
    return mask.astype(np.float32)
def load_seg(path: pathlib.Path,tract:str):
    # seg_list = [nib.load(seg) for seg in path.iterdir() if ((f'{tract}_left.nii.gz'==seg.name) or (f'{tract}_right.nii.gz'==seg.name))]
    try:
        seg_list = [nib.load(seg) for seg in [path/f'{tract}_left.nii.gz',path/f'{tract}_right.nii.gz']]
    except FileNotFoundError:
        seg_list = [nib.load(path/f'{tract}.nii.gz')]
    seg = np.logical_or.reduce([seg.get_fdata() for seg in seg_list])
    return seg
    

class HCPFiberDatasetDir(Dataset):
    def __init__(self, 
                 img_data_dir: pathlib.Path, 
                 transform=None, 
                 sub_num=10,
                 suffix='HCP_trained'):
        self.backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
        self.img_data_dir = sorted(path for path in pathlib.Path(img_data_dir).iterdir() if (path.is_dir() & ('seg' not in path.name)))[:sub_num]
        self.transform = transform
        tmp = nib.load(list(self.img_data_dir[0].glob('**/FA_MNI*'))[0])
        self.config = (tmp.affine,tmp.header)
        self.shape = tmp.get_fdata().shape
        self.indice_down = np.array(self.shape)//2-np.array([128//2, 160//2, 128//2])
        self.indice_up =(np.array(self.shape)//2)+np.array([128//2, 160//2, 128//2])
        self.suffix = suffix
    def __len__(self):
        return len(self.img_data_dir)

    def __getitem__(self, idx):
        images_dir_path = self.img_data_dir[idx]
        subid = torch.tensor(int(images_dir_path.name.split('_')[0])) if 'sub' not in images_dir_path.name else torch.tensor(idx)
        struct_dict = {}
        indice_down, indice_up = self.indice_down, self.indice_up
        FA_path = sorted(images_dir_path.glob('**/FA_MNI*'))[0]
        if 'HCP' in self.suffix:
            WGCseg_path = images_dir_path.parent/f'segmentations/{images_dir_path.name}/SegmentationMap_GMWMCSF_MNI.nii.gz'
        else:
            WGCseg_path = images_dir_path/'SegmentationMap_GMWMCSF_MNI.nii.gz'
        struct_dict['FA'] = self.transform(nib.load(FA_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
        struct_dict['WGCseg'] = self.transform(nib.load(WGCseg_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
        for tract in self.backbone_tracts:
            # TOM_path = images_dir_path/'T1w/Diffusion/tractseg_output/TOM_new/'/f'{tract}.nii.gz'
            
            TOM_path = list(images_dir_path.glob(f'**/tractseg_output/TOM_new/{tract}.nii.gz'))
            TOM_path = TOM_path[0] if len(TOM_path)==1 else print(images_dir_path,tract)
            
            struct_dict[tract] = self.transform(nib.load(TOM_path).get_fdata()[::-1],indice_down,indice_up).transpose(-1,0,1,2)
            # load seg of tract
            # seg_path = images_dir_path/'T1w/Diffusion/tractseg_output/seg_new'/f'{tract}.nii.gz'
            seg_path = list(images_dir_path.glob(f'**/tractseg_output/seg_new/{tract}.nii.gz'))
            seg_path = seg_path[0] if len(seg_path)==1 else None
            # begin_seg_path = list(images_dir_path.glob(f'**/tractseg_output/end_seg_new/{tract}_b.nii.gz'))
            # begin_seg_path = begin_seg_path[0] if len(begin_seg_path)==1 else None
            # ending_seg_path = list(images_dir_path.glob(f'**/tractseg_output/end_seg_new/{tract}_e.nii.gz'))
            # ending_seg_path = ending_seg_path[0] if len(ending_seg_path)==1 else None
            struct_dict[tract+'seg'] = self.transform(nib.load(seg_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
            # struct_dict[tract+'segb'] = self.transform(nib.load(begin_seg_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
            # struct_dict[tract+'sege'] = self.transform(nib.load(ending_seg_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
            # struct_dict[tract+'seg'] = extract_seg(struct_dict[tract])
        return struct_dict,subid
    
def crop(x,indice_down, indice_up):
    x = x.astype(np.float32)
    return x[indice_down[0]:indice_up[0],indice_down[1]:indice_up[1],indice_down[2]:indice_up[2]]

class FiberRegistor:
    def __init__(self,
                 tracts: list,
                 suffix: str = '100_alltracts',
                 model_path: str = '/data01/junyi/models/models_sep/snapshot100_alltracts_half_8to2_whole.pt',
                 data_path: str = '/data01/junyi/datasets/HCP_100',
                 device: str = 'cuda:7',
                 batch_size: int = 6,
                 sub_num: int = 10,
                 tracts_num=-1
                 ):
        # initialize file paths
        self.root_dir = '/data04/junyi/results/tracts_related'
        self.device = device
        self.tracts = tracts
        self.tracts_num = tracts_num    
        # initialize model
        self.model_path = model_path
        
        self.vol_shape = (128, 160, 128)
        self.data_dict = {'FA':torch.rand((4,1)+self.vol_shape),}
        for tract in self.tracts:
            self.data_dict[tract] = torch.rand((4,3)+self.vol_shape)
        self.suffix = suffix
        self.batch_size = batch_size
        # intialize Spacial Transformer
        self.bilinST = layers.SpatialTransformer(self.vol_shape,mode='bilinear')
        self.nnST = layers.SpatialTransformer(self.vol_shape,mode='nearest')
        # initialize reoriention layer
        self.reorient = layers.TOMReorientation(self.vol_shape)

    def load_model(self):
        TOM_models = {}
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        if 'FA_only' not in self.suffix:
            atlas = torch.empty(3, *self.vol_shape)
            for tract in self.tracts:
                TOM_models[tract] = networks.TemplateCreation(self.vol_shape,atlas,nb_unet_features=[enc_nf, dec_nf],altas_feats=3,src_feats=3)
            
            atlas = torch.empty(1, *self.vol_shape)
            self.whole_model = networks.WholeModel(self.vol_shape,atlas,TOM_models,nb_unet_features=[enc_nf, dec_nf],altas_feats=1,src_feats=1, FA_only=False, use_TOM=False, stream_len=self.tracts_num)
            self.whole_model.load_state_dict(torch.load(self.model_path,map_location='cpu')["MODEL_STATE"],assign=True)
        else:
            atlas = torch.empty(1, *self.vol_shape)
            self.whole_model = networks.WholeModel(self.vol_shape,atlas,TOM_models,nb_unet_features=[enc_nf, dec_nf],altas_feats=1,src_feats=1,FA_only=True,use_TOM=False)
            self.whole_model.load_state_dict(torch.load(self.model_path,map_location='cpu')["MODEL_STATE"])
        self.subid_list = []
    @calculate_time
    def register(self):
        self.load_model()
        with torch.no_grad():
            model = self.whole_model.eval().to(self.device)
            for i in range(4):
                flows = self.load_flow(i)
                if flows is None:
                    data = {key:im.to(self.device) for key,im in self.data_dict.items()}
                    if i == 1:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            register_FAs,flows = model(data,registration=True,) 
                        prof.export_chrome_trace("trace.json")
                        return 
                    else:
                        register_FAs,flows = model(data,registration=True,)
                    print(f'Computing Flow done')
        print(f'Registration done total SUBs Number: {len(self.FAs_list)}.')
    def load_flow(self,batch_id):
        
        return None
 


def main(registor: FiberRegistor):
    # t_start = timeit.timeit
    registor.register()
    torch.cuda.empty_cache()
    # print(f'Ready for Saving.')
    # registor.save_results()
    # print(f'Saving done.')
    # t_end = timeit.timeite
    # print(f'Total time: {t_end-t_start}')
    
    # p = Pool(20)
    # for i,tract in product(range(registor.sub_num),registor.tracts):
    #     # print(registor.TOMpath,registor.segpath,registor.tractpath,registor.vtkpath)
    #     p.apply_async(fibergen, args=(tract,i,registor.TOMpath,registor.segpath,registor.tractpath,registor.vtkpath))
    # p.close()
    # p.join()
    # [registor.generate_fiber(tract,i) for i,tract in product(range(registor.sub_num),backbone_tracts)]
if __name__ =='__main__':
    registor = FiberRegistor(backbone_tracts,
                             suffix='100_alltracts',
                             model_path='/data04/junyi/models/whole_model.pt',
                             sub_num=4,
                             device='cuda:1',
                             batch_size=4,
                             tracts_num=len(backbone_tracts))
    main(registor)