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


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('console')

def calculate_time(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kargs)
        end.record()
        torch.cuda.synchronize()
        logger.info(f'The epoch lasted {start.elapsed_time(end) / 1000} s\n')
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
        self.prefixes =['FA','seg','TOM','WGCseg','flow','tract','vtk']
        for attr in self.prefixes:
            if attr=='WGCseg':
                setattr(self, attr+'path', pathlib.Path(f'{self.root_dir}/seg_{suffix}'))
            else:
                setattr(self, attr+'path', pathlib.Path(f'{self.root_dir}/{attr}_{suffix}'))
                getattr(self, attr+'path').mkdir(parents=True, exist_ok=True) if not getattr(self, attr+'path').exists() else None
                
        # initialize model
        self.model_path = model_path
        
        self.vol_shape = (128, 160, 128)
        
        self.suffix = suffix
        
        
        # initialize dataset
        self.dataset = HCPFiberDatasetDir(data_path,crop,sub_num,self.suffix)
        self.sub_num = len(self.dataset)
        self.image_dir = self.dataset.img_data_dir
        self.raw_shape = self.dataset.shape
        self.indice_down, self.indice_up = self.dataset.indice_down, self.dataset.indice_up
        self.affine, self.header = self.dataset.config
        self.dataloader = DataLoader(self.dataset,batch_size,num_workers=10,shuffle=False)
        self.batch_size = batch_size
        # intialize Spacial Transformer
        self.bilinST = layers.SpatialTransformer(self.vol_shape,mode='bilinear')
        self.nnST = layers.SpatialTransformer(self.vol_shape,mode='nearest')
        
        # initialize reoriention layer
        self.reorient = layers.TOMReorientation(self.vol_shape)
        
        # initialize results dict
        self.segs_dict= {tract:[] for tract in tracts}
        self.segs_b_dict= {tract:[] for tract in tracts}
        self.segs_e_dict= {tract:[] for tract in tracts}
        self.TOMs_dict = {tract:[] for tract in tracts}
        self.FAs_list= []
        self.WGC_list= []
        self.flows_list = []
        self.tracts_num = tracts_num

    def load_model(self):
        TOM_models = {}
        enc_nf = [16, 32, 32, 32]
        dec_nf = [32, 32, 32, 32, 32, 16, 16]
        if 'FA_only' not in self.suffix:
            atlas = torch.empty(3, *self.vol_shape)
            for tract in self.tracts[:self.tracts_num]:
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
            for i, (data_dict,subid) in enumerate(self.dataloader):
                flows = self.load_flow(i)
                register_segs ={}
                if flows is None:
                    data = {key:im.to(self.device) for key,im in data_dict.items() if "seg" not in key}
                    # data['FA'].to(self.device)
                    
                    if i == 1:
                        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                            register_FAs,flows = model(data,registration=True,) 
                        prof.export_chrome_trace("trace.json")
                        return 
                    else:
                        register_FAs,flows = model(data,registration=True,)
                    logger.info(f'Computing Flow done')
                else:
                    register_FAs = self.bilinST(data_dict['FA'].float(),flows)
                this_batch_size = flows.shape[0]
                rotMat = self.reorient.to(self.device)(flows)
                flows, rotMat = flows.cpu(), rotMat.cpu()

                for tract in self.tracts:
                    logger.info(f'Registing on {tract}...')
                    register_segs[tract] = self.nnST(data_dict[tract+'seg'].float(),flows)
                    self.segs_dict[tract].extend(register_segs[tract][i].squeeze().numpy()[::-1] for i in range(this_batch_size))
                    tom_reoriented = self.reorient.batched_rotate(rotMat,data_dict[tract].float())
                    tom_registed = self.bilinST(tom_reoriented,flows)
                    self.TOMs_dict[tract].extend(tom_registed[i].permute(1,2,3,0).numpy()[::-1] for i in range(this_batch_size))
                    logger.info(f'Registing on {tract} done')
                self.FAs_list.extend(register_FAs[i].cpu().squeeze().numpy()[::-1] for i in range(this_batch_size))
                WGCseg = data_dict['WGCseg'].float()
                self.WGC_list.extend(self.nnST(WGCseg,flows)[i].squeeze().numpy()[::-1] for i in range(this_batch_size))
                self.flows_list.extend(flows[i].permute(1,2,3,0).numpy()[::-1] for i in range(this_batch_size))
                print(f'SUBID : {subid}')
                self.subid_list.extend([i for i in subid])
        logger.info(f'Registration done total SUBs Number: {len(self.FAs_list)}.')
    def center_padding(self, x,):
        if x.ndim==4:
            zeros = np.zeros((*self.raw_shape,x.shape[-1]))
        if x.ndim==3:
            zeros = np.zeros(self.raw_shape,)
        zeros[self.indice_down[0]:self.indice_up[0],self.indice_down[1]:self.indice_up[1],self.indice_down[2]:self.indice_up[2]] = x
        return zeros
        
    def save_results(self):
        flow_header = self.header.copy()
        vector_header = self.header.copy()
        flow_header['intent_code'] = 1006
        vector_header['intent_code'] = 1007
        
        Parallel(n_jobs=24)(delayed(nib.save)(nib.Nifti1Image(self.center_padding(FA), self.affine, self.header),self.FApath/f'SUB{i:03d}.nii.gz') for i,FA in enumerate(self.FAs_list))
        Parallel(n_jobs=24)(delayed(nib.save)(nib.Nifti1Image(self.center_padding(wgc), self.affine, self.header),self.WGCsegpath/f'WGC{i:03d}.nii.gz') for i,wgc in enumerate(self.WGC_list))
        Parallel(n_jobs=24)(delayed(nib.save)(nib.Nifti1Image(self.center_padding(flow)[...,None,:], self.affine, flow_header),self.flowpath/f'SUB{i:03d}.nii.gz') for i,flow in enumerate(self.flows_list))
        # Save TOMs of each tract
        list_len =self.sub_num
        for tract in self.tracts:
            assert len(self.segs_dict[tract])==list_len, f'{tract} length not match'
            assert len(self.TOMs_dict[tract])==list_len, f'{tract} length not match'
        Parallel(n_jobs=24)(delayed(nib.save)(nib.Nifti1Image(self.center_padding(self.segs_dict[tract][i]), self.affine, self.header),self.segpath/f'{tract}{i:03d}.nii.gz') for i,tract in product(range(list_len),self.tracts))
        Parallel(n_jobs=24)(delayed(nib.save)(nib.Nifti1Image(self.center_padding(self.TOMs_dict[tract][i]), self.affine, self.header),self.TOMpath/f'{tract}{i:03d}.nii.gz') for i,tract in product(range(list_len),self.tracts))
    def load_flow(self,batch_id):
        
        return None
def generate_fiber(tract:str, SubID:int,TOMpath:str,segpath:str,tractpath:str,vtkpath:str):
    print(f'Generating {tract} for {SubID:03d}...')
    subprocess.run(['tckgen',
                    '-algorithm', 'FACT',
                    f'{TOMpath}/{tract}{SubID:03d}.nii.gz',
                    f'{tractpath}/{tract}{SubID:03d}.tck',
                    '-minlength', '40',
                    '-maxlength', '250',
                    '-select', '2000',
                    '-nthreads', '16',
                    '-seed_image', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                    '-mask', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                    # '-include', f'{self.segpath}/{tract}{SubID:03d}_b.nii.gz',
                    # '-include', f'{self.segpath}/{tract}{SubID:03d}_e.nii.gz',
                    '-force',
                    '-quiet'
                    ])
    subprocess.run(f'tckconvert {tractpath}/{tract}{SubID:03d}.tck {vtkpath}/{tract}{SubID:03d}.vtk -force',shell=True)
    print(f'Generating {tract} for {SubID:03d} done.')   

def fibergen(tract:str, SubID:int,TOMpath,segpath,tractpath,vtkpath,ending_seg_path=None):
    tractpath.mkdir(exist_ok=True,parents=True)
    vtkpath.mkdir(exist_ok=True,parents=True)
    # tract = TOMpath.name.split('.')[0]
    # print(tract)
    tmp_dir = tractpath/f'tmp_{SubID:03d}'/tract
    # tmp_dir.mkdir(exist_ok=True)
    fixel_dir = tmp_dir/'fixel'
    fixel_dir.mkdir(exist_ok=True,parents=True)

    if not (fixel_dir/'amplitudes.nii.gz').exists() or not (fixel_dir/'directions.nii.gz').exists() or not (fixel_dir/'index.nii.gz').exists():
        img_utils.peaks2fixel(TOMpath/f'{tract}{SubID:03d}.nii.gz', fixel_dir)
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
                        f"{tractpath}/{tract}{SubID:03d}.tck",
                        '-seed_image', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                        '-mask', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                        '-minlength', '40',
                        '-maxlength', '250',
                        '-select', '2000',
                        '-nthreads', f'{nthreads}',
                        '-force',
                        '-quiet',
                        ])
    else:
        subprocess.run(['tckgen',
                        '-algorithm', 'iFOD2',
                        f'{fixel_dir}/sh.nii.gz',
                        f"{tractpath}/{tract}{SubID:03d}.tck",
                        '-seed_image', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                        '-mask', f'{segpath}/{tract}{SubID:03d}.nii.gz',
                        '-include', f'{ending_seg_path}/{tract}{SubID:03d}_b.nii.gz',
                        '-include', f'{ending_seg_path}/{tract}{SubID:03d}_e.nii.gz',
                        '-minlength', '40',
                        '-maxlength', '250',
                        '-select', '2000',
                        '-nthreads', f'{nthreads}',
                        '-force',
                        '-quiet',
                        ])
    subprocess.run(f'tckconvert {tractpath}/{tract}{SubID:03d}.tck {vtkpath}/{tract}{SubID:03d}.vtk -force',shell=True)
def main(registor: FiberRegistor):
    # t_start = timeit.timeit
    registor.register()
    torch.cuda.empty_cache()
    # logger.info(f'Ready for Saving.')
    # registor.save_results()
    # logger.info(f'Saving done.')
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
    import argparse
    arg = argparse.ArgumentParser()
    arg.add_argument('--sub_num',type=int,default=60)
    arg.add_argument('--device',type=str,default='cuda:0')
    arg.add_argument('--batch_size',type=int,default=2)
    arg.add_argument('--suffix',type=str,default='alltracts_HCP_trained')
    arg.add_argument('--model_path',type=str,default='/data04/junyi/models/whole_model.pt')
    arg.add_argument('--data_path',type=str,default='/data01/junyi/datasets/HCP_100')
    arg.add_argument('--tracts_num',type=int,default=len(backbone_tracts))
    args = arg.parse_args()
    registor = FiberRegistor(backbone_tracts,
                             suffix=args.suffix,
                             model_path=args.model_path,
                             data_path=args.data_path,
                             sub_num=args.sub_num,
                             device=args.device,
                             batch_size=args.batch_size,
                             tracts_num=args.tracts_num)
    print(f'suffix: {args.suffix}')
    main(registor)