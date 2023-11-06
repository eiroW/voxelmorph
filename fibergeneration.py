import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
import pathlib
import os
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import networks, layers
import nibabel as nib
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as nnf
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s','%m-%d %H:%M:%S')
console = logging.StreamHandler()
flie_log = logging.FileHandler('fiber_registration.log',mode='w')
logger.addHandler(console)
logger.addHandler(flie_log)

# gpus_list = ''
# for i in range(torch.cuda.device_count()):
#     used,all=torch.cuda.mem_get_info(device=i)
#     if used/all >=0.80:
#         gpus_list+=f'{i},'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list[:-1]

backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
# backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
#                    'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',]

class FiberDatasetDir(Dataset):
    def __init__(self, img_data_dir: pathlib.Path, transform=None, ):

        self.img_data_dir = sorted(list(pathlib.Path(img_data_dir).iterdir()))[:10]
        self.transform = transform
        tmp = nib.load(list(self.img_data_dir[0].glob('**/FA_MNI*'))[0])
        self.config = (tmp.affine,tmp.header)
        self.shape = tmp.get_fdata().shape

    def __len__(self):
        return len(self.img_data_dir)

    def __getitem__(self, idx):
        images_dir_path = self.img_data_dir[idx]
        struct_dict = {}
        indice_down = np.array(self.shape)//2-np.array([128//2, 160//2, 128//2])
        indice_up =(np.array(self.shape)//2)+np.array([128//2, 160//2, 128//2])
        FA_path = sorted(list(images_dir_path.glob('**/FA_MNI*')))[0]
        struct_dict['FA'] = self.transform(nib.load(FA_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
        
        

        for tract in backbone_tracts:
            TOM_path = sorted(list((images_dir_path/'T1w/Diffusion/tractseg_output/TOM_new/').glob(f'{tract}.nii.gz')))[0]
            struct_dict[tract] = self.transform(nib.load(TOM_path).get_fdata()[::-1],indice_down,indice_up).transpose(-1,0,1,2)
            
            seg_left_paths = sorted(list((images_dir_path/'T1w/Diffusion/tractseg_output/bundle_segmentations').glob(f'{tract}_left.nii.gz')))
            seg_left_path = seg_left_paths[0] if len(seg_left_paths) else None
            seg_right_paths = sorted(list((images_dir_path/'T1w/Diffusion/tractseg_output/bundle_segmentations').glob(f'{tract}_right.nii.gz')))
            seg_right_path = seg_right_paths[0] if len(seg_left_paths) else None
            if seg_left_path ==None:
                seg_path = sorted(list((images_dir_path/'T1w/Diffusion/tractseg_output/bundle_segmentations').glob(f'{tract}.nii.gz')))[0]
                struct_dict[tract+'seg'] = self.transform(nib.load(seg_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
            else:
                struct_dict[tract+'seg'] = self.transform(nib.load(seg_left_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]\
                +self.transform(nib.load(seg_right_path).get_fdata()[::-1],indice_down,indice_up)[np.newaxis, ...]
        
        return struct_dict
    
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
    
def crop(x,indice_down, indice_up):
    x = x.astype(np.float32)
    return x[indice_down[0]:indice_up[0],indice_down[1]:indice_up[1],indice_down[2]:indice_up[2]]
    
def main():
    device = 'cuda:0'
    i = 9
    snapshot_whole_path = f'/data01/junyi/models/models_sep/snapshot20_alltracts_whole.pt'
    fd = FiberDatasetDir('/data01/junyi/datasets/HCP_100',crop)
    root_dir = '/data01/junyi/results/tracts_related'
    # if not pathlib.Path(f'{root_dir}/TOM{i}').exists():
    #     os.mkdir(pathlib.Path(f'{root_dir}/TOM{i}'))
    FApath = pathlib.Path(f'{root_dir}/FA{i}/whole')
    if not FApath.exists():
        os.mkdir(FApath.parent)if not FApath.parent.exists() else None
        os.mkdir(FApath)
    flowpath = pathlib.Path(f'{root_dir}/flow{i}/whole')
    if not flowpath.exists():
        os.mkdir(flowpath.parent)if not flowpath.parent.exists() else None
        os.mkdir(flowpath)
    for tract in backbone_tracts:
        TOMpath = pathlib.Path(f'{root_dir}/TOM{i}/{tract}')
        os.mkdir(TOMpath.parent) if not TOMpath.parent.exists() else None
        os.mkdir(TOMpath) if not TOMpath.exists() else None
        segpath = pathlib.Path(f'{root_dir}/seg{i}/{tract}')
        if not segpath.exists():
            os.mkdir(segpath.parent)if not segpath.parent.exists() else None
            os.mkdir(segpath)

        tractpath = pathlib.Path(f'{root_dir}/tracts{i}/{tract}')
        if not tractpath.exists():
            os.mkdir(tractpath.parent)if not tractpath.parent.exists() else None
            os.mkdir(tractpath)
        vtkpath = pathlib.Path(f'{root_dir}/vtk{i}/{tract}')
        if not vtkpath.exists():
            os.mkdir(vtkpath.parent)if not vtkpath.parent.exists() else None
            os.mkdir(vtkpath)
    affine, header = fd.config
    
    vol_shape = (128, 160, 128)
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    batch_size = 4

    atlas = torch.empty(3, *vol_shape)
    snapshot = torch.load(snapshot_whole_path,map_location='cpu')

    TOM_models = {}
    for tract in backbone_tracts:
        # snapshot_path = f'snapshot1_{tract}.pt'
        # snapshot = torch.load(snapshot_path,map_location='cpu')
        TOM_models[tract] = networks.TemplateCreation(vol_shape,atlas,nb_unet_features=[enc_nf, dec_nf],altas_feats=3,src_feats=3)

    
    atlas = torch.zeros(1,*vol_shape)
    model = networks.WholeModel(vol_shape,atlas,TOM_models,nb_unet_features=[enc_nf, dec_nf],altas_feats=1,src_feats=1,FA_only=False,use_TOM=False)

    model.load_state_dict(snapshot["MODEL_STATE"])

    bilinST = SpatialTransformer(vol_shape,mode='bilinear')
    nnST = SpatialTransformer(vol_shape,mode='nearest')
    reorient = layers.TOMReorientation(vol_shape)
    segs_dict = {tract:[] for tract in backbone_tracts}
    TOMs_dict = {tract:[] for tract in backbone_tracts}
    FAs_list = []
    flows_list = []
    dataloader = DataLoader(fd,batch_size,num_workers=2)
    with torch.no_grad():
        model.eval().to(device)
        
        for data_dict in dataloader:
            data = {key:im.to(device) for key,im in data_dict.items() if "seg" not in key}
            register_FAs,flows = model(data,registration=True,)
            register_segs ={}
            register_TOMs = {}

            logger.info(f'Computing Flow done')
            flows = flows.cpu()
            this_batch_size = flows.shape[0]
            rotMat = reorient(flows)

            for tract in backbone_tracts:
                logger.info(f'Registing on {tract}...')
            
                register_segs[tract] = nnST(data_dict[tract+'seg'],flows)
                TOM_tmp = bilinST(data_dict[tract],flows)
                
                register_TOMs[tract] = reorient.batched_rotate(rotMat,TOM_tmp)
                segs_dict[tract] += [register_segs[tract][i].squeeze().numpy()[::-1] for i in range(this_batch_size)]
                TOMs_dict[tract] += [register_TOMs[tract][i].float().cpu().permute(1,2,3,0).numpy()[::-1] for i in range(this_batch_size)]
                logger.info(f'Registing on {tract} done')
            FAs_list += [register_FAs[i].cpu().squeeze().numpy()[::-1] for i in range(this_batch_size)]
            flows_list += [flows[i].permute(1,2,3,0).numpy()[::-1] for i in range(this_batch_size)]
            torch.cuda.empty_cache()
        logger.info(f'Ready for Saving.')
        for i,FA in enumerate(FAs_list):

            nib.save(nib.Nifti1Image(FA, affine, header),FApath/f'SUB{i:03d}.nii.gz')
            nib.save(nib.Nifti1Image(flows_list[i], affine, header),flowpath/f'SUB{i:03d}.nii.gz')
            for tract in backbone_tracts:
                logger.info(f'Saving {tract} of Sub{i:03d}...')
                nib.save(nib.Nifti1Image(TOMs_dict[tract][i], affine, header),TOMpath.parent/f'{tract}'/f'{tract}{i:03d}.nii.gz')
                nib.save(nib.Nifti1Image(segs_dict[tract][i], affine, header),segpath.parent/f'{tract}'/f'{tract}{i:03d}.nii.gz')
                os.system(f'tckgen -algorithm FACT {TOMpath.parent}/{tract}/{tract}{i:03d}.nii.gz {tractpath.parent}/{tract}/{tract}{i:03d}.tck -minlength 40 -maxlength 250 -select 2000  -nthreads 24 -seed_image {segpath.parent}/{tract}/{tract}{i:03d}.nii.gz -mask {segpath.parent}/{tract}/{tract}{i:03d}.nii.gz -force')
                os.system(f'tckconvert {tractpath.parent}/{tract}/{tract}{i:03d}.tck {vtkpath.parent}/{tract}/{tract}{i:03d}.vtk -force')
        
if __name__ =='__main__':
    main()
    
    