import numpy as np
import torch
from torch import nn
import functools
import os
from typing import Union
from dataset import FiberDatasetDir, center_crop
os.environ['VXM_BACKEND'] = 'pytorch'
os.environ['TORCHELASTIC_ERROR_FILE'] = 'error_history.json'
os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = gpus_list[:-1]

import logging.config
from voxelmorph.torch import networks
import voxelmorph as vxm
import nibabel as nib
import pathlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record
from torch.profiler import profile, record_function, ProfilerActivity






# backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
#                    'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',]

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    save_path = pathlib.Path(__file__).parent / 'logs'  
    save_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(save_path/ f"{name}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def ddp_setup():
   init_process_group(backend="nccl",)
   torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def calculate_time(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kargs)
        end.record()
        torch.cuda.synchronize()
        logger.info(f'The epoch lasted {start.elapsed_time(end) / 1000} s\n') if int(os.environ["LOCAL_RANK"])==0 else None
        return result

    return wrapper

def profile_record(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kargs):
        pass

class MRIDataset(Dataset):
    def __init__(self, img_data, transform=None, ):
        self.img_data = img_data[:, np.newaxis, ...].astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        if self.transform:
            image = self.transform(image)
        return image
    

class Trainer():
    def __init__(self,
                 model : nn.Module,
                 train_data : DataLoader,
                 optimizer : torch.optim.Optimizer,
                 save_every : int,
                 snapshot_path : pathlib.Path,
                 tract:str = None,
                 pretrained = False,
                 ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.loss_history = []
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.scaler = torch.amp.GradScaler()
        self.pretrained = pretrained
        self.tract = tract
        if os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)
        self.model = DDP(self.model,device_ids=[self.gpu_id])

    # @torch.compile
    def _run_batch(self,source,target):
        
        loss_funcs, loss_weights= self._set_loss_function()
        self.optimizer.zero_grad()
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            if self.pretrained:
                y_pred = self.model(source[self.tract])
                loss = torch.stack([loss_funcs[i](target[i],y_pred[i])*loss_weights[i] for i in range(len(loss_weights))])
            else:
                y_pred = self.model(source)
                loss = torch.stack(
                    [loss_funcs[i](target[i],y_pred[i])*loss_weights[i] for i in range(len(loss_weights))]
                    # +[0.01 * torch.abs(self.model.module.FA_model.atlas).mean(),]
                                   )
        self.scaler.scale(loss.sum()).backward()
        self.scaler.step(optimizer=self.optimizer)
        self.scaler.update()
        # torch.cuda.empty_cache()
        return loss
    
    def _set_loss_function(self,):
        if self.pretrained:
            image_loss_func = nn.MSELoss()
            # orient_loss_func = nn.CosineSimilarity()
            # pos_loss_func = lambda y_true, y_pred: 1-torch.mean(orient_loss_func(y_true,y_pred))
            # neg_loss_func = lambda _,y_pred: 1-torch.mean(image_loss_func(y_pred, torch.stack([self.model.module.atlas for _ in range(y_pred.shape[0])])))
            pos_loss_func = lambda _,y_pred: image_loss_func(y_pred, torch.stack([self.model.module.atlas for _ in range(y_pred.shape[0])]))
            neg_loss_func = lambda true_dict,y_pred: image_loss_func(y_pred, true_dict[self.tract])
            mean_flow_loss = lambda _,y_pred: torch.square(y_pred).mean()
            loss_funcs = [pos_loss_func,neg_loss_func,mean_flow_loss, vxm.losses.Grad('l2', loss_mult=2).loss,]
            loss_weights = [0.5, 0.5, 0.1, 0.01]
        else:
            image_loss_func = nn.SmoothL1Loss()
            # orient_loss_func = nn.CosineSimilarity()
            # pos_loss_func = lambda y_true, y_pred: 1-torch.mean(orient_loss_func(y_true,y_pred))
            # neg_loss_func = lambda _,y_pred: 1-torch.mean(image_loss_func(y_pred, torch.stack([self.model.module.atlas for _ in range(y_pred.shape[0])])))
            pos_loss_func = lambda _,y_pred: image_loss_func(y_pred, torch.stack([self.model.module.FA_model.atlas for _ in range(y_pred.shape[0])]))
            neg_loss_func_TOM = lambda y_true_dict,y_pred_dict: 0.5*torch.mean(torch.stack([image_loss_func(y_pred,y_true_dict[key])
                                                                                        for key,y_pred in y_pred_dict.items() if not key ==" FA"])
                                                                                        )if len(y_pred_dict.keys()) > 1 else 0
            neg_loss_func_FA = lambda y_true_dict,y_pred_dict: 0.5*torch.mean(torch.stack([image_loss_func(y_pred,y_true_dict[key]) 
                                                                                    for key,y_pred in y_pred_dict.items() if key =="FA"]))
            neg_loss_func = lambda y_true_dict,y_pred_dict: neg_loss_func_TOM(y_true_dict,y_pred_dict)+neg_loss_func_FA(y_true_dict,y_pred_dict)
            
            mean_flow_loss = lambda _,y_pred: torch.square(y_pred).mean()
            loss_funcs = [pos_loss_func,neg_loss_func, mean_flow_loss, vxm.losses.Grad('l2', loss_mult=2).loss,]
            loss_weights = [0.5, 0.5, 0.1, 0.01,]
        return loss_funcs, loss_weights

    
    @calculate_time
    def _run_epoch(self, epoch, steps_per_epoch=100):
        batch_size = len(next(iter(self.train_data)))
        logger.info(f'GPU:{self.gpu_id}|Epoch:{epoch}|Batch size:{batch_size}\n')
        loss = 0
        
        for batch, source in enumerate(self.train_data):
            source = {key:data.to(self.gpu_id)for key,data in source.items()}
            target = [None,source,None,None]
            # logger.info(f'Data prepared\n')if self.gpu_id==0 else None
            loss += self._run_batch(source, target)
            if (batch+1)%steps_per_epoch == 0:
                break
        
        return [l.item()/(batch+1) for l in loss]

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        snapshot['LOSS_HIS'] = self.loss_history
        torch.save(snapshot, self.snapshot_path)
        logger.info(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path,map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"],assign=True)
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.loss_history = snapshot['LOSS_HIS']
        logger.info(f"Resuming training from snapshot at Epoch {self.epochs_run}\n")
    
    def train(self,max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            loss = self._run_epoch(epoch,100)
            logger.info(f'Loss items{loss}\n') if self.gpu_id==0 else None
            self.loss_history.append(loss)
            if self.gpu_id==0 and (epoch+1) % self.save_every ==0:
                self._save_snapshot(epoch)

def prepare_dataloader(dataset: Dataset, batch_size=2,):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=False,
                      sampler=DistributedSampler(dataset),
                      num_workers=2
                      )



def load_models_obj(tracts,init_atlas,preTrain = True, FA_only = False):

    vol_shape = (128, 160, 128)
    # dataset = MRIDataset(x_vols,)
    
    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]
    if preTrain:
        atlas = torch.from_numpy(dataset.init_atlas[tracts])
        model = networks.TemplateCreation(vol_shape,atlas,nb_unet_features=[enc_nf, dec_nf],altas_feats=3,src_feats=3)
        # load trainable params 
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params, lr=1e-3,)
        return dataset, model, optimizer
    TOM_models = {}
    atlas = torch.empty(3,*vol_shape)
    
    if not FA_only:
        for tract in backbone_tracts:
            logger.info(f'Loading params from pretrained {tract} model...')
            snapshot_path = f'/data04/junyi/models/models_sep/{tract}_trained_{tract}.pt'
            snapshot = torch.load(snapshot_path,map_location='cpu')
            TOM_models[tract] = networks.TemplateCreation(vol_shape,atlas,nb_unet_features=[enc_nf, dec_nf],altas_feats=3,src_feats=3)
            TOM_models[tract].load_state_dict(snapshot["MODEL_STATE"],assign=True)

    
    # atlas = torch.mean((torch.from_numpy(dataset.img_data[:10])),dim=0).squeeze().unsqueeze(0)
    atlas = torch.empty(1,*vol_shape)if init_atlas == None else init_atlas
    model = networks.WholeModel(vol_shape,atlas,TOM_models,nb_unet_features=[enc_nf, dec_nf],altas_feats=1,src_feats=1,use_TOM=False,FA_only=FA_only)
    if not FA_only:
        for param in model.TOM_models.parameters():
            param.requires_grad = False
        for param in model.TOM_models.values():
            param.atlas.requires_grad = False
    # load trainable params 
    params = [p for p in model.parameters() if p.requires_grad]
    dataset = FiberDatasetDir('/data04/junyi/Datasets/datasets/HCP_100', backbone_tracts, center_crop,FA_only)
    optimizer=torch.optim.Adam(params, lr=1e-3,)
    return dataset, model, optimizer

def main(total_epochs: int, save_every: int, batch_size: int, save_path: str, tracts: str, pretrained = False,init_atlas_file = None):
    ddp_setup()
    init_atlas = torch.load(init_atlas_file) if not init_atlas_file == None else None
    dataset, model, optimizer = load_models_obj(tracts,init_atlas,pretrained,FA_only=False,)
    logger.info('Loading model done.')
    train_data = prepare_dataloader(dataset,batch_size,)
    save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model, train_data, optimizer, save_every, save_path,tracts,pretrained)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == '__main__':
    import argparse
    backbone_tracts = ['AF', 'ATR', 'CA', 'CC',
                   'CG', 'CST', 'FPT', 'FX', 'ICP', 'IFO', 'ILF', 'MCP', 'MLF', 'OR', 'POPT', 'SCP',
                   'SLF_I', 'SLF_II', 'SLF_III', 'STR',
                   'ST_FO', 'ST_OCC', 'ST_PAR', 'ST_POSTC', 'ST_PREC', 'ST_PREF', 'ST_PREM',
                   'T_OCC', 'T_PAR', 'T_POSTC', 'T_PREC', 'T_PREF', 'T_PREM', 'UF']
    # os.environ['TORCHELASTIC_ERROR_FILE'] = '~/Documents/scripts/registration/voxelmorph/error_history.json'
    parser = argparse.ArgumentParser(description='simple distributed training job')
    # torch._dynamo.config.verbose=True
    # torch._dynamo.config.suppress_errors = True
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('tracts', type=str, help='Tracts name prepared to be trained')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--save_path_prefix', default='snapshot', type=str, help='Output file of trained model')
    parser.add_argument('--init_atlas', type=str, help=' File of init atlas')
    parser.add_argument('--tracts_num', default=len(backbone_tracts), type=int, help='Number of tracts to train')
    args = parser.parse_args()
    logger = get_logger(args.save_path_prefix.split('/')[-1])
    backbone_tracts = backbone_tracts[:args.tracts_num]

    main(args.total_epochs, args.save_every, args.batch_size, args.save_path_prefix+'.pt', args.tracts,pretrained=False,init_atlas_file=args.init_atlas)
