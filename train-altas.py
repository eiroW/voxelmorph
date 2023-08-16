import numpy as np
import torch
from torch import nn
import os
os.environ['VXM_BACKEND'] = 'pytorch'
from voxelmorph import networks
import voxelmorph as vxm
import nibabel as nib
import pathlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record

def ddp_setup():
   init_process_group(backend="nccl",)
   torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class MRIDataset(Dataset):
    def __init__(self, img_data, transform=None, ):
        self.img_data = img_data.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        image = self.img_data[idx, np.newaxis, ...]
        if self.transform:
            image = self.transform(image)
        return image

class Trainer():
    def __init__(self,
                 model : nn.Module,
                 train_data : DataLoader,
                 optimizer : torch.optim.Optimizer,
                 save_every : int,
                 snapshot_path : str
                 ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.loss_history = []
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)
        self.model = DDP(self.model,device_ids=[self.gpu_id])
    @torch.compile
    def _run_batch(self,source,target):
        loss_funcs, loss_weights= self._set_loss_function()
        self.optimizer.zero_grad()
        y_pred = self.model(source)
        loss = torch.stack([loss_funcs[i](target[i],y_pred[i])*loss_weights[i] for i in range(len(loss_weights))])
        loss.sum().backward()
        self.optimizer.step()
        return loss
    
    def _set_loss_function(self,):
        image_loss_func = nn.MSELoss()
        neg_loss_func = lambda _,y_pred: image_loss_func(y_pred, torch.stack([self.model.module.atlas for _ in range(y_pred.shape[0])]))
        mean_flow_loss = lambda _,y_pred: torch.square(y_pred).mean()
        loss_funcs = [image_loss_func, neg_loss_func, mean_flow_loss, vxm.losses.Grad('l2', loss_mult=2).loss]
        loss_weights = [0.5, 0.5, 0.1, 0.01]
        return loss_funcs, loss_weights
    
    def _run_epoch(self, epoch, steps_per_epoch=100):
        batch_size = len(next(iter(self.train_data)))
        print(f'GPU:{self.gpu_id}|Epoch:{epoch}|Batch size:{batch_size}\n')
        loss = 0
        for batch, source in enumerate(self.train_data) :
            source = source.to(self.gpu_id)
            target = [source,None,None,None]
            loss += self._run_batch(source, target)
            if (batch+1)%steps_per_epoch == 0:
                break
        return [loss[i].item()/(batch+1) for i in range(4)]

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at snapshot.pt\n")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path,map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}\n")

    def train(self,max_epochs):
        for epoch in range(self.epochs_run, max_epochs):
            self.loss_history.append(self._run_epoch(epoch,))
            if self.gpu_id==0 and epoch % self.save_every ==0:
                self._save_snapshot(epoch)

def prepare_dataloader(dataset: Dataset):
    return DataLoader(dataset,
                      batch_size=4,
                      pin_memory=True,
                      shuffle=False,
                      sampler=DistributedSampler(dataset),
                      )



def load_models_obj():

    path = pathlib.Path('../OASIS')
    # subj_lst_m = [str(f/'aligned_norm.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
    subj_lst_m = [str(f/'aligned_norm.nii.gz') for f in path.iterdir() if str(f).endswith('MR1')]
    # prepare data
    vols = [nib.load(f).get_fdata() for f in subj_lst_m]
    x_vols = np.stack(vols, 0).squeeze()
    vol_shape = x_vols[1:]
    dataset = MRIDataset(x_vols,)

    enc_nf = [16, 32, 32, 32]
    dec_nf = [32, 32, 32, 32, 32, 16, 16]

    atlas = torch.mean((torch.from_numpy(dataset.img_data[:10])),dim=0).squeeze().unsqueeze(0)
    model = torch.compile(networks.TemplateCreation(vol_shape,atlas,nb_unet_features=[enc_nf, dec_nf]))

    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4,eps=1e-07)
    return dataset, model, optimizer

@record
def main(total_epochs: int, save_every: int):
    ddp_setup()
    dataset, model, optimizer = load_models_obj()
    train_data = prepare_dataloader(dataset)
    trainer = Trainer(model, train_data, optimizer, save_every, 'snapshot.pt')
    trainer.train(total_epochs)
    np.save('loss_his', np.array(trainer.loss_history))
    destroy_process_group()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.total_epochs, args.save_every)