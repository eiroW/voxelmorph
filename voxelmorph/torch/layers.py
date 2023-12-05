import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.utils.parametrizations as parametrizations
import numpy as np


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


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x
class MeanStream(nn.Module):
    def __init__(self,input_shape, cap=100, **kwargs) -> None:
        super(MeanStream, self).__init__()
        self.cap = float(cap)
        self.register_buffer(name='mean',
                                         tensor=torch.zeros((len(input_shape),*input_shape)))
        self.register_buffer(name='count',
                                         tensor=torch.zeros(1))
        # v = np.prod(*input_shape)
        # self.register_buffer(name='cov', 
        #                                 tensor= torch.zeros(v,v))
    # @torch.compile
    def forward(self,x,registration=False):
        batch_size = x.shape[0]
        if registration :
            return torch.minimum(torch.tensor(1.), self.count/self.cap) * torch.stack([self.mean for _ in range(batch_size)])
        
        # update mean and count
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)
        self.mean = new_mean.detach()
        self.count = new_count.detach()
        # self.cov = new_cov
        return torch.minimum(torch.tensor(1.), new_count / self.cap) * torch.stack([new_mean for _ in range(batch_size)])

def _mean_update(pre_mean, pre_count, x, pre_cap=None):
    # compute this batch stats
    this_mean = torch.mean(x, 0)
    pre_cap = torch.tensor(pre_cap)
    # increase count and compute weights
    new_count = pre_count + x.shape[0]
    alpha = x.shape[0] / torch.minimum(new_count, pre_cap)

    # compute new mean. Note that once we reach self.cap (e.g. 1000),
    # the 'previous mean' matters less

    new_mean = pre_mean * (1 - alpha) + (this_mean) * alpha

    return (new_mean, new_count)

class LocalParamWithInput(nn.Module):
    def __init__(self, shape, initializer=None, mult=1.0, **kwargs) -> None:
        super(LocalParamWithInput).__init__()
        self.shape = shape # no batch
        self.initializer = initializer
        self.biasmult = mult
        self.kernel = nn.Parameter(self.initializer(torch.empty(shape)))

    def forward(self, x):
        batch_size = x.shape[0]
        params = self.kernel * self.biasmult
        return torch.stack([params for _ in batch_size])



class TOMReorientation(nn.Module):
    """
    TOM Reorientation layer
    """

    def __init__(self, size):
        super().__init__()
        #size: (128, 160, 128)
        # create sampling grid
        # vectors = [torch.arange(0, s) for s in size]
        # grids = torch.meshgrid(vectors,indexing='ij')
        # grid = torch.stack(grids)
        # grid = torch.unsqueeze(grid, 0)
        # grid = grid.type(torch.FloatTensor)
        # self.register_buffer('grid', grid, persistent=False)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('ident', torch.eye(3), persistent=False)
        # self.ident = grid
        self.size = size
        self.flatten = nn.Flatten(start_dim=2)
    @torch.compile
    def forward(self,flow):
        # new locations
        # rotMat,keep = 
        return self.compute_rotMat(flow.detach())
    
    @torch.no_grad()
    def compute_rotMat(self, trans_grid:torch.Tensor):
        # Compute Jacobian matrix and trans it into size(*inshape, 3, 3)
        JacobiMat = torch.stack(gradient(trans_grid),dim=1).permute(1,2,3,4,5,0)
        JacobiMat = self.flatten(JacobiMat).permute(-1,0,1).contiguous()

        # U,_,V= torch.svd(JacobiMat)
        # rotMat = torch.matmul(U.transpose(-1,-2),V)
        return fast_polar(JacobiMat+self.ident)
    
    @torch.no_grad()
    def compute_rotMat_skew_sym(self, trans_grid:torch.Tensor):
        # Compute Jacobian matrix and trans it into size(*inshape, 3, 3)
        JacobiMat = torch.stack(gradient(trans_grid),dim=1).permute(1,2,3,4,5,0)
        JacobiMat = self.flatten(JacobiMat).permute(-1,0,1).contiguous()

        # U,_,V= torch.svd(JacobiMat)
        # rotMat = torch.matmul(U.transpose(-1,-2),V)
        return skew_sym(JacobiMat)
    
    # 
    def batched_rotate(self,rotMat:torch.Tensor, src:torch.Tensor):
        batch_size = src.shape[0]
        src = self.flatten(src.unsqueeze(2).permute(1,2,3,4,5,0))
        src = src.permute(-1,0,1)
        src = torch.matmul(rotMat, src)

        return src.reshape(*(self.size),batch_size,3).permute(-2,-1,0,1,2)

def gradient(trans_grid):
    return torch.gradient(trans_grid, dim=(-3,-2,-1))

def orthogonalize(X:torch.Tensor):
    xtype = X.dtype
    # Half SVD have not be implemented
    X = X.float() if xtype == torch.float16 else X
    U,_,V = torch.svd(X)
    # h, tau = torch.geqrf(X)
    # Q = torch.linalg.householder_product(h, tau)
    Q = U @ V.mT
    return Q.half() if xtype == torch.float16 else Q

def fast_polar(A:torch.Tensor):
    temp = A
    device = A.device
    temp_norm = torch.ones((*(A.shape[:-2]),1,1),device=device)
    mask = torch.ones(A.shape[:-2],dtype=bool,device=device)
    old_norm = torch.ones((*(A.shape[:-2]),1,1),device=device)
    tempinv = torch.ones((*(A.shape[:-2]),3,3),device=device)
    tempinv_norminv = torch.ones(A.shape,device=device)
    # gamma = torch.ones(A.shape,device=device)
    for _ in range(100):
        old_norm[mask] = temp_norm[mask]
        temp_norm[mask] = torch.linalg.matrix_norm(temp[mask],dim=(-2,-1),keepdim=True)
        tempinv[mask] = torch.inverse(temp[mask])
        tempinv_norminv[mask] = torch.linalg.matrix_norm(tempinv[mask],dim=(-2,-1),keepdim=True)
        # gamma[mask] = torch.sqrt(temp_norm[mask] * 1/tempinv_norminv[mask])
        # temp[mask] = 0.5*(gamma[mask]*temp[mask] + 1/gamma[mask]*tempinv[mask].mT)
        temp[mask] = 0.5*(temp[mask] + tempinv[mask].mT)
        mask = (torch.abs(temp_norm-old_norm)>1e-5).squeeze()
        if (~mask).all():
            break
    return temp


    


