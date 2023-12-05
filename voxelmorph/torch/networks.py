import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import List, Dict
from collections import OrderedDict
from .. import default_unet_features
from . import layers
from .modelio import LoadableModel, store_config_args
import logging


torch.set_float32_matmul_precision('high')



class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf
    # @torch.compile
    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 use_TOM = True,
                 **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.use_TOM = use_TOM
        if use_TOM:
            self.reorient = layers.TOMReorientation(inshape,)
        
    # @torch.compile
    def forward(self, source, target, registration=False, fusion_regist=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        
        x = torch.cat((source, target),dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)
        if fusion_regist :
            return pos_flow
        preint_flow = pos_flow
        
        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
        
        
        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        if self.use_TOM:
            rotMat = self.reorient(pos_flow) 
            target = self.reorient.batched_rotate(rotMat.mT, target)
            y_target = self.transformer(target, neg_flow) if self.bidir else None
            y_source = self.reorient.batched_rotate(rotMat, y_source)
        else:
            y_target = self.transformer(target, neg_flow) if self.bidir else None

        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, pos_flow, neg_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow
    
class TemplateCreation(LoadableModel):
    """
    VoxelMorph network to generate an unconditional template image.
    """

    @store_config_args
    def __init__(self, inshape, init_atlas, 
                 nb_unet_features=None, 
                 mean_cap=100,
                 altas_feats=1,
                 **kwargs):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. 
                See VxmDense documentation for more information.
            mean_cap: Cap for mean stream. Default is 100.
            atlas_feats: Number of atlas/template features. Default is 1.
            src_feats: Number of source image features. Default is 1.
            kwargs: Forwarded to the internal VxmDense model.
        """
        super().__init__()
        # configure inputs
        self.type = type
        # pre-warp (atlas) model: source input -> atlas
        self.atlas= nn.Parameter(init_atlas)
        # rand_atlas = nn.init.normal_(torch.empty(1, *inshape),mean=0, std=1e-7)
        # self.atlas= nn.Parameter(rand_atlas)

        self.mult = 1.0
        # warp model source input -> atlas,source input
        self.vxm_model = VxmDense(inshape, 
                                  nb_unet_features=nb_unet_features,
                                  bidir=True,
                                  trg_feats=altas_feats,
                                  **kwargs)

        # get mean stream of negative flow
        self.mean_stream = layers.MeanStream(inshape, cap=mean_cap)

        # TOM rotation 
        self.reorient = layers.TOMReorientation(inshape,)

        # initialize the keras model
    # @torch.compile
    def forward(self, source_input, registration=False, fusion_regist=False):
        batch_size = source_input.shape[0]

        atlas = torch.stack([self.atlas for _ in range(batch_size)])
        if fusion_regist:
            flow = self.vxm_model(source_input, atlas, registration, fusion_regist=fusion_regist)
            return flow
        
        if not registration:
            y_source, y_target, pos_flow, neg_flow = self.vxm_model(source_input, atlas ,registration)
            # TOM reoriention
            mean_flow = self.mean_stream(neg_flow, registration)
            return y_source, y_target, mean_flow, pos_flow
        else:
            y_source, pos_flow = self.vxm_model(source_input, atlas, registration)

            return y_source, pos_flow
class WholeModel(nn.Module):
    def __init__(self, inshape, init_altas, 
                 TOM_models:dict[str,TemplateCreation], 
                 *args,
                 int_downsize=2,
                 int_steps=7,
                 FA_only = False,
                 **kwargs):
        super(WholeModel,self).__init__()
        ndims = len(inshape)
        self.TOM_models = nn.ModuleDict(TOM_models)
        self.FA_model = TemplateCreation(inshape,init_altas,**kwargs)
        if not FA_only:
            self.flow_multi = ConvBlock(ndims,(len(TOM_models) + 1) * 3, 3,)
            nn.init.normal_(self.flow_multi.main.weight,mean=0,std=1e-3)
            nn.init.zeros_(self.flow_multi.main.bias)

        down_shape = [int(dim / int_downsize) for dim in inshape]
    
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None
        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None
        # configure transformer
        self.transformer = layers.SpatialTransformer(inshape)
        self.reorient = layers.TOMReorientation(inshape,)
        self.FA_only = FA_only

    # @torch.compile
    def forward(self, inputs_dict: dict[str,torch.Tensor],fusion_regist=True, registration=False,**kwargs):
        registed_dict = {}
        FA_only = self.FA_only
        if FA_only:
                FA_registed, registed_dict['FA'], mean_flow, pos_flow = self.FA_model(inputs_dict['FA'],fusion_regist=False)
                if registration:
                    return FA_registed, pos_flow
        else:
            flow_TOMs = []
            with torch.no_grad():
                for key, TOM_model in self.TOM_models.items():
                    flow_TOM = TOM_model(inputs_dict[key],fusion_regist = fusion_regist)
                    flow_TOMs.append(flow_TOM)
            self.flow_TOMs = flow_TOMs
            flow_FA = self.FA_model(inputs_dict['FA'],fusion_regist=fusion_regist)
            batch_size = flow_FA.shape[0]
            if self.training:
                drop_prob = 0
                rand_tensor_shape = (batch_size,len(inputs_dict)-1)+(1,)*(flow_FA.ndim-2)
                rand_tensor = 1-drop_prob+torch.rand(rand_tensor_shape,dtype=flow_FA.dtype,device=flow_FA.device)
                rand_tensor.floor_()
                rand_tensor = torch.stack([rand_tensor for _ in range(3)],dim=1).reshape((batch_size,3*(len(inputs_dict)-1),*rand_tensor_shape[2:]))
            else:
                rand_tensor = 1
            # logging.info(f'Drop path {rand_tensor}')
            flow_fusion = self.flow_multi(torch.cat([rand_tensor*torch.cat(flow_TOMs,dim=1),flow_FA],dim=1))
            # logging.info(f'Predict {flow_fusion}')
            if self.integrate:
                flow_fusion = self.integrate(flow_fusion)
                neg_flow_fusion = self.integrate(-flow_fusion)

                # resize to final resolution
                if self.fullsize:
                    pos_flow = self.fullsize(flow_fusion)
                    neg_flow = self.fullsize(neg_flow_fusion)
            # logging.info(f'Predict {neg_flow}')
            mean_flow = self.FA_model.mean_stream(neg_flow,registration)
            # logging.info(f'Predict {mean_flow}')
            # warp image with flow field
            FA_registed = self.transformer(inputs_dict['FA'], pos_flow)
            if registration:
                return FA_registed, pos_flow
            
            FA_atlas_registed = self.transformer(torch.stack([self.FA_model.atlas for _ in range(batch_size)]), neg_flow)

            registed_dict = {}
            registed_dict['FA'] = FA_atlas_registed
            rotMat = self.reorient(neg_flow)
            for keys, TOM_model in self.TOM_models.items():
                TOM_atlas_registed = self.reorient.batched_rotate(rotMat, torch.stack([TOM_model.atlas for _ in range(batch_size)]))
                TOM_atlas_registed = self.transformer(TOM_atlas_registed, neg_flow)
                registed_dict[keys] = TOM_atlas_registed

        
        return FA_registed, registed_dict, mean_flow, pos_flow
            
class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims,  in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)
    # @torch.compile
    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

