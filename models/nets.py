from typing import Any, Mapping
import torch

from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *

from models.gaussian_diffusion import (
    # MotionDiffusion,
    MotionSpatialControlNetDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
import random

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch


class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 archi='single',
                 return_intermediate=False,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.archi = archi
        
        self.text_emb_dim = 768
        self.spatial_emb_dim = 4
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)
        
        self.return_intermediate = return_intermediate
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(TransformerMotionGuidanceBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
            # self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))


    def mask_motion_guidance(self, motion_mask, cond_mask_prob = 0.1, force_mask=False):
        # This function is aims to mask the motion guidance with a probability of cond_mask_prob.
        # motion_mask of input has been mask the invalid time.
        # mask.shape=[B,P,T]
        bs, guidance_num = motion_mask.shape[0], motion_mask.shape[1]-1
        if guidance_num == 0:
            return motion_mask.clone()
        elif force_mask:
            mask_new = motion_mask.clone()
            mask_new[:,1:,:] = 0
            return mask_new
        elif cond_mask_prob > 0.:
            
            mask_new = torch.bernoulli(torch.ones(bs,guidance_num, device=motion_mask.device) * cond_mask_prob).view(bs,guidance_num,1).contiguous() # 1-> use null_cond, 0-> use real cond
            mask_new = torch.cat([torch.zeros(bs,1,1, device=motion_mask.device), mask_new], dim=1)
            
            return motion_mask * (1. - mask_new)
        else:
            return motion_mask
 
    def forward(self, x, timesteps, mask=None, cond=None, motion_guidance=None, **kwargs): # spatial_condition=None, **kwargs):
        """
        x: B, T, D
        """
        
        B, T = x.shape[0], x.shape[1]
 
        if motion_guidance is None:  # single motion generation
            x_cat = x.unsqueeze(1)  # B,1,T,D
        elif self.return_intermediate:  # control branch for two motion
            motion_guidance = motion_guidance.permute(0,2,1,3)  # B,P,T,D
            x_cat = torch.cat([x.unsqueeze(1), motion_guidance], dim=1)  # B,1,T,D, B,P,T,D -> B,P+1,T,D
        else:  # main branch for two motion
            x_cat = x.unsqueeze(1)  # B,1,T,D   
            
        if mask is not None:
            mask = mask.permute(0,2,1).contiguous()[:,:x_cat.shape[1],...]  # B,T,P -> B,P,T
        else:
            mask = torch.ones(B, x_cat.shape[1], T).to(x_cat.device)
        
        if x_cat.shape[1] > 1:
            mask = self.mask_motion_guidance(mask, 0.1)
        
        emb = self.embed_timestep(timesteps) + self.text_embed(cond)  # [4],[4,768] -> [4,1024]

        x_cat_emb = self.motion_embed(x_cat)  # B,P+1,T,D -> B,P+1,T,1024

        h_cat_prev = self.sequence_pos_encoder(x_cat_emb.view(-1, T, self.latent_dim)).view(B, -1, T, self.latent_dim)

        key_padding_mask = ~(mask > 0.5)

        h_cat_prev = h_cat_prev.view(B,-1,self.latent_dim)  # [B,P*T,D]
        key_padding_mask = key_padding_mask.view(B,-1)  # [B,P*T]
        
        if self.return_intermediate:  # For control branch | spatial control branch
            intermediate_feats = []

        for idx ,block in enumerate(self.blocks):
            
            h_cat_prev = block(h_cat_prev, T, emb, key_padding_mask)

            if motion_guidance is None:  # single motion generation
                pass
            elif self.return_intermediate:  # control branch
                intermediate_feats.append(h_cat_prev.view(B,-1,T,self.latent_dim)[:,:1,...])
            else:  # main branch for two motion
                h_cat_prev = h_cat_prev.view(B,-1,T,self.latent_dim)
                h_cat_prev = h_cat_prev + motion_guidance[idx]
                h_cat_prev = h_cat_prev.view(B,-1,self.latent_dim)
                
        if self.return_intermediate:
            return intermediate_feats
            
        h_cat_prev = h_cat_prev.view(B,-1,T,self.latent_dim)
        output = self.out(h_cat_prev)
        
        return output[:,0,...] # only return the first person for matching the dimension of diffusion process.


class InterDenoiserSpatialControlNet(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 archi='single',
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.archi = archi
        
        if self.input_feats == 262 or self.input_feats == 263:
            self.n_joints = 22
        else:
            self.n_joints = 21
        
        self.net = InterDenoiser(self.input_feats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                       cfg_weight=self.cfg_weight, archi=self.archi, return_intermediate=False)
        
        if self.archi != 'single':
            self.control_branch = InterDenoiser(self.input_feats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                        num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, 
                                        cfg_weight=self.cfg_weight, archi=self.archi, return_intermediate=True)
            
            self.zero_linear = zero_module(nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.num_layers)]))
            
            set_requires_grad(self.net, False)
        
        
    def forward(self, x, timesteps, mask=None, cond=None, motion_guidance=None, spatial_condition=None, **kwargs):
        
        """
        x: B, T, D
        spatial_condition: B, T, D
        """
        
        if motion_guidance is not None:
            intermediate_feats = self.control_branch(x, timesteps, mask=mask, cond=cond, motion_guidance=motion_guidance, spatial_condition=None,**kwargs)
            intermediate_feats = [self.zero_linear[i](intermediate_feats[i]) for i in range(len(intermediate_feats))]
        else:
            intermediate_feats = None
        
        output = self.net(x, timesteps, mask=mask, cond=cond, motion_guidance=intermediate_feats,**kwargs) #spatial_condition=None,**kwargs)
        
        return output


class InterDiffusionSpatialControlNet(nn.Module):
    def __init__(self, cfg, sampling_strategy="ddim50"):
        super().__init__()
        self.cfg = cfg
        self.archi = cfg.ARCHI
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampler = cfg.SAMPLER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiserSpatialControlNet(self.nfeats, self.latent_dim, ff_size=self.ff_size, num_layers=self.num_layers,
                                       num_heads=self.num_heads, dropout=self.dropout, activation=self.activation, cfg_weight=self.cfg_weight, archi=self.archi)

        self.diffusion_steps = self.diffusion_steps
        self.betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)

        timestep_respacing=[self.diffusion_steps]
        self.diffusion = MotionSpatialControlNetDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            archi= self.archi,
        )
        self.sampler = create_named_schedule_sampler(self.sampler, self.diffusion)

    def mask_cond(self, cond, cond_mask_prob = 0.1, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * cond_mask_prob).view([bs]+[1]*len(cond.shape[1:]))  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask), (1. - mask)
        else:
            return cond, None

    def generate_src_mask(self, T, length, person_num, B=0):
        if B==0:
            B = length.shape[0]
        else:
            if len(length.shape) == 1:
                length = torch.cat([length]*B, dim=0)
        src_mask = torch.ones(B, T, person_num)
        for p in range(person_num):
            for i in range(B):
                for j in range(length[i], T):
                    src_mask[i, j, p] = 0
        return src_mask

    def compute_loss(self, batch):  # 似乎这个函数是在训练阶段使用的
        cond = batch["cond"]
        x_and_xCondition = batch["motions"]  # [4,300,524]
        B,T = batch["motions"].shape[:2]

        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, 0.1)

        seq_mask = self.generate_src_mask(batch["motions"].shape[1], batch["motion_lens"], batch["person_num"]).to(x_and_xCondition.device)

        t, _ = self.sampler.sample(B, x_and_xCondition.device)
        
        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_and_xCondition,
            t=t,
            mask=seq_mask,
            t_bar=self.cfg.T_BAR,
            cond_mask=cond_mask,
            model_kwargs={"mask":seq_mask,
                          "cond":cond,
                          "person_num":batch["person_num"],
                        #   "spatial_condition":batch["spatial_condition"],
                          },
        )
        return output

    def forward(self, batch):
        
        cond = batch["cond"]
        
        motion_guidance = batch["motion_guidance"]
        # spatial_condition = batch["spatial_condition"]
        
        if motion_guidance is not None:
            B, T = motion_guidance.shape[:2]
        else:  # T equals valid motion lens, then all items of the mask is valid.
            B = cond.shape[0]
            T = batch["motion_lens"].item()
            
        # add valid time length mask
        seq_mask = self.generate_src_mask(T, batch["motion_lens"], batch["person_num"], B=cond.shape[0]).to(batch["motion_lens"].device)
        
        timestep_respacing= self.sampling_strategy
        self.diffusion_test = MotionSpatialControlNetDiffusion(  # MotionSpatialControlNetDiffusion
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps = False,
            archi= self.archi,
        )
        
        self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight) # cfg_weight=3.5
        
        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B,T,self.nfeats),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                "mask":seq_mask, # None,
                "cond":cond,
                "motion_guidance":motion_guidance,
                # "spatial_condition": spatial_condition
            },
            x_start=None)

        return {"output":output}  # output: [batch_size, n_ctx, 2*d_model]=[1,210,524]




