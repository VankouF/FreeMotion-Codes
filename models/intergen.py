from typing import Any, Mapping, Union, List

import torch
import clip

from torch import nn
from models import *
from collections import OrderedDict

class InterGenSpatialControlNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = cfg.LATENT_DIM
        self.decoder = InterDiffusionSpatialControlNet(cfg, sampling_strategy=cfg.STRATEGY)
        
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
        self.token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

        set_requires_grad(self.clip_transformer, False)
        set_requires_grad(self.token_embedding, False)
        set_requires_grad(self.ln_final, False)

        clipTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True)
        self.clipTransEncoder = nn.TransformerEncoder(
            clipTransEncoderLayer,
            num_layers=2)
        self.clip_ln = nn.LayerNorm(768)
        
        self.positional_embedding.requires_grad = False
        
        set_requires_grad(self.clipTransEncoder, False)
        set_requires_grad(self.clip_ln, False)
        

    def compute_loss(self, batch):
        
        batch = self.text_process(batch)
        losses = self.decoder.compute_loss(batch)
        return losses["total"], losses

    def decode_motion(self, batch):
        batch.update(self.decoder(batch))  # batch['output'].shape = [1, 210, 524]
        return batch

    def forward(self, batch):
        return self.compute_loss(batch)

    def forward_test(self, batch):  # batch: 'motion_lens', 'prompt', 'text'=['prompt']
        
        batch = self.text_process(batch)
        batch.update(self.decode_motion(batch))
        return batch

    def text_process(self, batch):
        device = next(self.clip_transformer.parameters()).device
        
        if "text" in batch and batch["text"] is not None and not isinstance(batch["text"], torch.Tensor):
            
            raw_text = batch["text"]

            with torch.no_grad():
                
                text = clip.tokenize(raw_text, truncate=True).to(device)  # [batch_szie, n_ctx]=[1,77]
                x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]=[1,77,768]
                pe_tokens = x + self.positional_embedding.type(self.dtype)
                x = pe_tokens.permute(1, 0, 2)  # NLD -> LND  [n_ctx, batch_size, d_model]=[77,1,768]
                x = self.clip_transformer(x)
                x = x.permute(1, 0, 2)
                clip_out = self.ln_final(x).type(self.dtype) # [batch_size, n_ctx, d_model]=[1,77,768]
                
            out = self.clipTransEncoder(clip_out)  # [batch_size, n_ctx, d_model]=[1,77,768]
            out = self.clip_ln(out)

            cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            batch["cond"] = cond
            
        if "text_multi_person" in batch and batch["text_multi_person"] is not None and not isinstance(batch["text_multi_person"], torch.Tensor):
            raw_text = batch["text_multi_person"]

            with torch.no_grad():
                
                text = clip.tokenize(raw_text, truncate=True).to(device)  # [batch_szie, n_ctx]=[1,77]
                x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]=[1,77,768]
                pe_tokens = x + self.positional_embedding.type(self.dtype)
                x = pe_tokens.permute(1, 0, 2)  # NLD -> LND  [n_ctx, batch_size, d_model]=[77,1,768]
                x = self.clip_transformer(x)
                x = x.permute(1, 0, 2)
                clip_out = self.ln_final(x).type(self.dtype) # [batch_size, n_ctx, d_model]=[1,77,768]
                
            out = self.clipTransEncoder(clip_out)  # [batch_size, n_ctx, d_model]=[1,77,768]
            out = self.clip_ln(out)

            cond = out[torch.arange(x.shape[0]), text.argmax(dim=-1)]
            
            if "cond" in batch:
                batch["cond"] = batch["cond"] + cond
            else:
                batch["cond"] = cond
            
        return batch

    def load_state_dict(self, state_dict: Union[List[Mapping[str, Any]],Mapping[str, Any]], strict: bool = True):
        
        # for test
        if isinstance(state_dict, list):
            state_dict_motion_condition = state_dict[0]
            state_dict_spatial_condition = state_dict[1]
            
            state_dict = state_dict_motion_condition
            
            for k, v  in state_dict_spatial_condition.items():
                if "decoder.net.control_branch" in k:
                    k_new = k.replace('decoder.net.control_branch', 'decoder.net.spatial_control_branch')
                    state_dict[k_new] = v
                elif "decoder.net.input_hint_block" in k:
                    state_dict[k] = v
                elif "decoder.net.zero_linear" in k:
                    k_new = k.replace('decoder.net.zero_linear', 'decoder.net.spatial_zero_linear')
                    state_dict[k_new] = v
                    
            return super().load_state_dict(state_dict, strict)
        
        
        for k in state_dict.keys(): 
            if "decoder.net.net" in k:
                return super().load_state_dict(state_dict, strict=False)
        
        new_state_dict = OrderedDict()
        for name, value in state_dict.items():
            if 'decoder.net' in name:
                name_new = name.replace('decoder.net', 'decoder.net.control_branch')
                new_state_dict[name_new] = value
                name_new = name.replace('decoder.net', 'decoder.net.net')
                new_state_dict[name_new] = value
            else:
                new_state_dict[name] = value
        return super().load_state_dict(new_state_dict, strict=False)
    