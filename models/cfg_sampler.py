import torch
import torch.nn as nn
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model, cfg_scale):
        super().__init__()
        self.model = model  # model is the actual model to run
        self.s = cfg_scale

    def forward(self, x, timesteps, cond=None, mask=None, motion_guidance = None, spatial_condition=None, **kwargs):
        B, T, D = x.shape

        x_combined = torch.cat([x, x], dim=0)
        timesteps_combined = torch.cat([timesteps, timesteps], dim=0)
        if cond is not None:
            cond = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        if mask is not None:
            mask = torch.cat([mask, mask], dim=0)
        if motion_guidance is not None:
            motion_guidance = torch.cat([motion_guidance, motion_guidance], dim=0)
        if spatial_condition is not None:
            spatial_condition = torch.cat([spatial_condition, spatial_condition], dim=0)
        out = self.model(x_combined, timesteps_combined, cond=cond, mask=mask, motion_guidance=motion_guidance, spatial_condition=spatial_condition, **kwargs)

        out_cond = out[:B]
        out_uncond = out[B:]

        cfg_out = self.s *  out_cond + (1-self.s) *out_uncond
        return cfg_out
