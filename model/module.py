import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model.vmamba import *

from timm.models.layers import DropPath, trunc_normal_

        
class STVMUnit(nn.Module):
    def __init__(self, 
                inchans,
                outchans,
                dim,
                depth,
                d_state, # 20240109
                drop, 
                attn_drop,
                drop_path,
                norm_layer,
                patch_size,
                patch_norm,
                is_cross,
                downsample=None,
                use_checkpoint=False):
        super(STVMUnit, self).__init__()
        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=inchans, embed_dim=dim,
            norm_layer=norm_layer if patch_norm else None)
        self.Layer = STVSSLayer(
                dim=dim,
                depth=depth,
                d_state=d_state, # 20240109
                drop=drop, 
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
                is_cross=is_cross,
                downsample=None,
                use_checkpoint=False,
            )
        self.pos_drop = nn.Dropout(p=drop)
        self.final_conv = nn.Conv2d(dim, outchans, kernel_size=1)

    def forward(self, x, style=None):
        x = self.patch_embed(x)
        if style is not None:
            style = self.patch_embed(style)
        x = self.Layer(x, style=style)
        x = self.pos_drop(x)   
        x = x.permute(0,3,1,2)
        x = self.final_conv(x)

        return x


class HybridSTM(nn.Module): # Hybrid CNN-VMamba Block
    def __init__(self, inchans,
                outchans,
                dim,
                depth,
                d_state, # 20240109
                drop, 
                attn_drop,
                drop_path,
                norm_layer,
                patch_size,
                patch_norm,
                downsample=None,
                use_checkpoint=False):
        super(HybridSTM, self).__init__()
        self.stvmunit1 = STVMUnit(inchans=inchans, outchans=outchans, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path[0], norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, is_cross=True, downsample=downsample, use_checkpoint=use_checkpoint)
        self.stvmunit2 = STVMUnit(inchans=dim, outchans=dim, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path[1], norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, is_cross=True, downsample=downsample, use_checkpoint=use_checkpoint)

    def forward(self, x, style=None):

        residual = x
        stvmunit_feat = self.stvmunit1(x, style=style)
        x = self.stvmunit2(stvmunit_feat, style=style)+ residual

        return x
