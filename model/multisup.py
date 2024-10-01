import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import cv2
import numpy as np
import pywt
from model.module import *
from fvcore.nn import FlopCountAnalysis

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x

class Conv2D(nn.Module):
    def __init__(self, in_chl, nf, n_blks=2, act='relu'):
        super(Conv2D, self).__init__()

        block = functools.partial(ResidualBlock, nf=nf)
        self.conv_L1 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
        self.blk_L1 = make_layer(block, n_layers=n_blks)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        fea_L1 = self.blk_L1(self.act(self.conv_L1(x)))

        return fea_L1



class FFU(torch.nn.Module):
    def __init__(self, dim=96):
        super(FFU, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.mul_conv1 = nn.Conv2d(dim*2, 192, kernel_size=3, stride=1, padding=1)
        self.mul_conv2 = nn.Conv2d(192, dim, kernel_size=3, stride=1, padding=1)
        self.add_conv1 = nn.Conv2d(dim*2, 192, kernel_size=3, stride=1, padding=1)
        self.add_conv2 = nn.Conv2d(192, dim, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, tar, ref):
        ref_residual = self.conv(self.conv(ref)-tar)+ref
        tar_residual = self.conv(tar-self.conv(ref))+tar
        cat_input = torch.cat((tar_residual, ref_residual), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.lrelu(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.lrelu(self.add_conv1(cat_input)))
        return tar * mul + add

    

class WavMCVM(nn.Module):
    def __init__(self, upscale, 
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
                downsample=None,
                use_checkpoint=False):
        super(WavMCVM, self).__init__()
        self.upscale = upscale
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True, count_include_pad=False)
        self.conv2d = Conv2D(in_chl=inchans, nf=dim, n_blks=depth)
        self.conv_first = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=(3, 3), stride=1, padding=1)
        self.conv_second = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=2, padding=1)
        self.conv_third = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=2, padding=1)
        self.reduce_conv = nn.Conv2d(192, 96, kernel_size=1)  # 1x1 convolution to reduce channels from 192 to 96
        self.stvmunit = STVMUnit(inchans=dim, outchans=dim, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path[0], norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, is_cross=False, downsample=downsample, use_checkpoint=use_checkpoint)
        self.hybridstm = HybridSTM(inchans=dim, outchans=dim, dim=dim, depth=depth, d_state=d_state, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer, patch_size=patch_size, patch_norm=patch_norm, downsample=None, use_checkpoint=False)
        self.conv_last = nn.Conv2d(in_channels=dim, out_channels=outchans, kernel_size=(3, 3), stride=1, padding=1)
        self.ffu = FFU(dim=dim)

    def wavelet_high_freq(self, image):
        # Convert tensor to NumPy for wavelet transform
        img_np = image.detach().cpu().numpy()  # 转换为 NumPy 数组 (N, C, H, W)
        high_freq_components = []
        
        # 对每个通道进行小波变换，并提取高频特征
        for c in range(img_np.shape[1]):
            coeffs = pywt.dwt2(img_np[:, c, :, :], 'haar')  # 对每个通道进行2D小波变换
            LL, (LH, HL, HH) = coeffs  # LL为低频分量, LH/HL/HH为高频分量
            high_freq_components.append(HH)  # 提取最高频分量HH

        # 将每个通道的高频分量合并回张量
        high_freq_np = np.stack(high_freq_components, axis=1)  # (N, C, H, W)
        high_freq_tensor = torch.from_numpy(high_freq_np).cuda()  # 转换回 tensor

        
        return high_freq_tensor


    def forward(self, tar, ref):
        tar_lr = self.conv2d(tar)
        tar_lr = self.stvmunit(tar_lr)

        # 小波变换
        ref_wavelet_1 = self.wavelet_high_freq(ref)
        ref_wavelet_1 = self.conv_first(ref_wavelet_1)
        ref_wavelet_1 = self.stvmunit(ref_wavelet_1)
        ref_wavelet_2 = self.conv_second(ref_wavelet_1)
        ref_wavelet_2 = self.stvmunit(ref_wavelet_2)
        

        ref_0 = self.conv_first(ref)
        ref_0 = self.stvmunit(ref_0)
        ref_1 = self.conv_second(ref_0)
        ref_1 = self.stvmunit(ref_1)
        ref_2 = self.conv_third(ref_1)
        ref_2 = self.stvmunit(ref_2)

        if self.upscale == 2:

            fuse_1 = self.ffu(tar_lr, ref_1)
            fuse_1 = self.hybridstm(fuse_1, style=ref_wavelet_1)
            fuse_1 = self.up(fuse_1)

            fuse_2 = self.ffu(fuse_1, ref_0)
            fuse_2 = self.hybridstm(fuse_2, style=F.interpolate(ref_wavelet_1, scale_factor=2, mode='bilinear', align_corners=False))


            out = self.conv_last(fuse_2)

        if self.upscale == 4:

            fuse_0 = self.ffu(tar_lr, ref_2)
            fuse_0 = self.hybridstm(fuse_0, style=ref_wavelet_2)
            fuse_0 = self.up(fuse_0)

            fuse_1 = self.ffu(fuse_0, ref_1)
            fuse_1 = self.hybridstm(fuse_1, style=ref_wavelet_1)
            fuse_1 = self.up(fuse_1)

            fuse_2 = self.ffu(fuse_1, ref_0)
            fuse_2 = self.hybridstm(fuse_2, style=F.interpolate(ref_wavelet_1, scale_factor=2, mode='bilinear', align_corners=False))

            out = self.conv_last(fuse_2)

        return out

        


    
if __name__ == '__main__':
    upscale = 2
    inchans = 3
    outchans = 3
    height = 56
    width = 56
    dim = 96
    depths = 2
    drop_rate = 0.
    drop_path = [0, 0.1]
    drop_path_rate = 0.1
    norm_layer=nn.LayerNorm
    attn_drop_rate=0.
    d_state=16
    patch_size=4
    patch_norm = True

    tar = torch.randn((1, 3, height, width)).cuda()
    ref = torch.randn((1, 3, height*upscale, width*upscale)).cuda()
    model = WavMCVM(upscale=upscale,
                inchans=inchans,
                outchans=outchans,
                dim=dim,
                depth=depths,
                d_state=d_state, # 20240109
                drop=drop_rate, 
                attn_drop=attn_drop_rate,
                drop_path=drop_path,
                norm_layer=norm_layer,
                patch_size=patch_size,
                patch_norm=patch_norm,
                downsample=None,
                use_checkpoint=False,).cuda()
    out = model(tar, ref)
    print(out.shape)

    num_param = sum([p.numel() for p in model.parameters() if p.requires_grad]) / 1e6
    print(f'Total Params: {num_param:.2f}M')
    flops = FlopCountAnalysis(model, (tar, ref)) 
    print(f"Total FLOPs: {flops.total() / 1e9:.2f}G")
    # Calculate Inference Time
    start_time = time.time()
    _ = model(tar, ref)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print(f"Inference Time per Image: {inference_time:.2f} ms")
