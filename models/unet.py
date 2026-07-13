import math
import torch
import torch.nn as nn
import ipdb
import numpy as np
from torch.nn import functional as F
from torch.backends.cuda import sdp_kernel



def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, hw):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.memMatrix = nn.Parameter(torch.randn(num_slots,slot_dim,hw,hw))  # M,C
        self.q = nn.Conv2d(slot_dim,
                                 slot_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(slot_dim,
                                 slot_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv2d(slot_dim,
                                 slot_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x, p):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """  
        
        # 패치 그리드 열 개수 = sqrt(num_slots). 원래 하드코딩 4는 num_slots=16(=4x4, image_size 256)일 때와 동일.
        cols = int(round(self.num_slots ** 0.5))
        part_flag = (torch.round(p)[:,0]*cols + torch.round(p)[:,1]).long()
        match_part = self.memMatrix[part_flag]
       
        q = self.q(x)
        k = self.k(match_part)
        v = self.v(match_part)
        b, c, h, w = q.shape
        q = q.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous() 
        k = k.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous() 
        with sdp_kernel(enable_math=False):
            h_ = F.scaled_dot_product_attention(q, k, v) # require pytorch 2.0
        h_ = h_.permute(0, 1, 3, 2).reshape(b, c, h, w)
        return h_
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.silu = nn.SiLU()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels is not None:
            self.temb_proj = nn.Linear(temb_channels,
                                         2*out_channels)        
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)
        if temb is not None:
            emb_out = self.temb_proj(self.silu(temb))[:, :, None, None]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = self.norm2(h)*(1+scale)+shift
            h = self.silu(h)
            h = self.dropout(h)
            h = self.conv2(h)
        else:
            h = self.norm2(h)
            h = self.silu(h)
            h = self.dropout(h)
            h = self.conv2(h)

            # + self.mo_proj(nonlinearity(mo_semantic))[:, :, None, None]
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous() 
        k = k.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(b, 1, -1, h*w).permute(0, 1, 3, 2).contiguous() 
        with sdp_kernel(enable_math=False):
            h_ = F.scaled_dot_product_attention(q, k, v) # require pytorch 2.0
        h_ = h_.permute(0, 1, 3, 2).reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x+h_

class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.patch_size
        resamp_with_conv = config.model.resamp_with_conv
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.silu = nn.SiLU() 
        # timestep embedding
       
        self.temb = nn.Sequential(
            nn.Linear(self.ch,self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch,self.temb_ch),
        )

        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        
        
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        
        

    def forward(self, xt, motion, ap_hs ,ap_semantic, t):
        B,_,_,H,W = xt.shape
        assert H == W == self.resolution

        x = torch.cat([motion,xt],dim=1).reshape(B,-1,H,W)
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb(temb)
        temb = temb + ap_semantic

        hs = [self.conv_in(x)] 
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb) 
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb) 
        
        # upsamplings
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop() + ap_hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h


class Encoder(nn.Module):
    def __init__(self, config ,in_channels):
        super().__init__()
        self.config = config
        memory_size = config.model.m_size
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = in_channels
        resolution = config.data.patch_size
        resamp_with_conv = config.model.resamp_with_conv
        self.ch = ch

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution


        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        # 메모리 뱅크 슬롯 수 = 패치 그리드 크기². 원래 하드코딩 16은 image_size 256 / patch 64 = 4x4 일 때와 동일.
        grid_n = config.data.image_size // config.data.patch_size
        self.ap_mem = Memory(grid_n * grid_n, block_in, memory_size)
        self.mem_pool = nn.Sequential(
                Normalize(block_in),
                nn.SiLU(),
                nn.Conv2d(block_in,
                                block_in,
                                kernel_size=1,
                                stride=1,
                                padding=0),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        
        
    def forward(self, x, p):
        hs = []
        h = self.conv_in(x)
        hs.append(h)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                h = self.down[i_level].downsample(h)
                hs.append(h)
        ap_h = self.ap_mem(h,p)
        ap_semantic = self.mem_pool(ap_h)
        return hs, ap_h, ap_semantic
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resolution = config.data.patch_size
        resamp_with_conv = config.model.resamp_with_conv
        self.ch = ch
        in_ch_mult = (1,)+ch_mult
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        curr_res = self.resolution//(2**(len(ch_mult)-1))
        block_in = ch*ch_mult[-1]
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                            out_channels=block_out,
                                            dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.silu = nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)      
    def forward(self,h):
      
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h
class DiffusionMA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.diffUNet = UNet(config)
        self.apEncoder = Encoder(config, in_channels=3)
        self.apDecoder = Decoder(config)
    def forward(self,x,t,p):
        ap = x[:,0]
        mo = x[:,1:-1] - x[:,:-2]
        hs, ap_h, ap_semantic = self.apEncoder(ap,p)
        ap_recon = self.apDecoder(ap_h)
        noise = self.diffUNet(x[:,-1:],mo,hs,ap_semantic,t)
        return noise, ap_recon