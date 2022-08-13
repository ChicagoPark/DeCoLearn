import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from torch.nn.functional import pad


activation_fn = {
     'relu': lambda: nn.ReLU(),
    'lrelu': lambda: nn.LeakyReLU(0.2),
}

####################################
# Registration Neutral Network
####################################

#Some implementation here is adopted from VoxelMorph.


# noinspection PyUnresolvedReferences
class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding
                            layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2 if dim == 2 else (1, 2, 2), mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        return y


class cvpr2018_net(nn.Module):
    """
    [cvpr2018_net] is a class representing the specific implementation for
    the 2018 implementation of voxelmorph.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
        """
        super(cvpr2018_net, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape), requires_grad=True)
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape), requires_grad=True)

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        x = self.unet_model(x)
        flow = self.flow(x)
        y = self.spatial_transform(src, flow)

        return y, flow


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # noinspection PyArgumentList
        return nnf.grid_sample(src, new_locs, mode=self.mode, align_corners=False)


class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        padding = 1
        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        if dim == 3:
            stride = (1, stride, stride)
            ksize = 3
            padding = (ksize // 2, 1, 1)

        self.main = conv_fn(in_channels, out_channels, ksize, stride, padding)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


##########################
# EDSR network
##########################
class ResBlock(nn.Module):
    def __init__(self, dimension, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        else:
            raise ValueError()

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(bn_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class EDSR(nn.Module):
    def __init__(self, dimension, n_resblocks, n_feats, res_scale, in_channels=1, out_channels=1, act='relu'):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
        else:
            raise ValueError()

        m_head = [conv_fn(in_channels, n_feats, 3, padding=3 // 2)]

        m_body = [
            ResBlock(
                dimension, n_feats, 3, res_scale=res_scale, act=activation_fn[act](),
            ) for _ in range(n_resblocks)
        ]

        m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))

        m_tail = [
            conv_fn(n_feats, out_channels, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, P = None, S = None, y = None):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x

from torch.nn.utils import spectral_norm

class DnCNN(nn.Module):
    def __init__(self, dimension = 2, depth=13, n_channels=64, i_nc=2, o_nc=2, kernel_size=3, is_batch_normalize=False, is_residual=True):

        self.is_residual = is_residual

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d

        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d

        else:
            raise ValueError()

        super().__init__()
        padding = kernel_size // 2

        layers = [
            spectral_norm(conv_fn(
                in_channels=i_nc,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False)),
            nn.ReLU(inplace=True)
        ]

        for _ in range(depth - 1):
            layers.append(spectral_norm(conv_fn(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False if is_batch_normalize else True)))

            if is_batch_normalize:
                layers.append(bn_fn(n_channels))

            layers.append(nn.ReLU(inplace=True))

        layers.append(
            spectral_norm(conv_fn(
                in_channels=n_channels,
                out_channels=o_nc,
                kernel_size=kernel_size,
                padding=padding,
                bias=False)))

        self.net = nn.Sequential(*layers)

    def forward(self, x, P = None, S = None, y = None):

        input_ = x

        x = self.net(x)

        ret = input_ - x if self.is_residual else x

        return ret

class CNNBlock(nn.Module):
    name = 'CNN'

    def __init__(self):
        super().__init__()

        # self.nn = DnCNN(
        #     dimension=2,
        #     depth=5,
        #     n_channels=32,
        #     i_nc=2,
        #     o_nc=2,
        # )

        self.nn = UNet(
            dimension=2,
            i_nc=2,
            o_nc=2,
            f_root=32,
            conv_times=3,
            is_bn=False,
            activation='relu',
            is_residual=False,
            up_down_times=3,
            is_spe_norm=True,
            padding=(0, 0)
        )

    def forward(self, x, P = None, S = None, y = None):
        '''
        x_hat = torch.view_as_real(x).permute([0, 3, 1, 2])        # x_hat shape before network: torch.Size([8, 2, 256, 232])

        print(f"\nx_hat size before network: {x_hat.shape}")

        # Youngil: Check the shape before and after
        x_hat = self.nn(x_hat)
        x_hat = torch.view_as_complex(x_hat.permute([0, 2, 3, 1])) # x_hat shape after network: torch.Size([8, 256, 232])

        print(f"\nx_hat size after network: {x_hat.shape}")

        return x_hat
        '''
        x_hat = self.nn(x)

        return x_hat

class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(spectral_norm(conv_fn(in_channels)) if is_spe_norm else conv_fn(in_channels))
            else:
                layers.append(spectral_norm(conv_fn(out_channels)) if is_spe_norm else conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvtranBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3, is_spe_norm=False):
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = spectral_norm(conv_fn()) if is_spe_norm else conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret


class UNet(nn.Module):
    def __init__(self, dimension, i_nc=1, o_nc=1, f_root=32, conv_times=3, is_bn=False, activation='relu',
                 is_residual=False, up_down_times=3, is_spe_norm=False, padding=(1, 4)):

        self.is_residual = is_residual
        self.up_down_time = up_down_times
        self.dimension = dimension
        self.padding = padding

        super().__init__()

        if dimension == 2:
            self.down_sample = nn.MaxPool2d((2, 2))
        elif dimension == 3:
            self.down_sample = nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None,
            is_spe_norm=is_spe_norm
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (up_down_times - 1)),
            out_channels=f_root * (2 ** up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
            is_spe_norm=is_spe_norm
        )

        self.down_list = nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                           )
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension,
                                               is_spe_norm=is_spe_norm
                                            )
                                           for i in range(up_down_times - 1)
                                       ])

        self.up_conv_list = nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

        self.up_conv_tran_list = nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension,
                is_spe_norm=is_spe_norm
            )
            for i in range(up_down_times)
        ])

    def forward(self, x, P = None, S = None, y = None):

        input_ = x

        x = pad(x, [0, self.padding[0], 0, self.padding[1]])

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_time):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_time):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_time - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        if self.padding[0] > 0:
            x = x[..., :-self.padding[0]]
        if self.padding[1] > 0:
            x = x[..., :-self.padding[1], :]

        # x = x[..., :-self.padding[1], :-self.padding[0]]

        ret = input_ - x if self.is_residual else x

        return ret

def fmult(x, S, P):
    # x, groundtruth, shape: batch, width, height; dtype: complex
    # S, sensitivity maps, shape: batch, coils, width, height; dtype: complex
    # P, sampling mask, shape: batch, width, height; dtype: float/bool

    # compute forward of fast MRI, y = PFSx

    # S
    x = torch.view_as_complex(x).unsqueeze(1)

    x = x * S

    # F
    y = torch.fft.fft2(x)

    # P
    P = torch.view_as_complex(P).unsqueeze(1)
    y = y * P

    y = torch.view_as_real(y)

    return y

def ftran(y, S, P):
    # y, under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    # S, sensitivity maps, shape: batch, coils, width, height; dtype: complex
    # P, sampling mask, shape: batch, width, height; dtype: float/bool

    # compute adjoint of fast MRI, x = S^H F^H P^H x
    y = torch.view_as_complex(y)
    P = torch.view_as_complex(P)
    P = P.unsqueeze(1)

    # P^H
    y = y * P

    # F^H
    x = torch.fft.ifft2(y)

    # S^H

    x = x * torch.conj(S)

    x = x.sum(1)

    x = torch.view_as_real(x)

    return x

import json

class DeepUnfoldingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        with open('config.json') as File:
            config = json.load(File)

        self.nn = EDSR(
            n_resblocks=config['module']['recon']['EDSR']['n_resblocks'],
            n_feats=config['module']['recon']['EDSR']['n_feats'],
            res_scale=config['module']['recon']['EDSR']['res_scale'],
            in_channels=2,
            out_channels=2,
            dimension=2, )
        self.gamma = 0.01
        self.alpha = 1.0

    def forward(self, x, P, S, y):
        x = x.permute([0, 2, 3, 1]).contiguous()

        with open('config.json') as File:
            config = json.load(File)

        dc = fmult(x, S, P)  # A x
        dc = ftran(dc - y, S, P)  # A^H (Ax - y)
        x = x - self.gamma * dc  # x^+ = x - gamma * A^H (Ax - y)

        x = x.permute([0, 3, 1, 2]).contiguous()

        prior = self.nn(x, P, S, y)

        return self.alpha * prior + (1 - self.alpha) * x

class DeepUnfolding(nn.Module):
    name = 'DU'

    def __init__(self, iterations):
        super().__init__()
        self.du_block = nn.ModuleList()
        self.iterations = iterations
        for i in range(self.iterations):
            DUblock = DeepUnfoldingBlock()
            self.du_block.append(DUblock)

    def forward(self, x, P, S, y):

        for i in range(self.iterations):
            x = self.du_block[i](x, P, S, y)

        return x