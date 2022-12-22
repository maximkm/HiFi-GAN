import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm


def init_weights(module, mean=0., std=1e-2):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        module.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.dilation = dilation

        self.conv_blocks1 = nn.ModuleList()
        self.conv_blocks2 = nn.ModuleList()

        for d in dilation:
            pad_d = get_padding(kernel_size, d)
            pad_1 = get_padding(kernel_size, 1)
            self.conv_blocks1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=pad_d)))
            self.conv_blocks2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=pad_1)))

        self.conv_blocks1.apply(init_weights)
        self.conv_blocks2.apply(init_weights)

    def forward(self, x):
        for i in range(len(self.dilation)):
            skip = x
            x = F.leaky_relu(x, 0.1)
            x = self.conv_blocks1[i](x)
            x = F.leaky_relu(x, 0.1)
            x = self.conv_blocks2[i](x) + skip
        return x

    def remove_weight_norm(self):
        for conv in self.conv_blocks1:
            remove_weight_norm(conv)
        for conv in self.conv_blocks2:
            remove_weight_norm(conv)


class Generator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_ups = len(config.upsample_rates)
        self.conv_pre = weight_norm(nn.Conv1d(80, config.upsample_initial_channel, 7, 1, 3))

        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        out_channels = None
        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            pad = (k - u) // 2
            in_channels = config.upsample_initial_channel // (2 ** i)
            out_channels = config.upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(weight_norm(nn.ConvTranspose1d(in_channels, out_channels, k, u, pad)))

            for (k_res, d) in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
                self.res_blocks.append(ResBlock(out_channels, k_res, d))

        assert out_channels is not None
        self.conv_post = weight_norm(nn.Conv1d(out_channels, 1, 7, 1, 3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i in range(self.num_ups):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x_skip = self.res_blocks[i * self.num_kernels](x)
            for j in range(1, self.num_kernels):
                x_skip += self.res_blocks[i * self.num_kernels + j](x)
            x = x_skip / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        for conv_tran in self.ups:
            remove_weight_norm(conv_tran)
        for res_block in self.res_blocks:
            res_block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class PeriodBlock(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.conv_blocks = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (get_padding(5, 1), 0))),
            weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x, fmap):
        tmp = []
        bsz, ch, t = x.shape
        if t % self.period != 0:
            pad = self.period - (t % self.period)
            x = F.pad(x, (0, pad), "reflect")
            t = t + pad
        x = x.view(bsz, ch, t // self.period, self.period)

        for conv in self.conv_blocks:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            tmp.append(x)

        x = self.conv_post(x)
        tmp.append(x)
        fmap.append(tmp)
        return x.flatten(1, -1)


class ScaleBlock(torch.nn.Module):
    def __init__(self, conv_norm=weight_norm):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            conv_norm(nn.Conv1d(1, 128, 15, 1, 7)),
            conv_norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            conv_norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            conv_norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            conv_norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            conv_norm(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            conv_norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = conv_norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x, fmap):
        tmp = []
        for conv in self.conv_blocks:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            tmp.append(x)

        x = self.conv_post(x)
        tmp.append(x)
        fmap.append(tmp)
        return x.flatten(1, -1)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodBlock(2),
            PeriodBlock(3),
            PeriodBlock(5),
            PeriodBlock(7),
            PeriodBlock(11),
        ])

    def forward(self, y, y_hat):
        y_rs, y_gs = [], []
        fmap_rs, fmap_gs = [], []
        for i, d in enumerate(self.discriminators):
            y_r = d(y, fmap_rs)
            y_g = d(y_hat, fmap_gs)
            y_rs.append(y_r)
            y_gs.append(y_g)
        return y_rs, y_gs, fmap_rs, fmap_gs


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleBlock(spectral_norm),
            ScaleBlock(weight_norm),
            ScaleBlock(weight_norm),
        ])
        self.mean_pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        fmap_rs, fmap_gs = [], []
        y_r = self.discriminators[0](y, fmap_rs)
        y_g = self.discriminators[0](y_hat, fmap_gs)
        y_rs, y_gs = [y_r], [y_g]
        for i, pool in enumerate(self.mean_pools):
            y = pool(y)
            y_hat = pool(y_hat)
            y_r = self.discriminators[i + 1](y, fmap_rs)
            y_g = self.discriminators[i + 1](y_hat, fmap_gs)
            y_rs.append(y_r)
            y_gs.append(y_g)
        return y_rs, y_gs, fmap_rs, fmap_gs
