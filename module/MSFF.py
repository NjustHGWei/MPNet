import numpy as np
import torch
from torch import nn
from torch.nn import init


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Frequency(nn.Module):
    def __init__(self, nc):
        super(Frequency, self).__init__()


        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))

        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True))


    def forward(self, x):
        _, _, H, W = x.shape
        x = x.to(torch.float32)

        x_freq = torch.fft.rfft2(x, norm='ortho')

        eps = 1e-6
        ori_mag = torch.sqrt(x_freq.real.pow(2) + x_freq.imag.pow(2) + eps)
        ori_pha = torch.atan2(x_freq.imag, x_freq.real + eps)

        mag = self.processmag(ori_mag)
        mag = ori_mag + mag

        pha = self.processpha(ori_pha)
        pha = ori_pha + pha

        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)

        x_out = torch.view_as_complex(torch.stack([real, imag], dim=-1))

        x_freq_spatial = torch.fft.irfft2(x_out, s=(H, W), norm='ortho')

        return x_freq_spatial


class Multiscaleconv1(nn.Module):
    def __init__(self, in_channels, out_c):
        super(Multiscaleconv1, self).__init__()
        self.mid_c = in_channels // 2
        self.conv0 = BasicConv2d(in_channels, out_c, kernel_size=1)
        self.conv1 = Frequency(out_c)
        self.conv3 = BasicConv2d(out_c, out_c, 3, 1, 1)
        self.conv5 = BasicConv2d(out_c*2, out_c, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x = self.conv3(x)
        out = self.conv5(torch.cat([x, x1], 1))
        return out


class Multiscaleconv2(nn.Module):
    def __init__(self, in_channels, out_c):
        super(Multiscaleconv2, self).__init__()
        self.mid_c = in_channels // 2
        self.conv0 = BasicConv2d(in_channels, out_c, kernel_size=1)
        self.conv1 = Frequency(out_c)
        self.conv3 = BasicConv2d(out_c, out_c, 5, 1, 2)
        self.conv5 = BasicConv2d(out_c*2, out_c, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x = self.conv3(x)
        out = self.conv5(torch.cat([x, x1], 1))
        return out

class Multiscaleconv3(nn.Module):
    def __init__(self, in_channels, out_c):
        super(Multiscaleconv3, self).__init__()
        self.mid_c = in_channels // 2
        self.conv0 = BasicConv2d(in_channels, out_c, kernel_size=1)
        self.conv1 = Frequency(out_c)
        self.conv3 = BasicConv2d(out_c, out_c, 7, 1, 3)
        self.conv5 = BasicConv2d(out_c * 2, out_c, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x = self.conv3(x)
        out = self.conv5(torch.cat([x, x1], 1))
        return out

class Multiscaleconv4(nn.Module):
    def __init__(self, in_channels, out_c):
        super(Multiscaleconv4, self).__init__()
        self.mid_c = in_channels//2
        self.conv0 = BasicConv2d(in_channels, out_c, kernel_size=1)
        self.conv1 = Frequency(out_c)
        self.conv3 = BasicConv2d(out_c, out_c, 9, 1, 4)
        self.conv5 = BasicConv2d(out_c * 2, out_c, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x1 = self.conv1(x)
        x = self.conv3(x)
        out = self.conv5(torch.cat([x, x1], 1))
        return out


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bag, self).__init__()
        self.se_blocks = []
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, in_channels, kernel_size=1, bias=False),
                nn.Sigmoid()
            )
            for _ in range(4)
        ])
        self.softmax = nn.Softmax(dim=1)
        self.conv = BasicConv2d(in_channels*4, out_channels, 1)
    def forward(self, p, i, d, f):
        b, c, h, w = d.size()

        pid_out = torch.stack([p, i, d, f], dim=1)


        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(pid_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)  #  [32, 3, 8, 1, 1]
        SE_out = SE_out.expand_as(pid_out)  #  [32, 3, 8, 256, 256]

        softmax_out = self.softmax(SE_out)

        out = pid_out * softmax_out
        out = out.view(b, -1, h, w)

        return self.conv(out)

class MSFFModule(nn.Module):
    def __init__(self, channel, out_c):
        super(MSFFModule, self).__init__()
        self.in_c = channel//4*3
        self.out_c = channel//4

        self.Mconv1 = Multiscaleconv1(self.in_c, self.out_c)
        self.Mconv2 = Multiscaleconv2(self.in_c, self.out_c)
        self.Mconv3 = Multiscaleconv3(self.in_c, self.out_c)
        self.Mconv4 = Multiscaleconv4(self.in_c, self.out_c)

        self.Bag = Bag(self.out_c, out_c)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, y):

        xy_abs1, xy_abs2, xy_abs3, xy_abs4 = torch.split(torch.abs(x - y), x.size(1) // 4, dim=1)
        x1, x2, x3, x4 = torch.split(x, x.size(1) // 4, dim=1)
        y1, y2, y3, y4 = torch.split(y, x.size(1) // 4, dim=1)

        xy_fusion1 = self.Mconv1(torch.cat([x1, y1, xy_abs1], 1))
        xy_fusion2 = self.Mconv2(torch.cat([x2, y2, xy_abs2], 1))
        xy_fusion3 = self.Mconv3(torch.cat([x3, y3, xy_abs3], 1))
        xy_fusion4 = self.Mconv4(torch.cat([x4, y4, xy_abs4], 1))

        output = self.Bag(xy_fusion1, xy_fusion2, xy_fusion3, xy_fusion4)


        return output


if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    input1 = torch.randn(50, 512, 7, 7)
    model = MSFFModule(512, 128)
    output = model(input, input1)
    print(output.shape)

