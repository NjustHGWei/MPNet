import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SA(nn.Module):
    def __init__(self, in_dim):
        super(SA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)

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
        m_batchsize, C, height, width = x.size()
        query = self.query_conv(x)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

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
        b, c, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual

class PCEModule(nn.Module):
    def __init__(self, channel):
        super(PCEModule, self).__init__()

        self.conv = BasicConv2d(channel * 2, channel * 2, 1)
        self.conv1 = BasicConv2d(channel * 2, channel * 2, 1)

        self.conv2 = nn.Sequential(BasicConv2d(channel * 2, channel, 3, 1, 1),
                                   SA(channel),
                                   nn.Conv2d(channel, 1, 1))

        self.conv3 = nn.Sequential(BasicConv2d(channel * 4, channel, 3, 1, 1))

        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)
        self.downsample2x = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample4x = nn.AvgPool2d(kernel_size=4, stride=4)

        self.conv5 = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv6 = BasicConv2d(channel, channel, 3, 1, 1)
        self.conv7 = BasicConv2d(channel, channel, 3, 1, 1)

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

    def forward(self, x1, x2, x3, x4):
        size2 = x2.size()[2:]

        x1 = F.interpolate(x1, size2, mode='bilinear', align_corners=True)
        x12_cat = self.conv(torch.cat([x1, x2], 1))
        x34_cat = self.conv1(torch.cat([x3, self.upsample2x(x4)], 1))

        x34_cat_mapping = self.upsample2x(self.conv2(x34_cat))
        x12_cat = (torch.sigmoid(x34_cat_mapping) + 1) * x12_cat + x12_cat

        x_cat = torch.cat([x12_cat, self.upsample2x(x34_cat)], 1)

        xy_mapping = self.conv3(x_cat)

        f1 = self.conv5(self.upsample2x(xy_mapping))
        f2 = xy_mapping
        f3 = self.conv6(self.downsample2x(xy_mapping))
        f4 = self.conv7(self.downsample4x(xy_mapping))

        return f1, f2, f3, f4


if __name__ == '__main__':
    input1 = torch.randn(50, 128, 64, 64)
    input2 = torch.randn(50, 128, 32, 32)
    input3 = torch.randn(50, 128, 16, 16)
    input4 = torch.randn(50, 128, 8, 8)
    model = PCEModule(128)
    output = model(input1, input2, input3, input4)[3]
    print(output.shape)

