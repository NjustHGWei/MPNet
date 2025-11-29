import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from module._utils import KaimingInitMixin, Identity
from module.backbones import resnet
from torch.nn import init
from module.MSFF import MSFFModule
from module.PCG import PCEModule


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d
class Backbone(nn.Module, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet101':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet152':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # x 3 256 256
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
        )#nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv(x)
        return x



class MSFFPCGNet(nn.Module):
    def __init__(self, in_ch=3, out_c=1):
        super(MSFFPCGNet, self).__init__()
        self.extract = Backbone(in_ch=in_ch, arch='resnet18')  #

        self.fusion1 = MSFFModule(64, 128)
        self.fusion2 = MSFFModule(128, 128)
        self.fusion3 = MSFFModule(256, 128)
        self.fusion4 = MSFFModule(512, 128)


        self.PCEModel = PCEModule(128)

        self.conv1 = BasicConv2d(128, 128, 3, 1, 1)
        self.conv2 = BasicConv2d(128, 128, 3, 1, 1)
        self.conv3 = BasicConv2d(128, 128, 3, 1, 1)
        self.conv4 = BasicConv2d(128, 128, 3, 1, 1)

        self.Up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder1 = BasicConv2d(256, 128, 3, 1, 1)

        self.Up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder2 = BasicConv2d(256, 128, 3, 1, 1)

        self.Up3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.decoder3 = BasicConv2d(256, 128, 3, 1, 1)


        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

        self.decoder_final1 = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1),
                                            nn.Conv2d(64, out_c, 1))

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

    def forward(self, A, B):

        size = A.size()[2:]
        layer1_A, layer2_A, layer3_A, layer4_A = self.extract(A)
        layer1_B, layer2_B, layer3_B, layer4_B = self.extract(B)


        ''' 特征融合 '''
        layer1 = self.fusion1(layer1_B, layer1_A)
        layer2 = self.fusion2(layer2_B, layer2_A)
        layer3 = self.fusion3(layer3_B, layer3_A)
        layer4 = self.fusion4(layer4_B, layer4_A)

        ''' 先验信息引导模块 '''
        f1, f2, f3, f4 = self.PCEModel(layer1, layer2, layer3, layer4)

        layer1 = self.conv1(f1 + layer1)
        layer2 = self.conv2(f2 + layer2)
        layer3 = self.conv3(f3 + layer3)
        layer4 = self.conv4(f4 + layer4)

        feature3 = self.decoder1(torch.cat([layer3, self.Up1(layer4)], 1))
        feature2 = self.decoder2(torch.cat([layer2, self.Up2(feature3)], 1))
        feature1 = self.decoder3(torch.cat([layer1, self.Up3(feature2)], 1))

        final_map = self.decoder_final1(feature1)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return final_map



if __name__=='__main__':
    #测试热图
    net = MSFFPCGNet().cuda()
    out = net(torch.rand((2, 3, 256, 256)).cuda(), torch.rand((2, 3, 256, 256)).cuda())
    print(out.shape)

    net = MSFFPCGNet(3, 1)  # 替换成你自己的模型类或实例化对象
    device = torch.device("cpu")

    input = torch.randn(1, 3, 256, 256)
    input1 = torch.randn(1, 3, 256, 256)
    flops, params = profile(net, inputs=(input, input1))

    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))