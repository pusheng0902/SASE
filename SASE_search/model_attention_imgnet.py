import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torchvision.models import ResNet


class Attention(nn.Module):
    def __init__(self, C, H, W, genotype):
        super(Attention, self).__init__()
        
        self.C, self.H, self.W = C, H, W
        self._ops = nn.ModuleList()
        self.genotype = genotype
        op_names, indices = zip(*genotype)
        self._compile(self.C, H, W, op_names, indices)
        
        self.mixer = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.BatchNorm2d(1, affine=True),
            nn.ReLU(inplace=False)
        )
        
    def _compile(self, C, H, W, op_names, indices):
        assert len(op_names) == len(indices)
        
        self.indices = indices
        self._ops = nn.ModuleList()
        
        # From left to right, top to bottom
        for op_name, idx in zip(op_names, indices):
            self._ops.append(edges[idx][op_name](C, H, W, True))
    
    def forward(self, s0):
        states=[s0]
        
        enc_out_ch = self._ops[0](states[0])
        states.append(enc_out_ch)
        
        dec_out_ch = self._ops[1](states[1])
        states.append(dec_out_ch)
        
        dec_interm_ch = self._ops[2](states[1])
        states.append(dec_interm_ch * s0)
        
        enc_out_sp = self.mixer( torch.cat([self._ops[3](states[3]), self._ops[4](states[0])], dim=1) )
        states.append(enc_out_sp)
        
        dec_out_sp = self._ops[5](states[4])
        states.append(dec_out_sp)
        
        return s0 * (states[2] + states[-1]) / 2 + s0


class SASEBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SASEBottleneck, self).__init__()
        c2wh = dict([(64,56), (128,28), (256,14) ,(512,7)])
        
        self.planes = planes
        self.downsample = downsample
        self.stride = stride
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        genotype = [('GAP+GMP', 0), ('2conv_1x1', 1), ('affine', 2), ('p-norm', 3), ('GAP', 4), ('2conv_3x3', 5)] # chspMixerRes cifar10unrolled seed 2
        
        self.attention = Attention(planes * 4, c2wh[planes], c2wh[planes], genotype)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out
        

def resnet50(num_classes=1000, pretrained=False):
    model = ResNet(SASEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def resnet101(num_classes=1000, pretrained=False):
    model = ResNet(SASEBottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
