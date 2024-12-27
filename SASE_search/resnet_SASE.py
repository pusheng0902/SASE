import torch.nn as nn
from torchvision.models import ResNet
import torch.nn.functional as F
from operations import *
from genotypes import *


class MixedOp(nn.Module):

    def __init__(self, C, H, W, OPS):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()  
        for primitive in OPS.keys():
            self._ops.append(OPS[primitive](C, H, W, False))

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Attention(nn.Module):

    def __init__(self, C, H, W, edges):
        super(Attention, self).__init__()
        
        self.C, self.H, self.W = C, H, W
        self._ops = nn.ModuleList()
        
        for primitive in edges:
            self._ops.append(MixedOp(C, H, W, primitive))
            
        self.mixer = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.BatchNorm2d(1, affine=False),
            nn.ReLU(inplace=False)
        )

    def forward(self, s0, weights):
        states=[s0]
        
        enc_out_ch = self._ops[0](states[0], weights[0])
        states.append(enc_out_ch)
        
        dec_out_ch = self._ops[1](states[1], weights[1])
        states.append(dec_out_ch)
        
        dec_interm_ch = self._ops[2](states[1], weights[2])
        states.append(dec_interm_ch * s0)
        
        enc_out_sp = self.mixer( torch.cat([self._ops[3](states[3], weights[3]), self._ops[4](states[0], weights[4])], dim=1) )
        states.append(enc_out_sp)
        
        dec_out_sp = self._ops[5](states[4], weights[5])
        states.append(dec_out_sp)
        
        return s0 * (states[2] + states[-1]) / 2 + s0


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class CifarAttentionBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, H, W):
        super(CifarAttentionBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),nn.BatchNorm2d(planes),)
        else:
            self.downsample = lambda x: x
        self.stride = stride
        
        self.attention=Attention(planes, H, W, edges)

    def forward(self,x,weights):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out,weights)
        out = out + residual
        out = self.relu(out)

        return out

class CifarAttentionBottleNeck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride, downsample, H, W):
        super(CifarAttentionBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        
        self.attention=Attention(planes*4, H, W, edges)
    
    def forward(self, x,weights):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out,weights)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class CifarAttentionResNet(nn.Module):
    def __init__(self, block, n_size, num_classes=10):
        super(CifarAttentionResNet, self).__init__()
        self.inplane = 16
        self.channel_in = 16
        self.conv1 = nn.Conv2d(3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, self.channel_in,   blocks=n_size, stride=1, H=32, W=32)
        self.layer2 = self._make_layer(block, self.channel_in*2, blocks=n_size, stride=2, H=16, W=16)
        self.layer3 = self._make_layer(block, self.channel_in*4, blocks=n_size, stride=2, H=8, W=8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.channel_in*4, num_classes)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, H, W):
        strides = [stride] + [1] * (blocks - 1)
        self.layers = nn.ModuleList()
        for stride in strides:
            Block = block(self.inplane, planes, stride, H, W)
            self.layers +=  [Block]
            self.inplane = planes
        return self.layers

    def forward(self, x, weights):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for i, layer in enumerate(self.layer1):
            x = layer(x,weights)
        for i, layer in enumerate(self.layer2):
            x = layer(x,weights)
        for i, layer in enumerate(self.layer3):
            x = layer(x,weights)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def attention_resnet20(**kwargs):
    """Constructs a ResNet-20 model.
    """
    model = CifarAttentionResNet(CifarAttentionBasicBlock, 3, **kwargs)
    return model
