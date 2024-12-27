import torch
import torch.nn as nn
import torch.nn.functional as F
import math


squeeze_channel = {
    'GAP' :  lambda C, H, W, affine: GAP_ch(),
    'L2-norm' : lambda C, H, W, affine: L2norm_ch(C),
    'p-norm' : lambda C, H, W, affine: Pnorm_ch(4),
    'GAP+GMP' : lambda C, H, W, affine: GAPnGMP_ch(C),
    'GAP+std' : lambda C, H, W, affine: GAPnSTD_ch(C),
    'SkewPooling' : lambda C, H, W, affine: GAPnSKP_ch(C, H, W),
    '2nd-order pooling' : lambda C, H, W, affine: BilinearPooling_ch(C, affine)
}

excitation_channel = {
    '3conv1D_3': lambda C, H, W, affine: Conv1D_3layers(C, 3),
    '2conv_1x1': lambda C, H, W, affine: nn.Sequential(
          nn.Conv2d(C, max(1, C // 16), 1, bias=False),
          nn.ReLU(),
          nn.Conv2d(max(1, C // 16), C, 1, bias=False)
    ), # FC
    'conv1D_3': lambda C, H, W, affine: Conv1D(C, 3),
    'conv1D_5': lambda C, H, W, affine: Conv1D(C, 5),
    'conv1D_7': lambda C, H, W, affine: Conv1D(C, 7),
    '2conv1D_3': lambda C, H, W, affine: Conv1D_2layers(C, 3),
    'affine': lambda C, H, W, affine: Affine_ch(C)
}

squeeze_spatial = {
    'GAP' :  lambda C, H, W, affine: GAP_sp(),
    'L2-norm' : lambda C, H, W, affine: L2norm_sp(H, W),
    'p-norm' : lambda C, H, W, affine: Pnorm_sp(4),
    'GAP+GMP' : lambda C, H, W, affine: GAPnGMP_sp(),
    'GAP+std' : lambda C, H, W, affine: GAPnSTD_sp(),
    'SkewPooling' : lambda C, H, W, affine: GAPnSKP_sp(C),
    '2nd-order pooling' : lambda C, H, W, affine: BilinearPooling_sp(H, W, affine)
}

excitation_spatial = {
    'conv_3x1_1x3': lambda C, H, W, affine: nn.Sequential(
        nn.Conv2d(1, 1, (1, 3), stride=(1, 1), padding=(0, 1), bias=False),
        nn.Conv2d(1, 1, (3, 1), stride=(1, 1), padding=(1, 0), bias=False),
        nn.Sigmoid()
    ),
    'conv_5x1_1x5': lambda C, H, W, affine: nn.Sequential(
        nn.Conv2d(1, 1, (1, 5), stride=(1, 1), padding=(0, 2), bias=False),
        nn.Conv2d(1, 1, (5, 1), stride=(1, 1), padding=(2, 0), bias=False),
        nn.Sigmoid()
    ),
    'conv_3x3': lambda C, H, W, affine: nn.Sequential(
          nn.Conv2d(1, 1, 3, bias=False, padding=1),
          nn.Sigmoid()
    ),
    'conv_5x5': lambda C, H, W, affine: nn.Sequential(
          nn.Conv2d(1, 1, 5, bias=False, padding=2),
          nn.Sigmoid()
    ),
    'conv_7x7': lambda C, H, W, affine: nn.Sequential(
          nn.Conv2d(1, 1, 7, bias=False, padding=3),
          nn.Sigmoid()
    ),
    '2conv_3x3': lambda C, H, W, affine: nn.Sequential(
          nn.Conv2d(1, 1, 3, bias=False, padding=1),
          nn.ReLU(inplace=False),
          nn.Conv2d(1, 1, 3, bias=False, padding=1),
          nn.Sigmoid()
    ),
    'affine': lambda C, H, W, affine: Affine_sp(H, W)
}

edges = [squeeze_channel] + 2*[excitation_channel] + 2*[squeeze_spatial] + 1*[excitation_spatial]


#-------------------------------------  Channel-wise Squeezing  ----------------------------------------
#----------------- Helper class --------------------
class ChannelMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        # x(B, C, 1, 2) --> (B, C, 1, 1)
        return self.mixer(x.transpose(-1, -3)).transpose(-1, -3)


#-----------------GAP--------------------
class GAP_ch(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        channel_out = self.avg_pool(x)
        
        return channel_out # B, C, 1, 1


#-----------------GAP + GMP--------------------
class GAPnGMP_ch(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        self.mixer = ChannelMixer()
        
    
    def forward(self, x):
        avg_out = self.avgpool(x) # B, C, 1, 1
        max_out = self.maxpool(x) # B, C, 1, 1
        
        channel_out = torch.cat( (avg_out, max_out), dim=-1 ) # B, C, 1, 2
        
        # Mixup the two channels for fixed output sizes
        channel_out = self.mixer(channel_out)
        
        return channel_out # B, C, 1, 1
        

#-----------------GAP + STD--------------------
class GAPnSTD_ch(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        self.mixer = ChannelMixer()
    
    def forward(self, x):
        avg_out = self.avgpool(x) # B, C, 1, 1
        std_out = torch.std(x,dim=(2,3),keepdim=True) # B, C, 1, 1
        
        channel_out = torch.cat( (avg_out, std_out), dim=-1 ) # B, C, 1, 2
        
        # Mixup the two channels for fixed output sizes
        channel_out = self.mixer(channel_out)
        
        return channel_out # B, C, 1, 1

        
#----------------- L2-norm pooling--------------------
class L2norm_ch(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm2d(C, affine=True)
        
    def forward(self, x):
        # Normalize x first before calculating high-order norm
        x = self.instanceNorm(x)
        
        l2_norm = torch.sum(x ** 2, dim=(2,3),keepdim=True) ** 0.5 # B, 1, H, W
        
        return l2_norm # B, 1, H, W


#-----------------p-norm pooling--------------------
class Pnorm_ch(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
        
    def forward(self, x):
        p_norm = torch.sum(x ** self.p, dim=(2,3),keepdim=True) ** (1 / self.p) # B, 1, H, W
        
        return p_norm # B, 1, H, W


#----------------- Skew pooling --------------------
class GAPnSKP_ch(nn.Module):
    def __init__(self, channel, H, W):
        super().__init__()
        self.H, self.W = H, W
        
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        
        self.mixer = ChannelMixer()
    
    def forward(self, x):
        avg_out = self.avgpool(x) # B, C, 1, 1
        x = x - torch.mean(x, dim=(2,3),keepdim=True)
        skew_out = torch.sum(x ** 3, dim=(2,3),keepdim=True) / (self.H * self.W)
        
        channel_out = torch.cat( (avg_out, skew_out), dim=-1 ) # B, C, 1, 2
        
        # Mixup the two channels for fixed output sizes
        channel_out = self.mixer(channel_out)
        
        return channel_out # B, C, 1, 1

#-----------------2nd-order pooling--------------------
class Chdim_reduce(nn.Module):
    def __init__(self, C):
        super().__init__()
        
        self.adppool1D = nn.AdaptiveAvgPool1d(C)
    
    def forward(self, x):
        # X of shape B, C, H*W
        # Flip dimensions before going into 1D-adppool
        x = self.adppool1D(x.transpose(-1, -2)).transpose(-1, -2)
        
        return x

class BilinearPooling_ch(nn.Module):
    def __init__(self, channel, affine):
        super().__init__()
        
        self.C_out = int(channel // 4)
        
        self.dim_reduction = Chdim_reduce(self.C_out)
        self.relu = nn.ReLU(inplace=False)
        self.row_bn_for_channel = nn.BatchNorm2d(self.C_out)
        
        self.row_conv_group_for_channel = nn.Conv2d(
            self.C_out, self.C_out, kernel_size=(self.C_out, 1),
            groups=self.C_out, bias=False)
        self.fc_adapt_channels_for_channel = nn.Conv2d(
            self.C_out, channel, kernel_size=1, groups=1, bias=False) # Need this to align the output sizes
    
    def forward(self, x):
        B, C, _, _ = x.shape
        x = x.view(B, C, -1)
        x = self.dim_reduction(x) # B, C // 4, H, W
        
        # Calculate Covariance
        x = x - torch.mean(x, dim=2).unsqueeze(-1)
        
        x = x.matmul(torch.transpose(x, 1,2)) / (self.C_out - 1) # B, C // 4, C // 4
        
        x = x.unsqueeze(-1) # B, C // 4, C // 4, 1
        
        x = self.row_bn_for_channel(x)
        x = self.row_conv_group_for_channel(x) # B, C // 4, 1, 1
        x = self.relu(x)
        x = self.fc_adapt_channels_for_channel(x) # B, C, 1, 1
        
        return x # B, C, 1, 1

#-------------------------------------  Channel-wise Squeezing Implementation ENDED  ----------------------------------------

#-------------------------------------  Spatial-wise Squeezing  ----------------------------------------
#-----------------GAP--------------------
class GAP_sp(nn.Module):
    def forward(self, x):
        spatial_out = torch.mean(x,dim=1,keepdim=True)
        
        return spatial_out # B, 1, H, W


#-----------------GAP + GMP--------------------
class GAPnGMP_sp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mixer=nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True) # B, 1, H, W
        max_out,_ = torch.max(x,dim=1,keepdim=True) # B, 1, H, W
        
        spatial_out = torch.cat( (avg_out, max_out), dim=1 ) # B, 2, H, W
        
        # Mixup the two channels for fixed output sizes
        spatial_out = self.mixer(spatial_out)
        
        return spatial_out # B, 1, H, W


#----------------- GMP --------------------
class GMP_sp(nn.Module):
    def forward(self, x):
        max_out,_ = torch.max(x,dim=1,keepdim=True) # B, 1, H, W
        
        return max_out # B, 1, H, W

#----------------- STD --------------------
class STD_sp(nn.Module):
    def forward(self, x):
        std_out = torch.std(x, dim=1, keepdim=True) # B, 1, H, W
        
        return std_out # B, 1, H, W

#----------------- L2-norm --------------------
class L2norm_sp(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        self.layerNorm = nn.LayerNorm((H, W), elementwise_affine=True)
        
    def forward(self, x):
        # Normalize x first before calculating high-order norm
        x = self.layerNorm(x)
        p_norm = torch.sum(x ** 2, dim=1, keepdim=True) ** 0.5 # B, 1, H, W
        
        return p_norm # B, 1, H, W
        

#----------------- P-norm --------------------
class Pnorm_sp(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        p_norm = torch.sum(x ** self.p, dim=1, keepdim=True) ** (1 / self.p) # B, 1, H, W
        
        return p_norm # B, 1, H, W
    

#-----------------GAP + STD--------------------
class GAPnSTD_sp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.mixer=nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True) # B, 1, H, W
        std_out = torch.std(x,dim=1,keepdim=True) # B, 1, H, W
        
        spatial_out = torch.cat( (avg_out, std_out), dim=1 ) # B, 2, H, W
        
        # Mixup the two channels for fixed output sizes
        spatial_out = self.mixer(spatial_out)
        
        return spatial_out # B, 1, H, W


#----------------- Skew pooling --------------------
class GAPnSKP_sp(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        
        self.mixer=nn.Sequential(
            nn.Conv2d(2, 1, 1, bias=False),
            nn.ReLU(inplace=False),
        )
    
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True) # B, 1, H, W
        x = x - torch.mean(x, dim=1,keepdim=True) # B, 1, H, W
        skew_out = torch.sum(x ** 3, dim=1,keepdim=True) / self.C
        
        channel_out = torch.cat( (avg_out, skew_out), dim=1 ) # B, 2, H, W
        
        # Mixup the two channels for fixed output sizes
        channel_out = self.mixer(channel_out)
        
        return channel_out # # B, 1, H, W


#-----------------2nd-order pooling--------------------
class BilinearPooling_sp(nn.Module):
    def __init__(self, H, W, affine):
        super().__init__()
        
        self.adppool_1 = F.adaptive_avg_pool2d
        
        self.h = 8
        self.resolution = self.h * self.h
        
        self.relu = nn.ReLU(inplace=False)
        self.row_bn_for_spatial = nn.BatchNorm2d(self.resolution)
        
        self.row_conv_group_for_spatial = nn.Conv2d(
            self.resolution, self.resolution, kernel_size=(self.resolution, 1), 
            groups=self.resolution, bias=False)
        self.fc_adapt_channels_for_spatial = nn.Conv2d(
            self.resolution, self.resolution, kernel_size=1, groups=1, bias=False)
        
        self.adppool_2 = F.adaptive_avg_pool2d
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Reduce spatial dimensionality to 8x8
        x = self.adppool_1(x, (self.h, self.h)) # B, C, 8, 8
        
        # Calculate Covariance
        x = x.view(B, C, -1).transpose(1,2) # B, 8*8, C
        x = x - torch.mean(x, dim=2).unsqueeze(-1)
        
        x = x.matmul(torch.transpose(x, 1,2)) / (self.resolution - 1) # B, 8*8, 8*8
        
        # Restore spatial dimensionality
        x = x.unsqueeze(-1) # B, 8*8, 8*8, 1
        
        x = self.row_bn_for_spatial(x)
        x = self.row_conv_group_for_spatial(x) # Nx256x1x1
        x = self.relu(x)
        x = self.fc_adapt_channels_for_spatial(x) #Nx64x1x1
        x = x.view(x.shape[0], 1, self.h, self.h).contiguous()#Nx1x8x8
        
        x = self.adppool_2(x, (int(H), int(W)))
        x = x.view(B, 1, H, W)
        
        return x

#-------------------------------------  Spatial-wise Squeezing Implementation ENDED  ----------------------------------------

#-------------------------------------  spatial-wise Excitation  ----------------------------------------
class Affine_sp(nn.Module):
    def __init__(self, H, W):
        super().__init__()
        
        self.alpha = nn.Parameter( torch.ones((1, H, W), requires_grad=True) )
        self.beta = nn.Parameter( torch.zeros((1, H, W), requires_grad=True) )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid( self.alpha * x + self.beta )
        

#-------------------------------------  Channel-wise Excitation  ----------------------------------------
class Conv1D(nn.Module):
    def __init__(self, C, kernel_size):
        super().__init__()
        
        self.conv1D = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flip dimensions before going into 1D-conv
        x = self.conv1D(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        return x


class Conv1D_2layers(nn.Module):
    def __init__(self, C, kernel_size):
        super().__init__()
        
        self.conv1D_1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.ReLU(inplace=False)
        )
        self.conv1D_2 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flip dimensions before going into 1D-conv
        x = self.conv1D_1(x.squeeze(-1).transpose(-1, -2))
        x = self.conv1D_2(x).transpose(-1, -2).unsqueeze(-1)
        
        return x


class Conv1D_3layers(nn.Module):
    def __init__(self, C, kernel_size):
        super().__init__()
        
        self.conv1D_1 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.ReLU(inplace=False)
        )
        self.conv1D_2 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.ReLU(inplace=False)
        )
        self.conv1D_3 = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size, bias=False, padding=kernel_size//2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flip dimensions before going into 1D-conv
        x = self.conv1D_1(x.squeeze(-1).transpose(-1, -2))
        x = self.conv1D_2(x)
        x = self.conv1D_3(x).transpose(-1, -2).unsqueeze(-1)
        
        return x


class Affine_ch(nn.Module):
    def __init__(self, C):
        super().__init__()
        
        self.alpha = nn.Parameter( torch.ones((C, 1, 1), requires_grad=True) )
        self.beta = nn.Parameter( torch.zeros((C, 1, 1), requires_grad=True) )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid( self.alpha * x + self.beta )

#-------------------------------------  Channel-wise Excitation Implementation ENDED  ----------------------------------------


