from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingSignal(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(GatingSignal, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1x1(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.relu(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=2, stride=2, padding=0, bias=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g is the gating signal and  x is the input feature map
        theta_x = self.W_x(x) 
        phi_g = self.W_g(g)  
        upsample_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode='bilinear', align_corners=True)
        concat = upsample_g + theta_x
        act_xg = self.relu(concat)
        psi = self.sigmoid(self.psi(act_xg))
        upsample_psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        return x * upsample_psi

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernelsize=3, mid_ch=None):
        super(DoubleConv, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        
        padding = (kernelsize - 1) // 2  # padding dinámicamente
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=kernelsize, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(mid_ch, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=kernelsize, stride=1, padding=padding, bias=True),
            nn.BatchNorm2d(out_ch, momentum=0.99),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernelsize, dropout_rate):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            DoubleConv(in_ch, out_ch, kernelsize),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class AttentionUNet(nn.Module):
    def __init__(self, kernelsize, dropout_rate, first_filters=8, batchnorm=True):
        super(AttentionUNet, self).__init__()
        
        self.filters = [
            (2**0) * first_filters, 
            (2**1) * first_filters, 
            (2**2) * first_filters, 
            (2**3) * first_filters, 
            (2**4) * first_filters
        ]
        
        # Downsampling layers
        self.conv_block1 = ConvBlock(3, self.filters[0], kernelsize, dropout_rate)

        self.conv_block2 = ConvBlock(self.filters[0], self.filters[1], kernelsize, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        self.conv_block3 = ConvBlock(self.filters[1], self.filters[2], kernelsize, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.conv_block4 = ConvBlock(self.filters[2], self.filters[3], kernelsize, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # Upsampling layers
        self.gating_4 = GatingSignal(self.filters[3], self.filters[2], batchnorm=batchnorm)
        self.attention_4 = AttentionBlock(F_g=self.filters[2], F_l=self.filters[2], F_int=self.filters[2])
        self.upconv4 = ConvBlock(self.filters[3] + self.filters[2], self.filters[2], kernelsize, dropout_rate)

        self.gating_3 = GatingSignal(self.filters[2], self.filters[1], batchnorm=batchnorm)
        self.attention_3 = AttentionBlock(F_g=self.filters[1], F_l=self.filters[1], F_int=self.filters[1])
        self.upconv3 = ConvBlock(self.filters[2] + self.filters[1], self.filters[1], kernelsize, dropout_rate)
        
        self.gating_2 = GatingSignal(self.filters[1], self.filters[0], batchnorm=batchnorm)
        self.attention_2 = AttentionBlock(F_g=self.filters[0], F_l=self.filters[0], F_int=self.filters[0])
        self.upconv2 = ConvBlock(self.filters[1] + self.filters[0], self.filters[0], kernelsize, dropout_rate)
        
        self.final_conv = nn.Conv2d(self.filters[0], 1, kernel_size=1)
        self.final_bn = nn.BatchNorm2d(1)
        self.activation = nn.Identity()  # Linear activation

    def forward(self, x):
        # Downsampling
        dn_1 = self.conv_block1(x)
        pool_1 = self.pool1(dn_1)
        
        dn_2 = self.conv_block2(pool_1)
        pool_2 = self.pool2(dn_2)
        
        dn_3 = self.conv_block3(pool_2)
        pool_3 = self.pool3(dn_3)
        
        dn_4 = self.conv_block4(pool_3)

        gating_4 = self.gating_4(dn_4)
        att_4 = self.attention_4(gating_4, dn_3)

        up_4 = F.interpolate(dn_4, size=dn_3.shape[2:], mode='bilinear', align_corners=True)
        up_4 = torch.cat([up_4, att_4], dim=1)
        upconv_4 = self.upconv4(up_4)

        gating_3 = self.gating_3(upconv_4)
        att_3 = self.attention_3(gating_3, dn_2)
        up_3 = F.interpolate(upconv_4, size=dn_2.shape[2:], mode='bilinear', align_corners=True)
        up_3 = torch.cat([up_3, att_3], dim=1)
        upconv_3 = self.upconv3(up_3)
        
        gating_2 = self.gating_2(upconv_3)
        att_2 = self.attention_2(gating_2, dn_1)
        up_2 = F.interpolate(upconv_3, size=dn_1.shape[2:], mode='bilinear', align_corners=True)
        up_2 = torch.cat([up_2, att_2], dim=1)
        upconv_2 = self.upconv2(up_2)
        
        output = self.final_conv(upconv_2)
        return output