from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleConv, self).__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride = 1, padding=1, bias=True),
            nn.BatchNorm2d(mid_ch, momentum=0.99 ),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride= 1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=0.99 ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch,dropout_rate):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            DoubleConv(in_ch, out_ch),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class down_block(nn.Module):
    def __init__(self, in_ch, out_ch,dropout_rate):
        super(down_block, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch,dropout_rate):
        super(up_block, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=3, stride=2),
            nn.BatchNorm2d(in_ch // 2, momentum=0.99 )
        )
        self.conv = nn.Sequential(
            DoubleConv(in_ch,out_ch),
            nn.Dropout2d(p=dropout_rate)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class U_Net(nn.Module):
 
    def __init__(self, in_ch=3, out_ch=1, n1=29, dropout_rate=0.089735):
        super(U_Net, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        self.In = conv_block(in_ch, filters[0],dropout_rate)

        self.Down1 = down_block(filters[0], filters[1],dropout_rate)
        self.Down2 = down_block(filters[1], filters[2],dropout_rate)
        self.Down3 = down_block(filters[2], filters[3],dropout_rate)

        self.Up3 = up_block(filters[3], filters[2],dropout_rate)
        self.Up2 = up_block(filters[2], filters[1],dropout_rate)
        self.Up1 = up_block(filters[1], filters[0],dropout_rate)

        self.Out = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x1 = self.In(x)

        e1 = self.Down1(x1)
        e2 = self.Down2(e1)
        e3 = self.Down3(e2)

        d = self.Up3(e3,e2)
        d = self.Up2(d,e1)
        d = self.Up1(d,x1)

        out = self.Out(d)

        return out