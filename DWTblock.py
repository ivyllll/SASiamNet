import torch
import torch.nn as nn

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class GCWTResDown(nn.Module):
    def __init__(self, in_channels, att_block, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dwt = DWT()
        if norm_layer:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                      norm_layer(in_channels),
                                      nn.PReLU(),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      norm_layer(in_channels),
                                      nn.PReLU())
        else:
            self.stem = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.PReLU(),
                                      nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.PReLU())
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv_down = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=2)
        #self.att = att_block(in_channels * 2, in_channels * 2)

    def forward(self, x):
        stem = self.stem(x)
        xLL, dwt = self.dwt(x)
        res = self.conv1x1(xLL)
        out = torch.cat([stem, res], dim=1)
        #out = self.att(out)
        return out, dwt

class GCIWTResUp(nn.Module):
    def __init__(self, in_channels, att_block, norm_layer=None):
        super().__init__()
        if norm_layer:
            self.stem = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                norm_layer(in_channels // 4),
                nn.PReLU(),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                norm_layer(in_channels // 4),
                nn.PReLU(),
            )
        else:
            self.stem = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                nn.PReLU(),
                nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
                nn.PReLU(),
            )
        self.pre_conv_stem = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, padding=0)
        self.pre_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        # self.prelu = nn.PReLU()
        self.post_conv = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1, padding=0)
        self.iwt = IWT()
        self.last_conv = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, padding=0)
        # self.se = SE_net(in_channels // 2, in_channels // 4)
    def forward(self, x, x_dwt):
        x = self.pre_conv_stem(x)
        stem = self.stem(x)
        x_dwt = self.pre_conv(x_dwt)
        x_iwt = self.iwt(x_dwt)
        x_iwt = self.post_conv(x_iwt)
        out = torch.cat((stem, x_iwt), dim=1)
        out = self.last_conv(out)
        return out

