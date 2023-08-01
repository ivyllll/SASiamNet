import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_ import resnet18


class Differentiation_path(torch.nn.Module):
    """Difference Network"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.init_weight()

    def forward(self, input):
        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class DifferentiationModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.rrg = RRG(nn.Conv2d, in_channels, 1, 16, nn.ReLU(True), 1)
        self.dp = Differentiation_path(in_channels, out_channels)
    def forward(self, x1, x2):
        dx1 = self.dp(self.rrg(x1))
        dx2 = self.dp(self.rrg(x2))
        dx = torch.abs(dx1 - dx2)
        return dx

class Spatial_path(torch.nn.Module):
    """Assimilation Network"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.init_weight()

    def forward(self, input):
        x = self.conv_block1(input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class AssimilationModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.rrg = RRG(nn.Conv2d, in_channels, 1, 16, nn.ReLU(True), 1)
        self.sp = Spatial_path(in_channels, out_channels)
    def forward(self, x1, x2):
        ax1 = self.sp(self.rrg(x1))
        ax2 = self.sp(self.rrg(x2))
        ax = ax1 + ax2
        return ax

"""Channel attention"""
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
"""Spatial attention"""
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale
"""DAB: Double attention block"""
class DAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)

        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res
"""RRG: Recursive residual group"""
class RRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act,  num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act)
            for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res



class RFEM(nn.Module):
    # Time-based Feature Fusion Module
    def __init__(self,in_d, out_d):
        super(RFEM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        # self.sab = SpatialAttentionBlock(spatial_filter=self.in_d)
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_mul = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 3, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # dx1 = self.sab(x1)
        # dx2 = self.sab(x2)
        dx1 = x1
        dx2 = x2
        x_sub = self.conv_sub(torch.abs(dx1 - dx2))
        ex1 = self.conv_enh1(dx1)
        ex2 = self.conv_enh2(dx2)
        x1_mul = ex1.mul(x1)
        x2_mul = ex2.mul(x2)
        x1_add = ex1 + x1_mul
        x2_add = ex2 + x2_mul
        x_mul = self.conv_mul(ex1 * ex2)
        x_add = x_sub + x_mul
        x_cat = self.conv_cat(torch.cat([x1_add,x2_add,x_add], dim=1))
        x = self.conv_dr(x_cat)
        return x

class PrimaryFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_refinement = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_concatenate = nn.Sequential(
            nn.Conv2d(self.in_d*2, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
    def forward(self, dm5, dm4, dm3, dm2,
                d5, d4, d3, d2,
                am5, am4, am3, am2):
        r1_5 = self.conv_refinement(dm5 + d5)
        r1_4 = self.conv_refinement(dm4 + d4)
        r1_3 = self.conv_refinement(dm3 + d3)
        r1_2 = self.conv_refinement(dm2 + d2)

        r2_5 = self.conv_refinement(d5 + am5)
        r2_4 = self.conv_refinement(d4 + am4)
        r2_3 = self.conv_refinement(d3 + am3)
        r2_2 = self.conv_refinement(d2 + am2)

        r5 = self.conv_concatenate(torch.cat([r1_5, r2_5], dim=1))
        r4 = self.conv_concatenate(torch.cat([r1_4, r2_4], dim=1))
        r3 = self.conv_concatenate(torch.cat([r1_3, r2_3], dim=1))
        r2 = self.conv_concatenate(torch.cat([r1_2, r2_2], dim=1))
        return r5, r4, r3, r2

class MFAM(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_add = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_multi = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, dm5, dm4, dm3, dm2,
                r5, r4, r3, r2,
                am5, am4, am3, am2):
        add5 = self.conv_add(r5 + am5)
        add4 = self.conv_add(r4 + am4)
        add3 = self.conv_add(r3 + am3)
        add2 = self.conv_add(r2 + am2)
        mul5 = self.conv_multi(add5 * dm5)
        mul4 = self.conv_multi(add4 * dm4)
        mul3 = self.conv_multi(add3 * dm3)
        mul2 = self.conv_multi(add2 * dm2)
        # mul5 = F.interpolate(mul5, mul2.size()[2:], mode='bilinear', align_corners=True)  # 插值法进行上采样,改变特征图size,使它们大小相同
        # mul4 = F.interpolate(mul4, mul2.size()[2:], mode='bilinear', align_corners=True)
        # mul3 = F.interpolate(mul3, mul2.size()[2:], mode='bilinear', align_corners=True)
        # cat = self.conv_concatenate(torch.cat([mul2, mul3, mul4, mul5], dim = 1))

        return mul5, mul4, mul3, mul2

class Eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

class CIEM(nn.Module):
    def __init__(self, in_d, out_d):
        super(CIEM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.eca = Eca_layer(self.in_d * 4, k_size=3)
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d * 4, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]), # kernel_size:池化窗口大小 stride:池化窗口移动步长
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(self.in_d),
            # nn.ReLU(inplace=True),
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(self.in_d),
            # nn.ReLU(inplace=True),
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(self.in_d),
            # nn.ReLU(inplace=True),
        )
        self.conv_concatenate = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
    def forward(self, r5, r4, r3, r2, r5_, r4_, r3_, r2_):
        # upsampling
        r5 = F.interpolate(r5, r2.size()[2:], mode='bilinear', align_corners=True)  # 插值法进行上采样,改变特征图size,使它们大小相同
        r4 = F.interpolate(r4, r2.size()[2:], mode='bilinear', align_corners=True)
        r3 = F.interpolate(r3, r2.size()[2:], mode='bilinear', align_corners=True)
        r2 = r2

        # fusion
        x = torch.cat([r5, r4, r3, r2], dim=1)
        x_eca = self.eca(x)
        x = self.conv_dr(x_eca)

        # pooling
        # 使用自适应平均池化来进行特征复原  池化操作是为了在降低像素的同时保存重要信息
        d2 = x
        d3 = self.conv_pool1(x)
        d4 = self.conv_pool2(x)
        d5 = self.conv_pool3(x)
        d2 = self.conv_concatenate(torch.cat([d2, r2_], dim=1))
        d3 = self.conv_concatenate(torch.cat([d3, r3_], dim=1))
        d4 = self.conv_concatenate(torch.cat([d4, r4_], dim=1))
        d5 = self.conv_concatenate(torch.cat([d5, r5_], dim=1))
        return d5, d4, d3, d2

class GuidedRefinementModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(GuidedRefinementModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_d5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d4 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p):
        # feature refinement
        d5 = self.conv_d5(d5_p + d5)
        d4 = self.conv_d4(d4_p + d4)
        d3 = self.conv_d3(d3_p + d3)
        d2 = self.conv_d2(d2_p + d2)

        return d5, d4, d3, d2

class Decoder(nn.Module):
    #可以尝试加channel attention来达到fusion的效果(另建一个module)
    def __init__(self, in_d, out_d):
        super(Decoder, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.eca = Eca_layer(self.in_d, k_size=3)

        self.conv_sum1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_sum3 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        self.cls = nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=False)

    def forward(self, d5, d4, d3, d2):

        d5 = F.interpolate(d5, d4.size()[2:], mode='bilinear', align_corners=True)  # 要把d5的HW变成和d4一样大小才能conv_sum(也就是fusion)
        d5 = self.eca(d5)
        # d5 = self.ca(d5)
        # d5 = d5 * self.ca(d5)
        d4 = self.conv_sum1(d4 + d5)

        d4 = F.interpolate(d4, d3.size()[2:], mode='bilinear', align_corners=True)  # 要把d4的HW变成和d3一样大小才能conv_sum(也就是fusion)
        d4 = self.eca(d4)
        # d4 = self.ca(d4)
        # d4 = d4 * self.ca(d4)
        d3 = self.conv_sum2(d3 + d4)

        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True)  # 要把d3的HW变成和d2一样大小才能conv_sum(也就是fusion)
        d3 = self.eca(d3)
        # d3 = self.ca(d3)
        # d3 = d3 * self.ca(d3)
        d2 = self.conv_sum3(d2 + d3)

        mask = self.cls(d2)

        return mask

class BaseNet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(BaseNet, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.mid_d = 64
        self.RFEM5 = RFEM(512, self.mid_d)
        self.RFEM4 = RFEM(256, self.mid_d)
        self.RFEM3 = RFEM(128, self.mid_d)
        self.RFEM2 = RFEM(64, self.mid_d)

        self.DM5 = DifferentiationModule(512, self.mid_d)
        self.DM4 = DifferentiationModule(256, self.mid_d)
        self.DM3 = DifferentiationModule(128, self.mid_d)
        self.DM2 = DifferentiationModule(64, self.mid_d)
        self.AM5 = AssimilationModule(512, self.mid_d)
        self.AM4 = AssimilationModule(256, self.mid_d)
        self.AM3 = AssimilationModule(128, self.mid_d)
        self.AM2 = AssimilationModule(64, self.mid_d)
        self.PFFM = PrimaryFeatureFusionModule(64, self.mid_d)
        self.MFAM = MFAM(64, self.mid_d)

        self.CIEM1 = CIEM(self.mid_d, self.mid_d)
        self.GRM1 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.CIEM2 = CIEM(self.mid_d, self.mid_d)
        self.GRM2 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.decoder = Decoder(self.mid_d, output_nc)


    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        dm5 = self.DM5(x1_5, x2_5)
        dm4 = self.DM4(x1_4, x2_4)
        dm3 = self.DM3(x1_3, x2_3)
        dm2 = self.DM2(x1_2, x2_2)
        am5 = self.AM5(x1_5, x2_5)
        am4 = self.AM4(x1_4, x2_4)
        am3 = self.AM3(x1_3, x2_3)
        am2 = self.AM2(x1_2, x2_2)

        # feature difference
        d5 = self.RFEM5(x1_5, x2_5)  # 1/32
        d4 = self.RFEM4(x1_4, x2_4)  # 1/16
        d3 = self.RFEM3(x1_3, x2_3)  # 1/8
        d2 = self.RFEM2(x1_2, x2_2)  # 1/4

        r5, r4, r3, r2 = self.PFFM(dm5, dm4, dm3, dm2,
                                  d5, d4, d3, d2,
                                  am5, am4, am3, am2)
        r5_, r4_, r3_, r2_ = self.MFAM(dm5, dm4, dm3, dm2,
                                   r5, r4, r3, r2,
                                   am5, am4, am3, am2)

        # change information guided refinement 1
        # d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(r5, r4, r3, r2, r5_, r4_, r3_, r2_) #加了PFFM
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # change information guided refinement 2
        # d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(r5, r4, r3, r2, r5_, r4_, r3_, r2_) #加了PFFM
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # decoder
        mask = self.decoder(d5, d4, d3, d2)
        # mask = self.decoder(d5, d4, d3, d2)
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        return mask

if __name__ == "__main__":
    # RRG
    # x = torch.randn(1, 64, 128, 128)
    # net1 = RRG(nn.Conv2d, 64, 1, 16, nn.ReLU(True), 1)
    # x1 = net1(x)
    # print(x1.shape)

    # Differentiation Path
    # x = torch.randn(1, 64, 128, 128)
    # net1 = Differentiation_path(64, 32)
    # x1 = net1(x)
    # print(x1.shape)

    # Differentiation module
    # x = torch.randn(1, 64, 128, 128)
    # y = torch.randn(1, 64, 128, 128)
    # net1 = DifferentiationModule(64)
    # x1 = net1(x, y)
    # print(x1.shape)

    # Spatial path
    # x = torch.randn(1, 64, 128, 128)
    # net1 = SpatialAttentionBlock(64)
    # x1 = net1(x)
    # print(x1.shape)

    # RFEM
    # x = torch.randn(10, 128, 64, 64) #(1,128,16,16)original
    # y = torch.randn(10, 128, 64, 64)
    # # net1 = TemporalFeatureInteractionModule(128,64)
    # net1 = RFEM(128, 64)
    # x1 = net1(x, y) # torch.Size([1, 64, 16, 16]) torch.Size([10, 64, 64, 64])
    # print(x1.shape)

    # MFAM

    # CIEM
    # # x = torch.randn(1, 4, 2, 2)
    # # y = torch.randn(1, 4, 4, 4)
    # # z = torch.randn(1, 4, 8, 8)
    # # w = torch.randn(1, 4, 16, 16)
    # x = torch.randn(10, 64, 128, 128)
    # y = torch.randn(10, 64, 64, 64)
    # z = torch.randn(10, 64, 32, 32)
    # w = torch.randn(10, 64, 16, 16)
    # net1 = ChangeInformationExtractionModule(64, 2) # 其实第二个参数不起任何作用，不管怎么变，输出的通道数都是第一个参数（输入通道数）
    # d5, d4, d3, d2 = net1(x, y, z, w)
    # # torch.Size([1, 4, 2, 2]) torch.Size([1, 4, 4, 4]) torch.Size([1, 4, 8, 8]) torch.Size([1, 4, 16, 16])
    # # torch.Size([10, 64, 2, 2]) torch.Size([10, 64, 4, 4]) torch.Size([10, 64, 8, 8]) torch.Size([10, 64, 16, 16])
    # print(d5.shape, d4.shape, d3.shape, d2.shape)

    # GRM
    # x = torch.randn(1, 4, 2, 2)
    # y = torch.randn(1, 4, 4, 4)
    # z = torch.randn(1, 4, 8, 8)
    # w = torch.randn(1, 4, 16, 16)
    # x_p = torch.randn(1, 4, 2, 2)
    # y_p = torch.randn(1, 4, 4, 4)
    # z_p = torch.randn(1, 4, 8, 8)
    # w_p = torch.randn(1, 4, 16, 16)
    # net1 = GuidedRefinementModule(4, 2)
    # d5, d4, d3, d2 = net1(x, y, z, w, x_p, y_p, z_p, w_p)
    # # torch.Size([1, 2, 2, 2]) torch.Size([1, 2, 4, 4]) torch.Size([1, 2, 8, 8]) torch.Size([1, 2, 16, 16])
    # print(d5.shape, d4.shape, d3.shape, d2.shape)

    # Decoder
    # d5 = torch.randn(1, 64, 2, 2)
    # d4 = torch.randn(1, 64, 4, 4)
    # d3 = torch.randn(1, 64, 8, 8)
    # d2 = torch.randn(1, 64, 16, 16)
    # net1 = Decoder(64,2)
    # decoder = net1(d5, d4, d3, d2)
    # print(decoder.shape) # torch.Size([1, 2, 16, 16]) decoder就是上采样还原图片尺寸的过程 d2的HW就是原尺寸而且不一定是相关倍数（4/8/16/32）

    # Base
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 3, 512, 512)
    net1 = BaseNet(6, 2)
    x1 = net1(x, y) # torch.Size([10, 2, 512, 512])
    print(x1.shape)


