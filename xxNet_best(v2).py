import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_ import resnet18

class TemporalFeatureInteractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureInteractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1_ca = self.ca(x1)
        # x2_ca = self.ca(x2)
        # difference enhance
        x_sub = self.conv_sub(torch.abs(x1 - x2))     # 将得到的t1和t2时刻的特征图进行做差,得到差异图,然后把差异图送入3x3
        x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)  # 分别与T1和T2的特征进行相乘,之后再与T1和T2进行相加,两个分支分别再经过一个3x3 conv
        # x1 = x1_ca * x1
        x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
        # x2 = x2_ca * x2
        # fusion
        x_f = torch.cat([x1, x2], dim=1)              # 拼接,进行通道变化
        x_f = self.conv_cat(x_f)                      # 3x3 conv
        x = x_sub + x_f                               # 与差异特征进行相加
        x = self.conv_dr(x)                           # 经过1x1的卷积来减少通道维度
        return x                                      # 最后得到输出特征

class Differentiation_path(torch.nn.Module):
    """Differece Network"""
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )
        self.init_weight()

    def forward(self, input):
        x = self.convblock1(input)
        x = self.convblock2(x)
        x = self.convblock3(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class TFIM(nn.Module):
    def __init__(self,in_d, out_d):
        super(TFIM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.differentiation_path = Differentiation_path(in_d)
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
        # dx1 = self.differentiation_path(x1)
        # dx2 = self.differentiation_path(x2)
        dx1 = x1
        dx2 = x2
        dx = torch.abs(dx1 - dx2)
        x_sub = self.conv_sub(dx)
        ex1 = self.conv_enh1(dx1)
        ex2 = self.conv_enh2(dx2)
        x1_mul = ex1.mul(x1)
        x2_mul = ex2.mul(x2)
        x1_add = ex1 + x1_mul
        x2_add = ex2 + x2_mul
        x_mul = self.conv_mul(ex1 * ex2)
        x_add = x_sub + x_mul
        # x_cat = torch.cat([x1_add,x2_add,x_add], dim=1)
        x_cat = self.conv_cat(torch.cat([x1_add,x2_add,x_add], dim=1))
        x = self.conv_dr(x_cat)
        return x

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

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # self.fc1 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(in_planes, in_planes // ratio, bias=False)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.ca = ChannelAttention(self.in_d * 4, ratio=16)
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

    def forward(self, d5, d4, d3, d2):
        # upsampling
        # print(d5.shape)
        # print(d4.shape)
        # print(d3.shape)
        # print(d2.shape)
        # print(d2.size()[2:])
        d5 = F.interpolate(d5, d2.size()[2:], mode='bilinear', align_corners=True) #插值法进行上采样,改变特征图size,使它们大小相同
        d4 = F.interpolate(d4, d2.size()[2:], mode='bilinear', align_corners=True) #interpolate(input(Tensor)：需要进行采样处理的数组, size(int或序列)：输出空间的大小)
        d3 = F.interpolate(d3, d2.size()[2:], mode='bilinear', align_corners=True) #这里d2.size()[2:]只改变后两个维度的大小(也就是图片大小)
        # print(d5.shape)
        # print(d4.shape)
        # print(d3.shape)
        # print(d2.shape)
        # fusion
        x = torch.cat([d5, d4, d3, d2], dim=1)


        x_eca = self.eca(x)
        x = self.conv_dr(x_eca)
        # # print(x.shape) #fusion后通道数相加所以要x4
        # x_ca = self.ca(x)
        # # print(x_ca.shape)  #通道注意力机制将图像压缩成1xCX1X1
        # x = x * x_ca
        # x = self.conv_dr(x)
        # print(x.shape)  #3x3卷积把通道恢复成fusion之前的in_d
        # feature = x[0:1, 0:64, 0:64, 0:64]
        # vis.visulize_features(feature)

        # pooling
        # 使用自适应平均池化来进行特征复原  池化操作是为了在降低像素的同时保存重要信息
        d2 = x
        d3 = self.conv_pool1(x)
        d4 = self.conv_pool2(x)
        d5 = self.conv_pool3(x)
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
        self.ca = ChannelAttention(self.in_d, ratio=16)
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
        # self.TFIM5 = TemporalFeatureInteractionModule(512, self.mid_d)
        # self.TFIM4 = TemporalFeatureInteractionModule(256, self.mid_d)
        # self.TFIM3 = TemporalFeatureInteractionModule(128, self.mid_d)
        # self.TFIM2 = TemporalFeatureInteractionModule(64, self.mid_d)
        self.TFIM5 = TFIM(512, self.mid_d)
        self.TFIM4 = TFIM(256, self.mid_d)
        self.TFIM3 = TFIM(128, self.mid_d)
        self.TFIM2 = TFIM(64, self.mid_d)

        self.CIEM1 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM1 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.CIEM2 = ChangeInformationExtractionModule(self.mid_d, output_nc)
        self.GRM2 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.decoder = Decoder(self.mid_d, output_nc)


    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone.base_forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone.base_forward(x2)

        # feature difference
        d5 = self.TFIM5(x1_5, x2_5)  # 1/32
        d4 = self.TFIM4(x1_4, x2_4)  # 1/16
        d3 = self.TFIM3(x1_3, x2_3)  # 1/8
        d2 = self.TFIM2(x1_2, x2_2)  # 1/4

        # change information guided refinement 1
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # change information guided refinement 2
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # decoder
        mask = self.decoder(d5, d4, d3, d2)
        mask = F.interpolate(mask, x1.size()[2:], mode='bilinear', align_corners=True)
        mask = torch.sigmoid(mask)

        return mask

if __name__ == "__main__":
    # Differentiation Path
    # x = torch.randn(10, 64, 128, 128)
    # y = torch.randn(10, 64, 128, 128)
    # net1 = Differentiation_path(64)
    # x1 = net1(x)
    # print(x1.shape)

    # TFIM
    # x = torch.randn(10, 128, 64, 64) #(1,128,16,16)original
    # y = torch.randn(10, 128, 64, 64)
    # # net1 = TemporalFeatureInteractionModule(128,64)
    # net1 = TFIM(128, 64)
    # x1 = net1(x, y) # torch.Size([1, 64, 16, 16]) torch.Size([10, 64, 64, 64])
    # print(x1.shape)

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
    x = torch.randn(10, 3, 512, 512)
    y = torch.randn(10, 3, 512, 512)
    net1 = BaseNet(6, 2)
    x1 = net1(x, y) # torch.Size([10, 2, 512, 512])
    print(x1.shape)


