from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# # from jupyter_core.paths import pjoin
# from numpy.ma import copy
# from torch import einsum
# from torch.nn import LayerNorm, Dropout
# from torch.nn.qat import Conv2d


# from resnet_ import resnet18
from torchvision.models.resnet import resnet18,resnet34

class Backbone_resnet(nn.Module):
    def __init__(self,backbone):
        super(Backbone_resnet, self).__init__()

        if backbone == 'resnet18':
            self.net = resnet18(pretrained=True)
            del self.net.avgpool
            del self.net.fc
        elif backbone == 'resnet34':
            self.net = resnet34(pretrained=True)
            del self.net.avgpool
            del self.net.fc
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def forward(self,x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        c1 = self.net.layer1(x)
        c2 = self.net.layer2(c1)
        c3 = self.net.layer3(c2)
        c4 = self.net.layer4(c3)
        return c1, c2, c3, c4

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

class TFFM(nn.Module):
    # Time-based Feature Fusion Module
    def __init__(self,in_d, out_d):
        super(TFFM, self).__init__()
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

###########################################################################################
def to_2tuple(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)#(224,224)
        # print("x10",img_size)
        patch_size = to_2tuple(patch_size)#(16,16)
        # print("x11", patch_size)

        self.img_size = img_size

        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  #grid_size=224÷16=14

        # print("x9", self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]  #num_patches=14*14
        # print("x8",self.num_patches)

        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #proj使用卷积，embed_dimension这一参数在vision transformer的base16模型用到的是768，所以默认是768。但是如果是large或者huge模型的话embed_dim也会变。
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        #norm_layer默认是None，就是进行nn.Identity()也就是不做任何操作；如果有传入（非None），则会进行初始化一个norm_layer。

    def forward(self, x):
        B, C, H, W = x.shape
        # print("x1",x.shape)
        # print("x2", self.img_size)
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #     #assert：进行判断，如果代码模型定义和实际输入尺寸不同则会报错
        x = self.proj(x)  #用卷积实现序列化 torch.Size([4, 768, 14, 14])
        # print("x3",x.shape)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            #flatten(2)操作实现了[B,C,H,W,]->[B,C,HW]，指从维度2开始进行展平
            #transpose(1,2)操作实现了[B,C,HW]->[B,HW,C]
            #torch.Size([4, 196, 768])
        x = self.norm(x)
        #通过norm层输出
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,             # 输入token的dim
                 num_heads=8,     # 多头注意力中head的个数
                 qkv_bias=False,  # 在生成qkv时是否使用偏置，默认否
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads  # 计算每一个head需要传入的dim  #96

        self.scale = head_dim ** -0.5  # head_dim的-0.5次方，即1/根号d_k，即理论公式里的分母根号d_k
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # qkv是通过1个全连接层参数为dim和3dim进行初始化的，也可以使用3个全连接层参数为dim和dim进行初始化，二者没有区别，
        self.attn_drop = nn.Dropout(attn_drop)  # 定义dp层 比率attn_drop
        self.proj = nn.Linear(dim, dim)  # 再定义一个全连接层，是 将每一个head的结果进行拼接的时候乘的那个矩阵W^O
        self.proj_drop = nn.Dropout(proj_drop)  # 定义dp层 比率proj_drop

    def forward(self, x):  # 正向传播过程
        # 输入是[batch_size,
        #      num_patches+1, （base16模型的这个数是14*14）
        #      total_embed_dim（base16模型的这个数是768   ）]
        B, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv->[batchsize, num_patches+1, 3*total_embed_dim]                      torch.Size([4, 197, 2304])
        # reshape->[batchsize, num_patches+1, 3, num_heads, embed_dim_per_head]   torch.Size([4, 197, 3, 8, 96])
        # permute->[3, batchsize, num_heads, num_patches+1, embed_dim_per_head]   torch.Size([3, 4, 8, 197, 96])
        q, k, v = qkv[0], qkv[1], qkv[2]  #torch.Size([4, 8, 197, 96])
        # make torchscript happy (cannot use tensor as tuple)
        # q、k、v大小均[batchsize, num_heads, num_patches+1, embed_dim_per_head]
        attn = (q @ k.transpose(-2, -1)) * self.scale#torch.Size([4, 8, 197, 197])
        # print(attn.shape)
        # 现在的操作都是对每个head进行操作
        # transpose是转置最后2个维度，@就是矩阵乘法的意思
        # q  [batchsize, num_heads, num_patches+1, embed_dim_per_head]
        # k^T[batchsize, num_heads, embed_dim_per_head, num_patches+1]
        # q*k^T=[batchsize, num_heads, num_patches+1, num_patches+1]
        # self.scale=head_dim的-0.5次方
        # 至此完成了(Q*K^T)/根号d_k的操作
        attn = attn.softmax(dim=-1)
        # dim=-1表示在得到的结果的每一行上进行softmax处理，-1就是最后1个维度
        # 至此完成了softmax[(Q*K^T)/根号d_k]的操作
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)#torch.Size([4, 197, 768])

        # @->[batchsize, num_heads, num_patches+1, embed_dim_per_head]
        # 这一步矩阵乘积就是加权求和
        # transpose->[batchsize, num_patches+1, num_heads, embed_dim_per_head]
        # reshape->[batchsize, num_patches+1, num_heads*embed_dim_per_head]即[batchsize, num_patches+1, total_embed_dim]
        # reshape实际上就实现了concat拼接
        x = self.proj(x)
        # 将上一步concat的结果通过1个线性映射，通常叫做W，此处用全连接层实现
        x = self.proj_drop(x)
        # dropout
        # 至此完成了softmax[(Q*K^T)/根号d_k]*V的操作
        # 一个head的attention的全部操作就实现了
        return x

class FeedForward(nn.Module):
#全连接层1+GELU+dropout+全连接层2+dropout
#全连接层1的输出节点个数是输入节点个数的4倍，即mlp_ratio=4.
#全连接层2的输入节点个数是输出节点个数的1/4
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 #多头注意力模块中的最后的全连接层之后的dropout层对应的drop比率
                 attn_drop=0.,
                 #多头注意力模块中softmax[Q*K^T/根号d_k]之后的dropout层的drop比率
                 drop_path=0.,
                 #本代码用到的是DropPath方法（上面右图的DropPath），所以上面右图的两个droppath层有这个比率
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #第一层LN
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        #第一个多头注意力
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #如果传入的drop_path大于0，就会实例化一个droppath方法；如果传入的drop_path等于0，则执行Identity()不做任何操作
        self.norm2 = norm_layer(dim)
        #第二个LN层
        mlp_hidden_dim = int(dim * mlp_ratio)
        #mlp层的隐层个数是输入的4倍，实例化一个MLP模块的时候需要传入mlp_hidden_dim这个参数，所以在此提前计算
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp = FeedForward(dim=dim,hidden_dim=mlp_hidden_dim)
        #act_layer是激活函数

    def forward(self, x):
    #前向传播过程：
    #第一部分：LN+Mul-Head-Attention+ Dropout之后，加上第一个LN之前的输入
    #第二部分：LN+MLP+Dropout之后，加上第二个LN之前的输入
    #输出x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CIEM(nn.Module):
    def __init__(self,img_size=128,
                 patch_size=16,
                 # in_chans=64,
                 in_chans=256,  ##########################################
                 out_chans=64,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 representation_size=None,
                 # representation_size是最后的MLP Head中的pre-logits中的全连接层的节点个数，默认为None，此时就不会去构建这个pre-logits，
                 # 也就是此时在MLP Head中只有一个全连接层，而没有pre-logits层。【pre-logits层是什么：就是全连接层+激活函数】
                 distilled=False,
                 # distilled后续的DeiT才用到这个参数
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 embed_layer=PatchEmbed,
                 # 这个参数是nn.Module类型，即模块PatchEmbed
                 norm_layer=None,
                 # 这个参数也是nn.Module类型
                 act_layer=None,
                 # weight_init=''
                 ):
        super().__init__()
        self.in_d = in_chans
        self.out_d = out_chans

        self.pools_sizes = [2, 4, 8]
        self.num_features = self.embed_dim = embed_dim  # 复制参数
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # 对图片进行patch和embed

        num_patches = self.patch_embed.num_patches
        # print("x7",num_patches)
        # print("patch_size",patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        print("x6",self.pos_embed.shape)  # 1, 1024, 768

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.conv_pool1 = nn.Sequential(
            # torch.reshape(input, shape=(, -1, self.h * self.w)).permute(0, 2, 1)
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),
            # kernel_size:池化窗口大小 stride:池化窗口移动步长
            # nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),
            # nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),
            # nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        )
    def forward(self, r2_, r3_, r4_, r5_):
        r5_ = F.interpolate(r5_, r2_.size()[2:], mode='bilinear', align_corners=True)  # 插值法进行上采样,改变特征图size,使它们大小相同
        r4_ = F.interpolate(r4_, r2_.size()[2:], mode='bilinear', align_corners=True)
        r3_ = F.interpolate(r3_, r2_.size()[2:], mode='bilinear', align_corners=True)
        r2_ = r2_

        # fusion
        x = torch.cat([r5_, r4_, r3_, r2_], dim=1)
        print("x_cat",x.shape)  # 4, 256, 128, 128

        # transformer
        x = self.patch_embed(x)   #############################################
        print("x4",x.shape)  # 4, 1024, 768
        # print("x5",self.pos_embed.shape)  # 1, 1024, 768

        x = self.pos_drop(x + self.pos_embed)
        # x = self.pos_drop(x)
        x = self.norm(self.blocks(x))
        print("x13",x.shape)  # 4, 1024, 768

        # pooling
        # 使用自适应平均池化来进行特征复原  池化操作是为了在降低像素的同时保存重要信息
        x = rearrange(x, 'b (l w) n -> b n l w', l=32, w=32)

        d2 = x
        print("d2",d2.shape)
        d3 = self.conv_pool1(x)
        print("d3", d2.shape)
        d4 = self.conv_pool2(x)
        print("d4", d2.shape)
        d5 = self.conv_pool3(x)
        print("d5", d2.shape)
        return d5, d4, d3, d2

class CIEM_original(nn.Module):
    def __init__(self, in_d, out_d):
        super().__init__()
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
        self.backbone = Backbone_resnet(backbone='resnet18')
        self.mid_d = 64
        # self.TFFM5 = TemporalFeatureInteractionModule(512, self.mid_d)
        # self.TFFM4 = TemporalFeatureInteractionModule(256, self.mid_d)
        # self.TFFM3 = TemporalFeatureInteractionModule(128, self.mid_d)
        # self.TFFM2 = TemporalFeatureInteractionModule(64, self.mid_d)
        self.TFFM5 = TFFM(512, self.mid_d)
        self.TFFM4 = TFFM(256, self.mid_d)
        self.TFFM3 = TFFM(128, self.mid_d)
        self.TFFM2 = TFFM(64, self.mid_d)

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

        self.CIEM1 = CIEM_original(128, self.mid_d)
        self.GRM1 = GuidedRefinementModule(self.mid_d, self.mid_d)

        self.CIEM2 = CIEM_original(128, self.mid_d)
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
        d5 = self.TFFM5(x1_5, x2_5)  # 1/32
        d4 = self.TFFM4(x1_4, x2_4)  # 1/16
        d3 = self.TFFM3(x1_3, x2_3)  # 1/8
        d2 = self.TFFM2(x1_2, x2_2)  # 1/4

        r5, r4, r3, r2 = self.PFFM(dm5, dm4, dm3, dm2,
                                  d5, d4, d3, d2,
                                  am5, am4, am3, am2)

        r5_, r4_, r3_, r2_ = self.MFAM(dm5, dm4, dm3, dm2,
                                       r5, r4, r3, r2,
                                       am5, am4, am3, am2)

        # change information guided refinement 1
        # d5_p, d4_p, d3_p, d2_p = self.CIEM1(d5, d4, d3, d2)
        d5_p, d4_p, d3_p, d2_p = self.CIEM1(r5_, r4_, r3_, r2_) #加了PFFM
        d5, d4, d3, d2 = self.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # change information guided refinement 2
        # d5_p, d4_p, d3_p, d2_p = self.CIEM2(d5, d4, d3, d2)
        d5_p, d4_p, d3_p, d2_p = self.CIEM2(r5, r4, r3, r2) #加了PFFM
        d5, d4, d3, d2 = self.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

        # decoder
        mask = self.decoder(d5, d4, d3, d2)
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

    # TFFM
    # x = torch.randn(10, 128, 64, 64) #(1,128,16,16)original
    # y = torch.randn(10, 128, 64, 64)
    # # net1 = TemporalFeatureInteractionModule(128,64)
    # net1 = TFFM(128, 64)
    # x1 = net1(x, y) # torch.Size([1, 64, 16, 16]) torch.Size([10, 64, 64, 64])
    # print(x1.shape)

    # MFAM

    # CIEM
    # # x = torch.randn(1, 4, 2, 2)
    # # y = torch.randn(1, 4, 4, 4)
    # # z = torch.randn(1, 4, 8, 8)
    # # w = torch.randn(1, 4, 16, 16)
    x = torch.randn(4, 64, 128, 128)
    y = torch.randn(4, 64, 64, 64)
    z = torch.randn(4, 64, 32, 32)
    w = torch.randn(4, 64, 16, 16)

    net1 = CIEM(128, 16)
    d5, d4, d3, d2 = net1(x, y, z, w)
    # torch.Size([1, 4, 2, 2]) torch.Size([1, 4, 4, 4]) torch.Size([1, 4, 8, 8]) torch.Size([1, 4, 16, 16])
    # torch.Size([10, 64, 2, 2]) torch.Size([10, 64, 4, 4]) torch.Size([10, 64, 8, 8]) torch.Size([10, 64, 16, 16])
    print(d5.shape, d4.shape, d3.shape, d2.shape)

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
    # x = torch.randn(1, 3, 512, 512)
    # y = torch.randn(1, 3, 512, 512)
    # net1 = BaseNet(6, 2)
    # x1 = net1(x, y) # torch.Size([10, 2, 512, 512])
    # print(x1.shape)


