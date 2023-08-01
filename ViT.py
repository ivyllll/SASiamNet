import torch.nn as nn
import torch
from collections import OrderedDict
from functools import partial
# m = nn.Linear(20, 30)
# input1 = torch.randn(128, 20)
# output = m(input1)
# print(output.size())

def to_2tuple(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)#(224,224)
        patch_size = to_2tuple(patch_size)#(16,16)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  #grid_size=224÷16=14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  #num_patches=14*14

        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #proj使用卷积，embed_dimension这一参数在vision transformer的base16模型用到的是768，所以默认是768。但是如果是large或者huge模型的话embed_dim也会变。
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        #norm_layer默认是None，就是进行nn.Identity()也就是不做任何操作；如果有传入（非None），则会进行初始化一个norm_layer。

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            #assert：进行判断，如果代码模型定义和实际输入尺寸不同则会报错
        x = self.proj(x)  #用卷积实现序列化 torch.Size([4, 768, 14, 14])

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


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
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

        self.num_classes = num_classes  # 复制参数
        self.num_features = self.embed_dim = embed_dim  # 复制参数
        # num_features for consistency with other models

        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # 因为ViT模型的distilled=False，所以前面这三句：
        # num_tokens=1
        # norm_layer=partial(nn.LayerNorm, eps=1e-6)
        # act_layer= nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  # 对图片进行patch和embed

        num_patches = self.patch_embed.num_patches  # 得到patches的个数   #196

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 加上class_token，零矩阵初始化，尺寸1*1*embed_dim.
        # 第一个1是batchsize维度，是为了后面进行拼接所以设置成1。
        # 第二、三个维度就是1*768
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 这一行可以直接忽略，本文（ViT）模型用不到dist_token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))#torch.Size([1, 197, 768])
        # position embedding，使用零矩阵初始化
        # 尺寸为1 *（num_patches + self.num_tokens）* embed_dim   (1,197,768)
        # 第一个维度1是batchsize维度
        # 第二个维度：num_tokens=1（见本段代码第29行），num_patches在base16模型中是14*14=196，加一起就是197
        # 第三个维度：embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 根据传入的drop_path_rate参数（默认为0），for i in的语句使得每一层的drop_path层的drop比率是递增的，但是默认为0，则不创建。
        # stochastic depth decay rule（随机深度衰减规则）

        # 下面利用for i in range(depth)，即根据模型深度depth（默认=12）堆叠Block
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # 不用管下面这一段因为本模型中representation_size=None
        # 前面提过这个参数的作用（本段代码第12行）
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))  # 其实就是全连接层+tanh激活函数
        else:
            self.pre_logits = nn.Identity()

        # 下面就是最终用于分类的全连接层的实现了
        # Classifier head(s)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()
        # 输入向量长度为num_features（定义在本段代码第26行，这个参数值=embed_dim），输出的向量长度为num_classes类别数
        # 下面的部分和ViT无关可以不看
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # self.init_weights(weight_init)

    def forward_features(self, x):
        x = self.patch_embed(x)  # 这个模块第一部分讲了#torch.Size([4, 196, 768])

        cls_token = self.cls_token.expand(x.shape[0], -1,-1)  # 本段第40行附近有解释，原cls_token尺寸1*1*embed_dim，将其在BatchSize维度复制B份，现cls_token尺寸为B*1*embed_dim
        #torch.Size([4, 1, 768])
        if self.dist_token is None:  # 本模型中这个值就是None
            x = torch.cat((cls_token, x), dim=1)#torch.Size([4, 197, 768])

            # 在维度1上进行拼接，即值为196的维度上拼接。本行之后->[B,14*14+1,embed_dim]
        else:  # 本模型不执行这句
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # self.pos_embed-----> torch.Size([1, 197, 768])
        x = self.pos_drop(x + self.pos_embed)  # 加上position embedding再通过51行定义的dropout层  torch.Size([4, 197, 768])

        x = self.blocks(x)  # 通过58行定义的transformer encoder堆叠模块#torch.Size([4, 197, 768])

        x = self.norm(x)  # 通过norm
        # print(x.shape)  #torch.Size([4, 197, 768])
        # print(x[:,0].shape)#torch.Size([4, 768])
        if self.dist_token is None:  # 本模型该参数为None
            return self.pre_logits(x[:, 0])
            # x[:, 0]将class_token通过切片取出，因为拼接的时候放在了最前面
            # 而前面提过pre_logits层在参数representation_size=None的时候返回nn.Identity()即无操作，所以本句输出就是x[:, 0]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # 前向部分
        x = self.forward_features(x)  #
        if self.head_dist is not None:
            # 本模型head_dist=None（81行）所以不执行此分支 不用看
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)  # 直接来到这，head是79行定义的分类头
        return x


if __name__ == '__main__':
    # input = torch.rand(4,3,224,224)
    # input1 = torch.rand(4,3,224,224)
    # pa = Attention(dim=768)
    # pa = VisionTransformer()
    input = torch.rand(4, 196, 768)
    pa = Block(dim=768)
    out = pa(input)
    print(out.shape)